import os
import random
from os import path
import numpy as np
from tsp_solver.greedy import solve_tsp
from typing import Dict, List
import time
import random as rng
from sklearn.model_selection import ShuffleSplit

import fy_score
import readData as rd
import zoneTSP as zt
import holdout_exp

from scipy.spatial import Voronoi
import matplotlib.pyplot as plt


def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.

    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.

    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.

    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1]  # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)


def plot_voronoi(plt, matrix, colorLst=['r', 'g', 'b', 'y', 'c', 'pink', 'purple']):
    '''
    回执一组点作为 site 的 voronoi 图。
    每一个 site 所在的 cell 依次按照 colorLst 中的颜色填充， 循环使用 colorLst

    Parameters
    ----------
    plt : matplotlib.pyplot
        DESCRIPTION.
    matrix : numpy.ndarray((n,2), dtype=float)
        每行代表点的 x 与 y 坐标
    colorLst: list[str]
        一组 matplotlib 的颜色

    '''

    vor = Voronoi(matrix)
    regions, vertices = voronoi_finite_polygons_2d(vor)
    # print(f"{regions = }")
    # print(f"{vertices = }")

    # colorize
    for idx, region in enumerate(regions):
        polygon = vertices[region]
#        print(f"{polygon = }")
        plt.fill(*zip(*polygon), alpha=0.4, c=colorLst[idx % len(colorLst)])

    plt.plot(matrix[:, 0], matrix[:, 1], 'ko')
    plt.xlim(vor.min_bound[0], vor.max_bound[0])
    plt.ylim(vor.min_bound[1], vor.max_bound[1])


def plotVisitedZoneSeq(route, zoneIdxLst, title, fileName=None, skipStation=True):
    if skipStation:
        zoneIdxLst = zoneIdxLst[1:-1]

    zones, _ = route.computeZones()

    # 拿到zones的经度与纬度坐标
    location = np.zeros((len(zoneIdxLst), 2))
    for i in range(len(zoneIdxLst)):
        zoneIdx = zoneIdxLst[i]
        location[i][1], location[i][0] = zones[zoneIdx].computeCenteroid()

    plt.figure(figsize=[20, 16])
    plot_voronoi(plt, location)

    x = []
    y = []
    for zoneIdx in zoneIdxLst:
        zone = zones[zoneIdx]
        lat, lng = zone.computeCenteroid()
        min_start, max_start, min_end, max_end, _ = zone.computeTimeWindowAndServiceTime()
        x.append(lng)
        y.append(lat)
        if zone.hasTimeWindow():
            if min_start < 0:
                min_start = 0
            if max_start < 0:
                max_start = 0
            if abs(max_start - min_start) < 1800:
                startLabel = f'{max_start / 3600:.1f}'
            else:
                startLabel = f'{min_start / 3600:.1f}~{max_start / 3600:.1f}'
            if abs(max_end - min_end) < 1800:
                endLabel = f'{min_end / 3600:.1f}'
            else:
                endLabel = f'{min_end / 3600:.1f}~{max_end / 3600:.1f}'
            label = f'{zoneIdx} [{startLabel},{endLabel}]'
        else:
            label = f'{zoneIdx}'
        plt.annotate(label, (lng, lat))

    plt.scatter(x, y, marker='o', s=100)
    plt.scatter(x[0], y[0], marker='*', s=2000, c='blue')
    plt.scatter(x[-1], y[-1], marker='P', s=2000, c='red')
    plt.plot(x, y, linestyle='solid', linewidth=3)

    plt.ylabel("latitude")
    plt.xlabel("longtitude")
    plt.title(title)
    # plt.legend()
    if fileName != None:
        from os import path
        plt.title(path.basename(fileName))
        dirName = path.dirname(fileName)
        if len(dirName) > 0:
            os.makedirs(dirName, exist_ok=True)
        plt.savefig(fileName)
    else:
        plt.show()
    plt.clf()
    plt.close()
    return None


def plotZoneSeq(route, station2zone2zone2prob, outDir='zoneTSP2'):

    zone2zone2prob = station2zone2zone2prob[route.station_code]
    probMatrix = route.computeHistoricalZoneTransitionMatrix(zone2zone2prob)
    normProbMatrix = zt.matrixNormalizationByRow(probMatrix)

    # 真实zone
    visitedZoneIdx = route.computeVisitedZoneIdxNoDuplicate()
    actualSeqStr = f'{visitedZoneIdx}'.replace(',', '')
    print(f'actual sequence: {actualSeqStr}')
    if outDir != None:
        fileName = f'{outDir}/-actualSeq.png'
    else:
        fileName = None
    plotVisitedZoneSeq(route, visitedZoneIdx, "actualSeq", fileName)

    seqNormProbMatrix ,(initStartCount ,initEndCount ,finalStartCount ,finalEndCount) = zt.getZoneRouteBySegGreedy(route, normProbMatrix, debug=True)
    greedySeqStr = f'{seqNormProbMatrix.nodes}'.replace(',', '')
    print(f' seqNormProbMatrix: {greedySeqStr}')
    if outDir != None:
        fileName = f'{outDir}/seqNormProbMatrix.png'
    else:
        fileName = None
    plotVisitedZoneSeq(route, seqNormProbMatrix.nodes, "normProbMatrixSeq", fileName)


from scipy.spatial import Voronoi
def findStopsInSameStation(routeDict):
    station2Stops  = {}

    for (i ,(rid ,route)) in enumerate(routeDict.items()):
        stationCode = route.station_code
        if stationCode not in station2Stops:
            station2Stops[stationCode] = []

        station2Stops[stationCode].extend(route.stops)



    station2StopsVor = {}
    for stationCode, stops in station2Stops.items():
        stopLocation = np.zeros((len(stops) ,2))
        for i ,stop in enumerate(stops):
            stopLocation[i][0] = stop.lng
            stopLocation[i][1] = stop.lat

        # 去掉重复的stop,得到从同一个station出发的所有stops
        uniqueStopLocation = np.unique(stopLocation, axis=0)
        vor = Voronoi(uniqueStopLocation)
        regions, vertices = rd.voronoi_finite_polygons_2d(vor)
        station2StopsVor[stationCode] = (stopLocation ,regions ,vertices)

    rd.savePkl(station2StopsVor, 'station2StopsVor.pkl')
    print("----------Save Successfully--------------")




def findAdjacentCells(station2StopsVor):
    station2Edge2RegionLst = {}
    for (i, (stationCode, (stopLocation ,regions ,vertices))) in enumerate(station2StopsVor.items()):
        edge2RegionLst = {}
        for idx ,region in enumerate(regions):
            for i in range(len(region)):
                idx1 = region[i]
                idx2 = region[( i +1 ) %len(region)]
                if idx1==idx2:
                    raise RuntimeError

                minIdx = min(idx1 ,idx2)
                maxIdx = max(idx1 ,idx2)
                edge = (minIdx ,maxIdx)
                if edge not in edge2RegionLst:
                    edge2RegionLst[edge] = []

                edge2RegionLst[edge].append(idx)

        station2Edge2RegionLst[stationCode] = edge2RegionLst

    rd.savePkl(station2Edge2RegionLst, 'station2Edge2RegionLst.pkl')
    print("----------Save Successfully--------------")


# 所有 High 数据用来训练，之后又测试训练集的每条 High 的路径，所以结论不可靠
def runExp(routeDict, station2zone2zone2prob, outDir='zoneTSP'):
    import os
    if len(outDir) > 0:
        os.makedirs(outDir, exist_ok=True)

    with open(f'{outDir}/zoneTSP-summary.csv', 'w') as summaryCSV:
        summaryCSV.write(
            'route_idx, route_score, actual zone sequence useFirst, actual zone sequence, greedy zone sequence, score_useFirst, score, time(s), initStartCount,initEndCount,finalStartCount,finalEndCount\n')

        for (i, (rid, route)) in enumerate(routeDict.items()):
            if route.route_score != "High":
                continue
            zone2zone2prob = station2zone2zone2prob[route.station_code]
            probMatrix = route.computeHistoricalZoneTransitionMatrix(zone2zone2prob)
            normProbMatrix = zt.matrixNormalizationByRow(probMatrix)

            zoneDistMatrix = route.computeZoneDistMatrix()
            # zoneDistRank = route.computeZoneDistRank()
            # exploreData(probMatrix, zoneDistMatrix, zoneDistRank)

            # 真实zone
            visitedZoneIdx = route.computeVisitedZoneIdxNoDuplicate(first_occurence=True)
            actualSeqStr = f'{visitedZoneIdx}'.replace(',', '')
            print(f'{i}: actual sequence useFirst: {actualSeqStr}')
            title = f'{outDir}/{i}-actualSeq-useFirst'
            plotVisitedZoneSeq(route, visitedZoneIdx, title=title, fileName=title+'.svg')

            visitedZoneIdx_Longest = route.computeVisitedZoneIdxNoDuplicate(first_occurence=True)
            actualSeqStr_Longest = f'{visitedZoneIdx_Longest}'.replace(',', '')
            print(f'{i}: actual sequence use longest: {actualSeqStr}')
            title = f'{outDir}/{i}-actualSeq'
            plotVisitedZoneSeq(route, visitedZoneIdx, title=title, fileName=title+'.svg')

            import time
            startTime = time.time()
            seqNormProbMatrix, (initStartCount, initEndCount, finalStartCount, finalEndCount) = zt.getZoneRouteBySegGreedy(
                route, normProbMatrix, debug=False)
            endTime = time.time()
            greedySeqStr = f'{seqNormProbMatrix.nodes}'.replace(',', '')
            print(f'{i}: seqNormProbMatrix: {greedySeqStr}')
            title = f'{outDir}/{i}-seqNormProbMatrix'
            plotVisitedZoneSeq(route, seqNormProbMatrix.nodes, title=title, fileName=title+'.svg')

            import fy_score
            s_useFirst, _, _, _ = fy_score.score(seqNormProbMatrix.nodes, visitedZoneIdx, zoneDistMatrix)
            s, _, _, _ = fy_score.score(seqNormProbMatrix.nodes, visitedZoneIdx_Longest, zoneDistMatrix)
            print(f'{i}--{s_useFirst=}, {s=}')

            summaryCSV.write(
                f'{i}, {route.route_score}, {actualSeqStr}, {actualSeqStr_Longest}, {greedySeqStr}, {s_useFirst}, {s}, {endTime - startTime}, {initStartCount}, {initEndCount}, {finalStartCount}, {finalEndCount}\n')
            summaryCSV.flush()



def prob_norm_prob(model, route):
    station2zone2zone2prob,station2zone2rankProb,station2rankProb = model
    zone2zone2prob = station2zone2zone2prob[route.station_code]
    probMatrix = route.computeHistoricalZoneTransitionMatrix(zone2zone2prob)
    return zt.matrixNormalizationByRow(probMatrix)

def prob_norm_cond_rank(model, route):
    station2zone2zone2prob,station2zone2rankProb,station2rankProb = model
    zone2rankProb = station2zone2rankProb[route.station_code]
    conditionalProbMatrixByRank = route.computeHistoricalConditionalZoneTransitionMatrixByRank(zone2rankProb)
    return zt.matrixNormalizationByRow(conditionalProbMatrixByRank)

def prob_norm_uncond_rank(model, route):
    station2zone2zone2prob,station2zone2rankProb,station2rankProb = model
    rankProb = station2rankProb[route.station_code]
    unConditionalrobMatrixByRank = route.computeHistoricalUnconditionalZoneTransitionMatrixByRank(rankProb)
    return zt.matrixNormalizationByRow(unConditionalrobMatrixByRank)

def prob_exp_negative_dist(model, route):
    zoneDistMatrix = route.computeZoneDistMatrix()
    return np.exp(-zoneDistMatrix)

def prob_norm_prob_redistribute(model, route):
    station2zone2zone2prob,station2zone2rankProb,station2rankProb = model
    zone2zone2prob = station2zone2zone2prob[route.station_code]
    probMatrix = route.computeHistoricalZoneTransitionMatrix(zone2zone2prob)
    normProbMatrix = zt.matrixNormalizationByRow(probMatrix)
    zone2rankProb = station2zone2rankProb[route.station_code]
    conditionalProbMatrixByRank = route.computeHistoricalConditionalZoneTransitionMatrixByRank(zone2rankProb)
    normConditionalProbMatrixByRank = zt.matrixNormalizationByRow(conditionalProbMatrixByRank)
    return zt.redistributeProb(normProbMatrix, normConditionalProbMatrixByRank)


def predict_zone_seqs(testDict, model, f, prefix, prob_func=prob_norm_prob, algo='merge'):
    total_score = 0.0
    total_count = 0
    for i, (route_id, route) in enumerate(testDict.items()):
        if route.route_score != "High":
            raise RuntimeError

        station2zone2zone2prob, station2zone2rankProb, station2rankProb = model
        zone2zone2prob = station2zone2zone2prob[route.station_code]
        unseen_zone_count, unseen_zone2zone_count = holdout_exp.count_unseen_zone_and_zone2zone(route, zone2zone2prob)

        # visitedZoneIdx = route.computeVisitedZoneIdxNoDuplicate(first_occurence=True)
        # actualSeqStr = f'{visitedZoneIdx}'.replace(',', '')
        # title = f'{outDir}/{i}-actualSeq-useFirst'
        # plotVisitedZoneSeq(route, visitedZoneIdx, title=title, fileName=title + '.svg')
        visitedZoneIdx_Longest = route.computeVisitedZoneIdxNoDuplicate(first_occurence=False)
        actualSeqStr_Longest = f'{visitedZoneIdx_Longest}'.replace(',', '')

        if hasattr(route, 'probMatrix'):
            delattr(route, 'probMatrix')
        import time

        startTime = time.time()
        normProbMatrix = prob_func(model, route)
        if algo == 'merge':
            seqNormProbMatrix, (initStartCount, initEndCount, finalStartCount, finalEndCount) = zt.getZoneRouteBySegGreedy(
                route, normProbMatrix, debug=False)
            predict_seq = seqNormProbMatrix.nodes
        elif algo == 'solve_tsp':
            predict_seq = solve_tsp(zt.negativeLog(normProbMatrix),endpoints=(0,0))
        endTime = time.time()
        greedySeqStr = f'{predict_seq}'.replace(',', '')
        # print(f'{i}: seqNormProbMatrix: {greedySeqStr}')
        # title = f'{outDir}/{i}-seqNormProbMatrix'
        # plotVisitedZoneSeq(route, seqNormProbMatrix.nodes, title=title, fileName=title+'.svg')

        zoneDistMatrix = route.computeZoneDistMatrix()
        score, _, _, _ = fy_score.score(predict_seq, visitedZoneIdx_Longest, zoneDistMatrix)

        f.write(
            f'{prefix}, {prob_func.__name__}, {algo}, {i}, {unseen_zone_count}, {unseen_zone2zone_count}, {actualSeqStr_Longest}, {greedySeqStr}, {score}, {endTime-startTime}, {route.route_score}\n')
        f.flush()

        total_score += score
        total_count += 1
        print(f'{prefix}, {prob_func.__name__}, {algo}, {i}, {unseen_zone_count}, {unseen_zone2zone_count}, {score:.4f}, avg: {total_score / total_count:.4f}')

    return total_score,total_count

# 用 holdout 的方法进行训练与测试
# 每次用训练集学习统计信息，然后根据统计信息尝试用不同的方法估算 zone 到 zone 的转移概率
# 每个概率矩阵用 merge 或者 tsp （概率取 - log) 之后
def holdout_zone_exp(routeDict, repeat_count=5, test_size=0.2, exe_size = 0.2, file_name='../result/holdOut5_zone_seq.csv'):
    item_list = []
    high_route_dict = {}
    for route_id, route in routeDict.items():
        if route.route_score != 'High':
            continue
        item_list.append((route_id, route))
        high_route_dict[route_id] = route

    rs = ShuffleSplit(n_splits=repeat_count, test_size=test_size, random_state=0)

    output_dir = os.path.dirname(file_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(f'{file_name}', 'w') as f:
        f.write('run_idx, data_set, prob_func, algo, route_idx, unseen_zone_count, unseen_zone2zone_count, act_seq, vit_seq, score, time, route_score\n')

        for run_idx, (train_index, test_index) in enumerate(rs.split(item_list)):
            trainDict = holdout_exp.getSubDict(item_list, train_index)
            testDict = holdout_exp.getSubDict(item_list, test_index)
            station2zone2zone2prob = rd.computeTransMatrix(trainDict)
            station2zone2rankProb = rd.computeZoneToRankTransMatrix(routeDict)
            station2rankProb = rd.computeToRankProb(routeDict)
            model = (station2zone2zone2prob, station2zone2rankProb, station2rankProb)

            exe_count = min(len(test_index), int(len(item_list)*exe_size))

            for prob_func in [prob_norm_prob, prob_norm_cond_rank, prob_norm_uncond_rank, prob_exp_negative_dist]:
                for algo in ['merge', 'solve_tsp']:
                    print(f'{run_idx=}: {len(train_index) = }, {len(test_index) = }')
                    out_sample_test = holdout_exp.getSubDict(item_list, test_index[:exe_count])
                    predict_zone_seqs(out_sample_test, model, f, prefix=f'{run_idx}, test_set',
                                                                 prob_func=prob_func, algo=algo)

                    in_sample_test = holdout_exp.getSubDict(item_list, train_index[:exe_count])
                    predict_zone_seqs(in_sample_test, model, f, prefix=f'{run_idx}, train_set',
                                                                 prob_func=prob_func, algo=algo)





def runExp2(routeDict, station2zone2zone2prob, outDir='../result/stopTSPbyZone'):
    import os
    if len(outDir) > 0:
        os.makedirs(outDir, exist_ok=True)

    with open(f'{outDir}/stopTSPbyZone-summary.csv', 'w') as summaryCSV:
        summaryCSV.write('route_idx, route_scoure, actual sequence, sequence by zone, score, time(s)')
        summaryCSV.write(', sequence by zone (sliding_DP), score (sliding_DP), time(s) (sliding_DP)')
        summaryCSV.write(', sequence by zone (post_sliding_DP), score (post_sliding_DP), time(s) (post_sliding_DP)')
        summaryCSV.write(', tsp seq, tsp score, tsp time(s)\n')
        zone_tsp_score_sum = 0.0
        sliding_score_sum = 0.0
        post_score_sum = 0.0
        route_count = 0
        tsp_score_sum = 0.0
        for (i, (rid, route)) in enumerate(routeDict.items()):
            if route.route_score != "High":
                continue

            # if i < 5:
            #     continue

            _, actual_stop_idx = route.getVisitedStops(computeTime=True, repeatStation=True)
            actual_seq_str = f'{actual_stop_idx}'.replace(',', '')

            startTime = time.time()
            zone2zone2prob = station2zone2zone2prob[route.station_code]
            zone_tsp_visited_stop_idx_list = zt.zone_tsp_solver(route, zone2zone2prob)
            # print(f'route_idx: {i}; {len(zone_tsp_visited_stop_idx_list)=}')
            # print(f'{zone_tsp_visited_stop_idx_list=}')
            route.verify_visit_sequence(zone_tsp_visited_stop_idx_list)
            zone_tsp_visited_stop_idx_list_str = f'{zone_tsp_visited_stop_idx_list}'.replace(',', '')
            zone_tsp_time = time.time() - startTime
            zone_tsp_score, _, _, _ = fy_score.score(actual_stop_idx, zone_tsp_visited_stop_idx_list, route.dist, g=1000)

            startTime = time.time()
            zone2zone2prob = station2zone2zone2prob[route.station_code]
            zone_tsp_sliding_dp_visited_stop_idx_list = zt.zone_tsp_solver(route, zone2zone2prob, sliding_window_len=10)
            if len(zone_tsp_sliding_dp_visited_stop_idx_list) != len(zone_tsp_visited_stop_idx_list):
                print(f'{zone_tsp_visited_stop_idx_list=}')
                print(f'{zone_tsp_sliding_dp_visited_stop_idx_list=}')
                print(f'{len(zone_tsp_visited_stop_idx_list)=}')
                print(f'{len(zone_tsp_sliding_dp_visited_stop_idx_list)=}')
                raise RuntimeError
            route.verify_visit_sequence(zone_tsp_sliding_dp_visited_stop_idx_list)
            zone_tsp_sliding_dp_visited_stop_idx_list_str = f'{zone_tsp_sliding_dp_visited_stop_idx_list}'.replace(',', '')
            zone_tsp_sliding_dp_time = time.time() - startTime
            zone_tsp_sliding_dp_score, _, _, _ = fy_score.score(actual_stop_idx, zone_tsp_sliding_dp_visited_stop_idx_list, route.dist, g=1000)

            startTime = time.time()
            zone2zone2prob = station2zone2zone2prob[route.station_code]
            zone_tsp_post_visited_stop_idx_list = zt.zone_tsp_solver(route, zone2zone2prob, post_sliding_window_len=9)
            if len(zone_tsp_post_visited_stop_idx_list) != len(zone_tsp_visited_stop_idx_list):
                print(f'{zone_tsp_visited_stop_idx_list=}')
                print(f'{zone_tsp_post_visited_stop_idx_list=}')
                print(f'{len(zone_tsp_visited_stop_idx_list)=}')
                print(f'{len(zone_tsp_post_visited_stop_idx_list)=}')
                raise RuntimeError
            route.verify_visit_sequence(zone_tsp_post_visited_stop_idx_list)
            zone_tsp_post_visited_stop_idx_list_str = f'{zone_tsp_post_visited_stop_idx_list}'.replace(',', '')
            zone_tsp_post_time = time.time() - startTime
            zone_tsp_post_score, _, _, _ = fy_score.score(actual_stop_idx, zone_tsp_post_visited_stop_idx_list, route.dist, g=1000)


            startTime = time.time()
            tspVisitedStopIdxs = solve_tsp(route.dist, endpoints=(actual_stop_idx[0], actual_stop_idx[0]))
            route.verify_visit_sequence(tspVisitedStopIdxs)
            tspTime = time.time() - startTime
            tsp_seq_str = f'{tspVisitedStopIdxs}'.replace(',', '')
            tspScore, _, _, _ = fy_score.score(actual_stop_idx, tspVisitedStopIdxs, route.dist, g=1000)

            print(f'{i} zone_tsp t: {zone_tsp_time:.1f}; s: {zone_tsp_score:.4f}; sliding t: {zone_tsp_sliding_dp_time:.1f}; s: {zone_tsp_sliding_dp_score:.4f}; post t: {zone_tsp_post_time:.1f}; s: {zone_tsp_post_score:.4f}; tsp t: {tspTime:.1f}; s: {tspScore:.4f}, {route.route_score}')

            summaryCSV.write(f'{i}, {route.route_score}, {actual_seq_str}')
            summaryCSV.write(f', {zone_tsp_visited_stop_idx_list_str}, {zone_tsp_score}, {zone_tsp_time}')
            summaryCSV.write(f', {zone_tsp_sliding_dp_visited_stop_idx_list_str}, {zone_tsp_sliding_dp_score}, {zone_tsp_sliding_dp_time}')
            summaryCSV.write(f', {zone_tsp_post_visited_stop_idx_list_str}, {zone_tsp_post_score}, {zone_tsp_post_time}')
            summaryCSV.write(f', {tsp_seq_str}, {tspScore}, {tspTime}\n')
            summaryCSV.flush()

            zone_tsp_score_sum += zone_tsp_score
            sliding_score_sum += zone_tsp_sliding_dp_score
            post_score_sum += zone_tsp_post_score
            tsp_score_sum += tspScore
            route_count += 1

    print(f'zone_tsp avg score: {zone_tsp_score_sum/route_count:.4f}; sliding_dp avg score: {sliding_score_sum/route_count:.4f}; post avg score: {post_score_sum/route_count:.4f}; tsp avg: {tsp_score_sum / route_count:.4f}')


def runExp3(routeDict, station2zone2zone2prob, outDir='../result/stopTSPbyZone'):
    import os
    if len(outDir) > 0:
        os.makedirs(outDir, exist_ok=True)

    with open(f'{outDir}/stopTSPbyZone-summary.csv', 'w') as summaryCSV:
        summaryCSV.write('route_idx, route_scoure, actual sequence')
        summaryCSV.write(', sequence by zone (sliding_DP), score (sliding_DP), time(s) (sliding_DP)')
        summaryCSV.write(', tsp seq, tsp score, tsp time(s)\n')
        zone_tsp_score_sum = 0.0
        sliding_score_sum = 0.0
        route_count = 0
        tsp_score_sum = 0.0
        for (i, (rid, route)) in enumerate(routeDict.items()):
            if route.route_score != "High":
                continue

            # if i < 5:
            #     continue

            _, actual_stop_idx = route.getVisitedStops(computeTime=True, repeatStation=True)
            actual_seq_str = f'{actual_stop_idx}'.replace(',', '')

            startTime = time.time()
            zone2zone2prob = station2zone2zone2prob[route.station_code]
            zone_tsp_sliding_dp_visited_stop_idx_list = zt.zone_tsp_solver(route, zone2zone2prob, sliding_window_len=10)
            route.verify_visit_sequence(zone_tsp_sliding_dp_visited_stop_idx_list)
            zone_tsp_sliding_dp_visited_stop_idx_list_str = f'{zone_tsp_sliding_dp_visited_stop_idx_list}'.replace(',', '')
            zone_tsp_sliding_dp_time = time.time() - startTime
            zone_tsp_sliding_dp_score, _, _, _ = fy_score.score(actual_stop_idx, zone_tsp_sliding_dp_visited_stop_idx_list, route.dist, g=1000)

            startTime = time.time()
            tspVisitedStopIdxs = solve_tsp(route.dist, endpoints=(actual_stop_idx[0], actual_stop_idx[0]))
            route.verify_visit_sequence(tspVisitedStopIdxs)
            tspTime = time.time() - startTime
            tsp_seq_str = f'{tspVisitedStopIdxs}'.replace(',', '')
            tspScore, _, _, _ = fy_score.score(actual_stop_idx, tspVisitedStopIdxs, route.dist, g=1000)

            print(f'{i} sliding t: {zone_tsp_sliding_dp_time:.1f}; s: {zone_tsp_sliding_dp_score:.4f}; tsp t: {tspTime:.1f}; s: {tspScore:.4f}, {route.route_score}')

            summaryCSV.write(f'{i}, {route.route_score}, {actual_seq_str}')
            summaryCSV.write(f', {zone_tsp_sliding_dp_visited_stop_idx_list_str}, {zone_tsp_sliding_dp_score}, {zone_tsp_sliding_dp_time}')
            summaryCSV.write(f', {tsp_seq_str}, {tspScore}, {tspTime}\n')
            summaryCSV.flush()

            sliding_score_sum += zone_tsp_sliding_dp_score
            tsp_score_sum += tspScore
            route_count += 1

    print(f'sliding_dp avg score: {sliding_score_sum/route_count}; tsp avg: {tsp_score_sum / route_count}')



def exploreData(probMatrix, zoneDistMatrix, zoneDistRank):
    print('从 i 到 j 有概率的，j 排第几, 从 i 出发的最小距离')
    (row, col) = probMatrix.shape
    for i in range(row):
        import math
        old = zoneDistMatrix[i, i]
        zoneDistMatrix[i, i] = math.inf
        minDist = np.min(zoneDistMatrix[i])
        zoneDistMatrix[i, i] = old
        for j in range(col):
            if probMatrix[i, j] > 0:
                print(
                    f'({i},{j}): p={probMatrix[i, j]:.2f}, rank_of_d={zoneDistRank[i, j]}, min_d={minDist:.4f}, diff_min={zoneDistMatrix[i, j] - minDist:.4f}')

    print('从 i 到最近的 j, j 的概率')
    for i in range(row):
        for j in range(col):
            if abs(zoneDistRank[i, j] - 1) < 0.1:
                print(f'({i},{j}): p={probMatrix[i, j]:.2f}, d={zoneDistMatrix[i, j]:.4f}')


class ZoneSegment:
    def __init__(self, seg_idx, zone_id, zone_idx, stop_count=0):
        self.seg_idx = seg_idx
        self.zone_id = zone_id
        self.zone_idx = zone_idx
        self.stop_count = stop_count

    def __repr__(self):
        return f'ZoneSegment({self.seg_idx}, {self.zone_id}, {self.zone_idx}, {self.stop_count})'

# 把连续是相同 zone 的 stop 合并
# S A A B B A B B B C C A S ==> S(1) A(2) B(2) A(1) B(3) C(2) A(1) S(1)
# 统计最长段不是最早段的zone，比如 B 最早一段为 2 个 stop， 最长一段为 3
# 统计合并后的段数 / unique zone 个数： （8-1） / 4 = 2
def zone_distribution(route: rd.Route, debug=False):
    zones, zone_id2zone = route.computeZones()
    zone_segs = []

    visited_stops, _ = route.getVisitedStops(computeTime=True, repeatStation=True)
    zone_idx_list = [zone_id2zone[s.zone_id].idx for s in visited_stops if s.zone_id is not None]  # 不包括 station
    zone_idx_list = [0] + zone_idx_list + [0]
    current_zone_idx = None
    for zone_idx in zone_idx_list:
        if zone_idx != current_zone_idx:
            zone_seg = ZoneSegment(len(zone_segs), zones[zone_idx].zone_id, zone_idx, 1)
            zone_segs.append(zone_seg)
            current_zone_idx = zone_idx
        else:
            last_seg = zone_segs[-1]
            last_seg.stop_count += 1

    if debug:
        print(f'{zone_segs=}')

    max_seg_len_list = np.zeros(len(zones))
    for zone_seg in zone_segs:
        max_seg_len_list[zone_seg.zone_idx] = max(zone_seg.stop_count, max_seg_len_list[zone_seg.zone_idx])

    if debug:
        print(f'{max_seg_len_list=}')

    first_not_longest_count = 0
    max_diff_longest_first = 0
    counted_zone = np.zeros(len(zones), dtype=bool)
    for zone_seg in zone_segs:
        if counted_zone[zone_seg.zone_idx]:
            continue
        max_seg_len = max_seg_len_list[zone_seg.zone_idx]
        if zone_seg.stop_count < max_seg_len:
            first_not_longest_count += 1
            diff = max_seg_len - first_not_longest_count
            if diff > max_diff_longest_first:
                max_diff_longest_first = diff
        counted_zone[zone_seg.zone_idx] = True

    avg_repeated_segs = float(len(zone_segs) - 1) / len(zones)

    if debug:
        print(f'{first_not_longest_count=}, {len(zone_segs)=}, {len(zones)=}, {avg_repeated_segs=}')

    return first_not_longest_count, max_diff_longest_first, len(zones), len(zone_segs) - 1,

# 按一条路径上一个 zone 包含的 stop 数量统计 zone 的个数
# stop_count <= 10: zone_count: 83.9677 %
# stop_count <= 11: zone_count: 88.4734 %
# stop_count <= 12: zone_count: 91.6717 %
# stop_count <= 13: zone_count: 94.1999 %
# stop_count <= 14: zone_count: 96.0172 %
def num_of_zone_by_stop_count(routeDict: Dict[str, rd.Route], route_score, file) -> Dict[int,int]:
    stop_count2zone_count = {}
    max_stop_count = 0
    for i, (route_id, route) in enumerate(routeDict.items()):
        if route.route_score != route_score:
            continue
        zones, _ = route.computeZones()
        for zone in zones:
            if zone.idx == 0: # skip station
                continue
            stop_count = len(zone.stops)
            if stop_count in stop_count2zone_count:
                stop_count2zone_count[stop_count] += 1
            else:
                stop_count2zone_count[stop_count] = 1
            if stop_count > max_stop_count:
                max_stop_count = stop_count

    accumulative_sum = np.zeros(max_stop_count+1)
    for stop_count in range(1,max_stop_count+1):
        zone_count = stop_count2zone_count.get(stop_count, 0)
        accumulative_sum[stop_count] = accumulative_sum[stop_count-1] + zone_count
    total_zone_count = accumulative_sum[-1]
    print(f'{accumulative_sum/total_zone_count}')
    for n in range(10,15):
        print(f'stop_count <= {n}: zone_count: {100.0*accumulative_sum[n]/total_zone_count:.4f} %')

    with open(f'{file}', 'w') as summaryCSV:
        summaryCSV.write('stop count,zone count,accumulated percentage\n')
        for stop_count in range(1,max_stop_count+1):
            zone_count = stop_count2zone_count.get(stop_count, 0)
            summaryCSV.write(f'{stop_count}, {zone_count}, {float(accumulative_sum[n])/total_zone_count}')

    return stop_count2zone_count, max_stop_count


def useful_stats(outDir='../result/stats'):
    routeDict, station2zone2zone2prob, _, _ = rd.loadOrCreateAll()

    import os
    if len(outDir) > 0:
        os.makedirs(outDir, exist_ok=True)

    route_score_set = set()
    with open(f'{outDir}/route_stats.csv', 'w') as summaryCSV:
        summaryCSV.write('route_idx, route_score, first_not_longest_count, max_diff_longest_first, seg_count, unique_zones, avg_repeated_segs')
        summaryCSV.write(', stop_count\n')
        for (i, (rid, route)) in enumerate(routeDict.items()):
            route_score_set.add(route.route_score)

            first_not_longest_count, max_diff_longest_first, zone_count, seg_count = zone_distribution(route, False)
            avg_repeated_segs = seg_count/zone_count
            print(f'{i}, {first_not_longest_count=}, {max_diff_longest_first=}, {avg_repeated_segs:.2f}')

            summaryCSV.write(f'{i}, {route.route_score}, {first_not_longest_count}, {max_diff_longest_first}, {seg_count}, {zone_count}, {avg_repeated_segs:.2f}')
            summaryCSV.write(f', {len(route.stops)}\n')
            summaryCSV.flush()

    for route_score in route_score_set:
        num_of_zone_by_stop_count(routeDict, route_score, f'{outDir}/stop_count2zone_count_{route_score}.csv');


    station_set = set()
    station2zone_id_set = {}

    for (i, (rid, route)) in enumerate(routeDict.items()):
        if route.route_score != "High":
            continue

        station_set.add(route.station_code)

        if route.station_code in station2zone_id_set:
            zone_id_set = station2zone_id_set.get(route.station_code)
        else:
            zone_id_set = set()
            station2zone_id_set[route.station_code] = zone_id_set

        for s in route.stops:
            if s.zone_id is None:
                continue
            zone_id_set.add(s.zone_id)

    with open(f'{outDir}/station2zone2zone2prob_sparsity.csv', 'w') as f:
        f.write('station, zone_id count, entry count, zone2zone2prob no zero entry count, nonzero ratio\n')
        for s in station_set:
            zone_id_set = station2zone_id_set[s]
            zone2zone2prob = station2zone2zone2prob[s]
            count = 0
            for zid1 in zone_id_set:
                if zid1 not in zone2zone2prob:
                    continue
                zone2prob = zone2zone2prob[zid1]
                for zid2 in zone_id_set:
                    if zid2 in zone2prob:
                        count += 1

            total_count = len(zone_id_set) * len(zone_id_set)
            f.write(f'{s}, {len(zone_id_set)}, {total_count}, {count}, {float(count) / total_count}\n')


def randomly_replace_a_zone() -> rd.RouteDict:
    """
    读入历史数据，把每条 high quality 的 route 的一个随机的 zone 替换成新 zone
    :return: rd.RouteDict: 改过的route
    """
    routeDict = rd.loadOrCreate()

    rng.seed(3)
    new_zone_id = 'zub_fy'

    i: int
    rid: str
    route: rd.Route
    for (i, (rid, route)) in enumerate(routeDict.items()):
        if route.route_score != "High":
            continue

        # if i < 55:
        #     continue

        zone_id2stops: Dict[str, List[rd.Stop]] = {}
        unique_zone_id_list: List[str] = []
        for s in route.stops:
            if s.isDropoff() and s.zone_id is not None:
                # if s.zone_id == 'C-13.1J':
                #     print(f'**** {s.idx=}, {s.zone_id=}')

                if s.zone_id in zone_id2stops:
                    zone_id2stops[s.zone_id].append(s)
                else:
                    zone_id2stops[s.zone_id] = [s]
                    unique_zone_id_list.append(s.zone_id)

        zone_idx = rng.randrange(0, len(unique_zone_id_list))
        selected_zone_id = unique_zone_id_list[zone_idx]
        print(f'route_idx {i}: change zone {selected_zone_id} to {new_zone_id}')
        for s in zone_id2stops[selected_zone_id]:
            # print(f'---- {s.idx}, zone {s.zone_id} ==> {new_zone_id}')
            s.zone_id = new_zone_id

        if hasattr(route, 'zones'):
            raise RuntimeError
        if hasattr(route, 'zones_filled'):
            raise RuntimeError

        zones, zone_id2zones = route.computeZones()  # 用 fill_missing_zone 会把附件没有 zone_id 的加进来
        if selected_zone_id in zone_id2zones:
            raise RuntimeError
        if not (new_zone_id in zone_id2zones):
            raise RuntimeError
        if len(zone_id2zones[new_zone_id].stops) != len(zone_id2stops[selected_zone_id]):
            # print(f' new_zone stops: {[s.idx for s in zone_id2zones[new_zone_id].stops]}')
            raise RuntimeError(f'{len(zone_id2zones[new_zone_id].stops)=} != {len(zone_id2stops[selected_zone_id])=}')

    return routeDict


def randomly_remove_zone_id(k=2) -> rd.RouteDict:
    """
    读入历史数据，把每条 high quality 的 route 的随机 k 个 stop 的 zone_id 删除
    :return: rd.RouteDict: 改过的route
    """
    routeDict = rd.loadOrCreate()

    rng.seed(3)
    new_zone_id = 'zub_fy'

    i: int
    rid: str
    route: rd.Route
    for (i, (rid, route)) in enumerate(routeDict.items()):
        if route.route_score != "High":
            continue

        # if i < 55:
        #     continue

        stop_with_zone_id_list = [s for s in route.stops if s.isDropoff() and s.zone_id is not None]
        stops = rng.choices(stop_with_zone_id_list, k=k)
        for s in stops:
            print(f'route_idx: {i}, remove zone_id for stop_idx: {s.idx}')
            s.zone_id = None

        if hasattr(route, 'zones'):
            raise RuntimeError

    return routeDict



if __name__ == '__main__':
    useful_stats()

    # routeDict, station2zone2zone2prob, station2zone2rankProb, station2rankProb = rd.loadOrCreateAll()
    # runExp(routeDict, station2zone2zone2prob, outDir='../result/zoneTSP')

    # routeDict, station2zone2zone2prob, station2zone2rankProb, station2rankProb = rd.loadOrCreateAll()
    # compute_all_zone_distribution(routeDict)

    # routeDict, station2zone2zone2prob, station2zone2rankProb, station2rankProb = rd.loadOrCreateAll()
    # stop_count2zone_count, max_stop_count = num_of_zone_by_stop_count(routeDict)
    # print(stop_count2zone_count)

    # routeDict, station2zone2zone2prob, station2zone2rankProb, station2rankProb = rd.loadOrCreateAll()
    # runExp2(routeDict, station2zone2zone2prob, outDir='../result/stopTSPbyZone4')

    # newRouteDict = randomly_replace_a_zone()
    # _, station2zone2zone2prob, station2zone2rankProb, station2rankProb = rd.loadOrCreateAll()
    # runExp3(newRouteDict, station2zone2zone2prob, outDir='../result/replaceOneZone')

    # k = 2
    # newRouteDict = randomly_remove_zone_id(k=k)
    # _, station2zone2zone2prob, station2zone2rankProb, station2rankProb = rd.loadOrCreateAll()
    # runExp3(newRouteDict, station2zone2zone2prob, outDir=f'../result/removeZoneId-k={k}')

    # routeDict = rd.loadOrCreate()
    # holdout_zone_exp(routeDict, exe_size=0.2, file_name='../result/holdOut5_zone_seq.csv')

