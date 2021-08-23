# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 09:27:43 2021

@author: YingFu
"""

import numpy as np
from os import path
from typing import List, Dict
from tsp_solver.greedy import solve_tsp

import readData as rd
import fy_dp


# =============================================================================
# 用转移矩阵，两两合并最有可能片段，计算最有可能的zone sequence
# =============================================================================

class Segment:
    def __init__(self, nodes, left, right, score, dist):
        self.nodes = nodes  # nodes[i]表示访问的第 i 个节点
        self.left = left  
        self.right = right
        self.dist = dist
        if score == None:
            self.score = 0
            for i in range(len(self.nodes)-1):
                self.score += dist[self.nodes[i],self.nodes[i+1]]
        else:
            self.score = score

    def getHead(self):
        return self.nodes[0]

    def getTail(self):
        return self.nodes[-1]
                
    def append(self, other):
        nodes = self.nodes + other.nodes
        newScore = self.score + other.score + self.dist[self.nodes[-1],other.nodes[0]]
        newS = Segment(nodes, self, other, newScore, self.dist)
        return newS
    
    # def __str__(self):
    #     return f's={self.score}; n={self.nodes}'
    
    def __repr__(self): # 应该是返回一个 eval() 可以重新构造的表达
        return f'({self.score:.2f}: {self.nodes})'
    
    
    def splitAtAllPossiblePosition(self):
        result = []
        leftScore = 0
        nodes = self.nodes
        for i in range(0,len(self.nodes)-2):
            s1 = Segment(nodes[0:i+1], None, None, leftScore, self.dist)
            rightScore = self.score - leftScore - self.dist[nodes[i],nodes[i+1]]
            s2 = Segment(nodes[i+1:], None, None, rightScore, self.dist)
            result.append((s1,s2))
        return result

    def distTo(self, other):
        return self.dist[self.nodes[-1],other.nodes[0]]
    
    def merge(self, other):
        best = self.append(other)
#        print(f"org = {str(best)}")
        
        selfSplits = self.splitAtAllPossiblePosition()
        otherSplits = other.splitAtAllPossiblePosition()
        
        for (s1,s2) in selfSplits:
            for (s3,s4) in otherSplits:
                new = s1.append(s3).append(s2).append(s4)  # 与best的首尾要一样，只是中间打散
#                print(f"{str(best) = }, {new.score = }")
                if new.score < best.score:
                    best = new
#        print(f'{str(best)=}')
        return best

# 通过把片段两两合拼直到只剩下一个完整的路径
# log + merge 方法貌似效果最好 
#                   visitedZoneIdx=[0, 19, 3, 9, 14, 20, 24, 11, 15, 4, 28, 22, 2, 8, 6, 10, 7, 29, 21, 17, 18, 25, 12, 23, 13, 16, 1, 5, 30, 26, 27, 0]
# **log + merge: bestSeg=(-104.23: [0, 17, 18, 19, 3, 9, 14, 20, 24, 11, 15, 28, 4, 22, 2, 8, 6, 5, 30, 26, 10, 7, 29, 21, 25, 12, 23, 13, 16, 1, 27, 0])
# prob + append: bestSeg=(22.26:   [0, 21, 25, 12, 23, 13, 16, 1, 17, 18, 19, 3, 9, 14, 20, 24, 11, 15, 28, 4, 22, 2, 8, 6, 5, 30, 26, 10, 7, 29, 27, 0])
# log + append:  bestSeg=(-105.33: [0, 21, 25, 12, 23, 13, 16, 1, 17, 18, 19, 3, 9, 14, 20, 24, 11, 15, 28, 4, 22, 2, 8, 6, 5, 30, 26, 10, 7, 29, 27, 0])
# prob + merge:  bestSeg=(22.59:   [0, 28, 4, 22, 2, 8, 6, 5, 30, 26, 10, 7, 29, 21, 25, 12, 23, 13, 16, 1, 17, 18, 19, 3, 9, 14, 20, 24, 11, 15, 27, 0])
def greedyMerge(segLst,dist, append=False, debug=False):
    '''
    每次找一对连接概率最大的片断，合成一个新片段，直到剩下一个片段为止。
    合并片段 a,b 有两种方式：
        appned=True: b 简单的连到 a 的尾部
        append=False: 把 a 拆成所有可能的 (a1, a2), b 拆成所有可能的 （b1, b2)
                      尝试重组为 (a1 b1 a2 b2)
    从station出发的片段，只能作为开头与其它片段相连
    到station的片段，只能作为结尾连在其它片段后面
    Parameters
    ----------
    segLst : list[Segment]
        待合并的片段
    probMatrix : numpy.ndarray((n,n), dtype=float)
        probMatrix[i,j] 为 zone i 到 j 的概率。 i 与 j 为 zone 的索引
    debug : bool, optional
        是否打印调试信息 The default is False.

    Returns
    -------
    Segment
        合并后的片段

    '''

    if (debug):
        print(f"    greedyMerge {segLst = }")
    
    while len(segLst)>1:
        bestProb = None
        bestPair = None
        for s1 in segLst:
            if len(s1.nodes) > 1 and s1.getTail() == 0: # 回到 station 的不能作为开头一段
                continue
            for s2 in segLst:
                if s1 == s2:
                    continue
                if len(s2.nodes) > 1 and s2.getHead() == 0: # 从 station 出发的不能作为结尾一段
                    continue
                
                if len(segLst) > 2 and s1.getHead() == 0 and s2.getTail() == 0: # 没到最后，不许station开始与station结束的相连
                    continue
                
                d = s1.distTo(s2)
                if bestProb == None or d < bestProb:
                    bestProb = d
                    bestPair = (s1,s2)
                        
#        print(f'{bestS[0].head = }, {bestS[0].tail = }, {bestS[1].head = }, {bestS[1].tail = }')
        if append:
            mergedSeg = bestPair[0].append(bestPair[1])
        else:
            mergedSeg = bestPair[0].merge(bestPair[1])
        if (debug):
            print(f'    {bestPair[0]=}')
            print(f'    {bestPair[1]=}')
            print(f'    {bestPair[0].distTo(bestPair[1])=}')
            print(f'    {mergedSeg=}')
        segLst.remove(bestPair[0])
        segLst.remove(bestPair[1])
        segLst.append(mergedSeg)
        
        if (debug):
            print(f"    {segLst = }")
            print("---------------------------------------")
                    
    return segLst[0]


# 对矩阵的行进行normalization
#  rows will sum to 1
def matrixNormalizationByRow(matrix):
    norm = np.zeros(matrix.shape)
    for i in range(matrix.shape[0]):
        sum = np.sum(np.abs(matrix[i]))
        if sum != 0:
            norm[i] = matrix[i] / sum
        else:
            norm[i] = matrix[i]
    return norm
    # from sklearn.preprocessing import normalize
    # return normalize(matrix, axis=1, norm='l1')

def negativeLog(probMatrix):
#    normProbMatrix = matrixNormalizationByRow(probMatrix)
    logProb = np.clip(probMatrix,0.00000000000001,None)
    logProb = -np.log(logProb)
    return logProb
    

def getZoneRouteBySegGreedy(route, probMatrix, debug=False):
    '''
    强制把 station 与它能到达的 zone i （即 probMatrix[s,i] > 0, 如果没有这样的zone就用最近的 5 个zone) 连在一起
    强制把能到 station 的 zone j （即 probMatrix[j,s] > 0, 如果没有这样的zone就用可以到 station 的最近的 5 个总额） 连在一起
    尝试用贪心的方法把 [s,i], ...., [j,s] 合并成一条路，尝试所有可能的 i 与 j 找到一条最好的路径

    Parameters
    ----------
    route : readData.Route
        一条路径
    probMatrix : numpy.ndarray((n,n), dtype=float)
        probMatrix[i,j] 为 route.zones[i] 到 route.zones[j] 的转移概率
    debug : bool, optional
        是否打印调试信息. The default is False.

    Returns
    -------
    seg : Segment
        找到的最好 zone 的片段，seg.nodes 为按次序访问的 zone 的索引
    
    (initStartCount, initEndCount, finalStartCount, finalEndCount): (int,int,int,int)
        initStartCount: 从 station 出发以大于 0 概率能到的 zone 的个数； 如果没有概率大于 0 的zone 则用最近的zone替代
        initEndCount: 到 station 的概率大于 0 的 zone的个数
        finalStartCount: 修正后的从 station 出发的 zone
        finalEndCount: 修正后的能到 station 的 zone 的个数
    '''
    dist = negativeLog(probMatrix)
    segLst = [Segment([i],None,None,None,dist) for i in range(0,len(dist))]

    if len(dist) == 1:
        return segLst[0]
    elif len(dist) == 2:  # 只有一个 zone 时， station zone station
        return segLst[0].append(segLst[1]).append(segLst[0]),1,1,1,1
    
    possibleStartSegments = []  # 从 0 出发到一个站， prob > 0
    possibleEndSegments = []    # 从一个站到 0， prob > 0
    for i in range (1,len(dist)):
        if probMatrix[0,i] > 0:
            startSeg = segLst[0].append(segLst[i])
            possibleStartSegments.append(startSeg)
        if probMatrix[i,0] > 0:
            endSeg = segLst[i].append(segLst[0])
            possibleEndSegments.append(endSeg)
    
    initStartCount = len(possibleStartSegments)
    initEndCount = len(possibleEndSegments)
    
    if len(possibleStartSegments) == 0:
        idxOfNeariestZoneFromStation = route.computeIdxOfNeariestZoneFromStation()
        for zoneIdx in idxOfNeariestZoneFromStation:
            startSeg = segLst[0].append(segLst[zoneIdx])
            possibleStartSegments.append(startSeg)
    if len(possibleEndSegments) == 0 or (
            len(possibleStartSegments) == 1 and
            len(possibleEndSegments) == 1 and
            possibleStartSegments[0].getTail() == possibleEndSegments[0].getHead()):
        idxOfNeariestZoneToStation = route.computeIdxOfNeariestZoneToStation()
        for zoneIdx in idxOfNeariestZoneToStation:
            endSeg = segLst[zoneIdx].append(segLst[0])
            possibleEndSegments.append(endSeg)

    finalStartCount = len(possibleStartSegments)
    finalEndCount = len(possibleEndSegments)
    
    if debug:
        print(f'{possibleStartSegments=}')
        print(f'{possibleEndSegments=}')
    
    bestSeg = None
    for startSeg in possibleStartSegments:
        for endSeg in possibleEndSegments:
            if endSeg.getHead() == startSeg.getTail(): # 同一个station不能出现两次
                continue
            
            usedNodes = [0,startSeg.getTail(),endSeg.getHead()]
            if debug:
                print(f'{startSeg=}')
                print(f'{endSeg=}')
                print(f'{usedNodes=}')
            
            otherSeg = [Segment([i],None,None,None,dist) for i in range(0,len(dist)) if i not in usedNodes]
            otherSeg.append(startSeg)
            otherSeg.append(endSeg)
            
            # if debug:
            #     print(f'{otherSeg=}')                

            zoneRoute = greedyMerge(otherSeg,dist,append=False, debug=False)
            if debug:
                print(f'{zoneRoute=}')
                print('----------------')
            if bestSeg == None or zoneRoute.score < bestSeg.score:
                bestSeg = zoneRoute
            
#            raise RuntimeError
    if debug:
        print(f'{bestSeg=}')

    return bestSeg,(initStartCount,initEndCount,finalStartCount,finalEndCount)



# 当一行的概率最大值小于 maxP (即一个zone出发，能到的zone概率比较分散时)
# 把相应的行替换为按照 rank 计算的概率的前 k = 5 个 zone
def redistributeProb(probMatrix, probMatrixByRank, maxP=0.3):
    combined = np.zeros(probMatrix.shape)
    rowMax = np.max(probMatrix,axis=1)
    for i in range(probMatrix.shape[0]):
        if rowMax[i] < maxP:
            print(f'redistribute zone: {i}, maxP: {rowMax[i]}')
            combined[i] = probMatrixByRank[i]
        else:
            combined[i] = probMatrixByRank[i]
    return combined




def tsp_within_zone(stops: List[rd.Stop], start: rd.Stop, end: rd.Stop, dist: rd.DistMatrix) -> List[int]:
    """
    用 Python 的 tsp-solver 计算一组 stop 的最短路。
    :param stops: 要访问的所有 stop 包括 start 与 stop
    :param start: TSP 的起点
    :param end:  TSP 的终点
    :param dist: dist[i,j] 是索引为 i 的 stop 到 j 的距离
    :return: 按访问顺序的 stop 的索引，每个元素是一个 stop 的索引
    """
    stop_idxs = [s.idx for s in stops]
    stop_count = len(stops)
    d: rd.DistMatrix = np.zeros((stop_count, stop_count))
    start_idx = None
    end_idx = None
    for i in range(stop_count):
        if stop_idxs[i] == start.idx:
            start_idx = i
        if stop_idxs[i] == end.idx:
            end_idx = i
        for j in range(stop_count):
            d[i,j] = dist[stop_idxs[i], stop_idxs[j]]

    visited_nodes = solve_tsp(d, endpoints=(start_idx, end_idx))
    return [stop_idxs[i] for i in visited_nodes]


def verify_valid_improvement(stop_idx_list: List[int], new_stop_idx_list: List[int]):
    if len(stop_idx_list) != len(new_stop_idx_list):
        raise RuntimeError(f'{len(stop_idx_list)=} != {len(new_stop_idx_list)=}')
    if stop_idx_list[0] != new_stop_idx_list[0]:
        raise RuntimeError(f'{stop_idx_list[0]=} != {new_stop_idx_list[0]=}, {stop_idx_list=}, {new_stop_idx_list=}')
    if stop_idx_list[-1] != new_stop_idx_list[-1]:
        raise RuntimeError(f'{stop_idx_list[-1]=} != {new_stop_idx_list[-1]=}, {stop_idx_list=}, {new_stop_idx_list=}')
    org_set = set()
    for idx in stop_idx_list:
        org_set.add(idx)
    new_set = set()
    for idx in new_stop_idx_list:
        new_set.add(idx)
    if org_set != new_set:
        raise RuntimeError(f'{org_set=} != {new_set=}, {stop_idx_list=}, {new_stop_idx_list=}')


def tsp_route_to_closest_stop_in_next_zone(route: rd.Route, zone_idx_seq: List[int],
                                           sliding_window_len: int = None,
                                           debug: bool = False) -> List[int]:
    """
    按照访问 zone 的顺序，访问 stops，从 station 开始每次找一条 tsp 路径去到最近的下一个 zone 的 stop。
    假设区域 z1 之后访问区域 z2， 区域z1 中的配送点 e1 到区域 z2 中配送点 s2 是最近的。用 TSP 计算美国总内的配送路径。
    有两个变种：
        1. z1 之内的配送路径必须以 e1 为终点然后到 s2
        2. z1 之内的配送路径可以以 e 为终点（不一定是 e1）然后到达 s2 （本函数采用的方法）
    :param zones: [i] 表示索引为 i 的 zone
    :param zone_idx_seq: [j] = i 表示第 j 个访问的 zone 的索引为
    :param sliding_window_len: None: zone 内的 tsp 是不再优化；整数则为 sliding_dp 的窗口，用 sliding_dp 优化长度
    :param debug: 是否打印调试信息
    :return: 按次序访问的站点的索引
    """
    station, drop_off_stops = route.get_station_and_drop_off_stops()
    zones, zone_id2zone = route.fill_missing_zone_by_knn()

    visited_stop_idxs = []
    cur_zone = zones[zone_idx_seq[0]]  # must be station
    cur_zone_first_stop = station
    for next_zone_idx in zone_idx_seq[1:]:
        next_zone = zones[next_zone_idx]
        cur_zone_last_stop, next_zone_first_stop = \
            cur_zone.nearest_stop_in_next_zone(next_zone, cur_zone_first_stop, route.dist)

        # print(f'tsp for zone: {cur_zone.idx}, start: {cur_zone_first_stop.idx} to end: {cur_zone_last_stop.idx}')
        # print(f'    neariest: (z: {cur_zone.idx}, s: {cur_zone_last_stop.idx}) to (z: {next_zone.idx}, s: {next_zone_first_stop.idx}) ')

        if debug:
            print(f'neariest: (z: {cur_zone.idx}, s: {cur_zone_last_stop.idx}) to (z: {next_zone.idx}, s: {next_zone_first_stop.idx}) ')
            print(f'  tsp from (z: {cur_zone.idx}, s: {cur_zone_first_stop.idx}) to (z: {next_zone.idx}, s: {next_zone_first_stop.idx})')
        stops = cur_zone.stops+[next_zone_first_stop]
        cur_tsp = tsp_within_zone(stops, cur_zone_first_stop, next_zone_first_stop, route.dist)

        # if cur_tsp[0] == 154 and cur_tsp[-1] == 187:
        #     print(f'****************')
        #     print(f'stops: {[s.idx for s in stops]}')
        #     print(f'cur_zone.stops: {[s.idx for s in cur_zone.stops]}')
        #     print(f'next_zone_first_stop: {next_zone_first_stop.idx}')

        if debug:
            print(f'  tsp: {cur_tsp}')
            print(f'  visited_stop_idxs: {visited_stop_idxs}')

        if sliding_window_len is not None:
            improved_tsp = fy_dp.improve_route_by_sliding_DP(cur_tsp, route.dist, windowLen=sliding_window_len)
            if debug:
                print(f'  improved_by_sliding_dp: {improved_tsp}')
                verify_valid_improvement(cur_tsp,improved_tsp)
            cur_tsp = improved_tsp

        visited_stop_idxs.extend(cur_tsp[:-1])

        cur_zone = next_zone
        cur_zone_first_stop = next_zone_first_stop

    last_zone = zones[zone_idx_seq[-1]]
    visited_stop_idxs.append(last_zone.stops[0].idx)

    return visited_stop_idxs


def zone_tsp_solver(route: rd.Route, zone2zone2prob: Dict[str, Dict[str, float]],
                    sliding_window_len: int=None,
                    post_sliding_window_len: int=None,
                    debug: bool = False) -> List[int]:
    """
    用基于历史数据估计的一个zone到下一个zone的概率（zone2zone2prob)，找到 zone 的访问顺序，
    这条路径的概率最大：P(Z1 -> Z2 -> ... -> Z_k) = P (Z_k | Z_(k-1),...,Z_1) * P(Z_(k-1) | Z_(k-2), ..., Z_1) * ... * P(Z_2 | Z_1)
    按照朴素贝叶斯思想，假设 P(Z_i | Z_(i-1), ..., Z_1) = P(Z_i | Z_(i-1))
        P (path) = P(Z_k | Z_(k-1)) * P(Z_(k-1） | Z_(k-2)) * ... * P(Z_2 | Z_1)
    找到最优的路径以后，从一个 zone 到下一个 zone 用最近的两个 stop 连接。从一个 zone 到下一个zone最近的 stop 用 python 的 tsp-solver
    :param route:
    :param zone2zone2prob: Dict[zone_id, Dict[zone_id, float]]. [zid1][zid2] = P(zid2 | zid1)
    :param sliding_window_len: 不是 None 则在找到一个zone内的 tsp 之后，用 sliding DP 的方法改进，建议用 10 或 9
    :param post_sliding_window_len: 不是 None 则在找到全部的 stop 的 tsp 之后用 sliding DP 的方法改进路径的距离, 建议用 9 或 8
    :param debug: True 打印调试信息
    :return: 从 station 出发到 station 结束，依次访问每个配送站点。每个元素是配送点 stop 的索引。
    """
    probMatrix = route.computeHistoricalZoneTransitionMatrix(zone2zone2prob)
    normProbMatrix = matrixNormalizationByRow(probMatrix)
    # zoneDistMatrix = route.computeZoneDistMatrix()

    seqNormProbMatrix,_ = getZoneRouteBySegGreedy(route, normProbMatrix)
    # zones, _ = route.fill_missing_zone_by_knn()

    stop_idx_list = tsp_route_to_closest_stop_in_next_zone(
        route, seqNormProbMatrix.nodes, sliding_window_len=sliding_window_len, debug=debug)

    if post_sliding_window_len is not None:
        improved_tsp = fy_dp.improve_route_by_sliding_DP(stop_idx_list, route.dist, windowLen=post_sliding_window_len)

        if debug:
            print(f'  improved_by_sliding_dp: {improved_tsp}')
            verify_valid_improvement(stop_idx_list, improved_tsp)

        stop_idx_list = improved_tsp
    return stop_idx_list

if __name__ == '__main__':
    BASE_DIR = path.dirname(path.dirname(path.abspath(__file__)))
    buildInputDir = path.join(BASE_DIR, 'data/model_build_inputs')
    buildOutputDir = path.join(BASE_DIR, 'data/model_build_outputs')

    # routeDict = rd.loadOrCreate('./history.pkl', buildInputDir)
    # station2zone2zone2prob,station2zone2rankProb,station2rankProb = loadOrCreate("fy_zoneTSP.pkl", buildInputDir)
    # runExp(routeDict, station2zone2zone2prob, outDir = '../result/zoneTSP')


    