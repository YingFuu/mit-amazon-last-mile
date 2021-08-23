# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 14:29:28 2021

@author: dhlsf
"""


def gap_sum(path,g):
    '''
    Calculates ERP between two sequences when at least one is empty.
    equal to g*len(A) (B is empty) or g*len(B) (A is empty) 
    
    Parameters
    ----------
    path : list
        Sequence that is being compared to an empty sequence.
    g : int/float
        Gap penalty.

    Returns
    -------
    res : int/float
        ERP between path and an empty sequence.
    '''
    res=0
    for p in path:
        res = res+g
    return res




def dist_erp(p_1,p_2,mat,g=1000):
    '''
    Finds cost between two points. Outputs g if either point is a gap.

    Parameters
    ----------
    p_1 : int
        index of point.
    p_2 : int
        index of other point.
    mat : numpy array
        Normalized cost matrix.
    g : int/float, optional
        Gap penalty. The default is 1000.

    Returns
    -------
    dist : int/float
        Cost of substituting one point for the other.

    '''
    if p_1 == None or p_2 == None:
        dist=g
    else:
        dist=mat[p_1][p_2]
    return dist




def erp_per_edit_helper(actual,sub,matrix,g=1000,memo=None):
    '''
    Calculates ERP and counts number of edits in the process.

    Parameters
    ----------
    actual : list
        Actual route.
    sub : list
        Submitted route.
    matrix : numpy array
        Normalized cost matrix.
    g : int/float, optional
        Gap penalty. The default is 1000.
    memo : dict, optional
        For memoization. The default is None.

    Returns
    -------
    d : float
        ERP from comparing sub to actual.
    count : int
        Number of edits in ERP.

    '''
    if memo==None:
        memo={}
    actual_tuple=tuple(actual)
    sub_tuple=tuple(sub)
    if (actual_tuple,sub_tuple) in memo:
        d,count=memo[(actual_tuple,sub_tuple)]
        return d,count
    
    if len(sub)==0:  
        d=gap_sum(actual,g)
        # d = g*len(actual)
        count=len(actual)
    elif len(actual)==0:
        d=gap_sum(sub,g)
        # d = g*len(sub)
        count=len(sub)
        
    else:
        head_actual=actual[0] 
        head_sub=sub[0] 
        rest_actual=actual[1:]  
        rest_sub=sub[1:]

        score1,count1=erp_per_edit_helper(rest_actual,rest_sub,matrix,g,memo) # 都去头
        score2,count2=erp_per_edit_helper(rest_actual,sub,matrix,g,memo)  # actual去头
        score3,count3=erp_per_edit_helper(actual,rest_sub,matrix,g,memo) # sub去头
        option_1=score1+dist_erp(head_actual,head_sub,matrix,g)
        option_2=score2+dist_erp(head_actual,None,matrix,g)
        option_3=score3+dist_erp(head_sub,None,matrix,g)

        d=min(option_1,option_2,option_3)
        if d==option_1:
            if head_actual==head_sub:
                count=count1
            else:
                count=count1+1
        elif d==option_2:
            count=count2+1
        else:
            count=count3+1
    memo[(actual_tuple,sub_tuple)]=(d,count)
    return d,count



def erp_per_edit_helper_quick(actual,a_start,sub,s_start,matrix,g=1000,memo=None):
    '''
    Calculates ERP and counts number of edits in the process.

    Parameters
    ----------
    actual : list
        Actual route.
    a_start : int
        start index of actual route
    sub : list
        Submitted route.
    s_start : int
        start index of sub route
    matrix : numpy array
        Normalized cost matrix.
    g : int/float, optional
        Gap penalty. The default is 1000.
    memo : dict, optional
        For memoization. The default is None.

    Returns
    -------
    d : float
        ERP from comparing sub to actual.
    count : int
        Number of edits in ERP.

    '''
    if memo==None:
        memo={}
    if (a_start,s_start) in memo:
        return memo[(a_start,s_start)]
    
    if len(sub)==s_start:  
        count=len(actual)-a_start
        d = g * count
    elif len(actual)==a_start:
        count = len(sub) - s_start
        d = g * count
        
    else:
        head_actual=actual[a_start] 
        head_sub=sub[s_start] 

        score1,count1=erp_per_edit_helper_quick(actual,a_start+1,sub,s_start+1,matrix,g,memo)
        score2,count2=erp_per_edit_helper_quick(actual,a_start+1,sub,s_start,matrix,g,memo)
        score3,count3=erp_per_edit_helper_quick(actual,a_start,sub,s_start+1,matrix,g,memo)
        option_1=score1+dist_erp(head_actual,head_sub,matrix,g)
        option_2=score2+dist_erp(head_actual,None,matrix,g)
        option_3=score3+dist_erp(head_sub,None,matrix,g)

        d=min(option_1,option_2,option_3)
        if d==option_1:
            if head_actual==head_sub:
                count=count1
            else:
                count=count1+1
        elif d==option_2:
            count=count2+1
        else:
            count=count3+1
    memo[(a_start,s_start)]=(d,count)
    return d,count


def erp_per_edit_helper_super(actual,sub,matrix,g=1000):
    '''
    Calculates ERP and counts number of edits in the process.

    Parameters
    ----------
    actual : list
        Actual route.
    sub : list
        Submitted route.
    matrix : numpy arrayA
        Normalized cost matrix.
    g : int/float, optional
        Gap penalty. The default is 1000.

    Returns
    -------
    d : float
        ERP from comparing sub to actual.
    count : int
        Number of edits in ERP.

    '''
    
    alen = len(actual)
    slen = len(sub)
    countTab = np.zeros((alen+1,slen+1))
    distTab = np.zeros((alen+1,slen+1))
    for i in range(slen):
        distTab[alen,i] = g
        countTab[alen,i] = 1
    for i in range(alen):
        distTab[i,slen] = g
        countTab[i,slen] = 1
    distTab[alen,slen] = 0
    countTab[alen,slen] = 0

    for r in range(alen-1,-1,-1):    
        for c in range(slen-1,-1,-1):
            score = distTab[r+1,c+1] + matrix[actual[r],sub[c]]
            count = countTab[r+1,c+1]
            if actual[r] != sub[c]:
                count += 1
            score2 = distTab[r+1,c] + g
            count2 = countTab[r+1,c]
            if score2 < score:
                score = score2
                count = count2
            score2 = distTab[r,c+1] + g
            count2 = countTab[r,c+1]
            if score2 < score:
                score = score2
                count = count2
            distTab[r,c] = score
            countTab[r,c] = count
    return distTab[0,0],countTab[0,0]
    


def seq_dev(actual,sub):
    '''
    Calculates sequence deviation.

    Parameters
    ----------
    actual : list, the first and last element both represent station
        Actual route.
    sub : list, the first and last element both represent station
        Submitted route.

    Returns
    -------
    float
        Sequence deviation.

    '''
    actual=actual[1:-1]
    sub=sub[1:-1]
    comp_list=[]
    for i in sub:
        comp_list.append(actual.index(i))
        comp_sum=0
    for ind in range(1,len(comp_list)):
        comp_sum+=abs(comp_list[ind]-comp_list[ind-1])-1
    n=len(actual)
    return (2/(n*(n-1)))*comp_sum


import numpy as np
def normalize_matrix(mat):
    '''
    Normalizes cost matrix.

    Parameters
    ----------
    mat : numpy array
        Cost matrix.

    Returns
    -------
    new_mat : numpy array
        Normalized cost matrix.

    '''
    avg = np.mean(mat)
    std = np.std(mat)
    
    normMat = (mat - avg)/std

    return normMat - np.min(normMat)\
        
        

def erp_per_edit(actual,sub,matrix,g=1000):
    '''
    Outputs ERP of comparing sub to actual divided by the number of edits involved
    in the ERP. If there are 0 edits, returns 0 instead.

    Parameters
    ----------
    actual : list
        Actual route.
    sub : list
        Submitted route.
    matrix : numpy array
        Normalized cost matrix.
    g : int/float, optional
        ERP gap penalty. The default is 1000.

    Returns
    -------
    int/float
        ERP divided by number of ERP edits or 0 if there are 0 edits.

    '''
    total,count=erp_per_edit_helper(actual,sub,matrix,g)
    if count==0:
        return 0
    else:
        return total/count


def score(actual,sub,cost_mat,g=1000):
    '''
    Scores individual routes.

    Parameters
    ----------
    actual : list
        Actual route.
    sub : list
        Submitted route.
    cost_mat : dict
        Cost matrix.
    g : int/float, optional
        ERP gap penalty. Irrelevant if large and len(actual)==len(sub). The
        default is 1000.

    Returns
    -------
    float
        Accuracy score from comparing sub to actual.

    '''
    norm_mat=normalize_matrix(cost_mat)
    seqdev = seq_dev(actual,sub)
    total,count=erp_per_edit_helper(actual,sub,norm_mat,g)
    if count == 0:
        erp = 0
    else:
        erp = total / count
    return seqdev * erp, seqdev, total, count


###############################################################################

import pandas as pd
from tsp_solver.greedy import solve_tsp

from route import insertionBySaving
def score_zwb(routeDict,fileName='../result/score_zwb.csv'):
    tspScore = np.zeros(len(routeDict))
    tspTime = np.zeros(len(routeDict))
    insertionBySavingScore = np.zeros(len(routeDict))
    insertionBySavingTime =  np.zeros(len(routeDict))
    routeScore = []
#    beginTime = time.time()
    for idx,(route_id, route) in enumerate(routeDict.items()):
        routeScore.append(route.route_score)
        visitedStops,visitedStopIdxs = route.getVisitedStops(computeTime = True, repeatStation = True)   
        dist = route.dist   
        import time
        startTime = time.time()
        tspVisitedStopIdxs = solve_tsp(dist,endpoints=(visitedStopIdxs[0],visitedStopIdxs[0]))
        tspTime[idx] = time.time() - startTime
        startTime = time.time()
        zwbSeq = insertionBySaving(route).getVisitingIdx()
        insertionBySavingTime[idx] = time.time() - startTime
        s1,_,_,_ = score(visitedStopIdxs,tspVisitedStopIdxs,dist,g=1000)
        s2,_,_,_ = score(visitedStopIdxs,zwbSeq,dist,g=1000)
        tspScore[idx] = s1
        insertionBySavingScore[idx] = s2
        print(f"{idx=},{s1=},{s2=}")
        # if idx > 5:
        #     break
    
    print(f"tsp avg score: {np.mean(tspScore)}")    
    print(f"insertion avg score: {np.mean(insertionBySavingScore)}")    
    print(f"tsp better count: {np.sum(tspScore < insertionBySavingScore)}")

    df = pd.DataFrame()
    df["tspScore"] = pd.Series(tspScore)
    df["tspTime"] = pd.Series(tspTime)
    df["insertionBySaving"] = pd.Series(insertionBySavingScore)
    df["insertionBySavingTime"] = pd.Series(insertionBySavingTime)
    df["routeScore"] = pd.Series(routeScore)
    df.to_csv(fileName)
    return df
    

from route import insertionByMinDist_reloc_fast
def score_lzx(routeDict,fileName='../result/score_lzx.csv'):
    insertionByMinDistScore = np.zeros(len(routeDict))
    insertionByMinDistScoreTime = np.zeros(len(routeDict))
    routeScore = []
    for idx,(route_id, route) in enumerate(routeDict.items()):
        visitedStops,visitedStopIdxs = route.getVisitedStops(computeTime = True, repeatStation = True)
        routeScore.append(route.route_score)
        import time
        startTime = time.time()
        sub = insertionByMinDist_reloc_fast(route).getVisitingIdx()
        insertionByMinDistScoreTime[idx] = time.time() - startTime
        s,_,_,_ = score(visitedStopIdxs,sub,route.dist,g=1000)
        insertionByMinDistScore[idx] = s
        print(f"{idx=},{s=}, time: {insertionByMinDistScoreTime[idx]}")
    

    df = pd.DataFrame()
    df["insertionByMinDistScore"] = pd.Series(insertionByMinDistScore)
    df["insertionByMinDistScoreTime"] = pd.Series(insertionByMinDistScoreTime)
    df["routeScore"] = pd.Series(routeScore)
    df.to_csv(fileName)
    return df


from route import insertionByMinDist_reduceLatenessAndEarliness_fast
def score_fy(routeDict,fileName='../result/score_fy.csv'):
    insertionByMinDistScore = np.zeros(len(routeDict))
    insertionByMinDistScoreTime = np.zeros(len(routeDict))
    routeScore = []
    for idx,(route_id, route) in enumerate(routeDict.items()):
        visitedStops,visitedStopIdxs = route.getVisitedStops(computeTime = True, repeatStation = True)
        routeScore.append(route.route_score)
        import time
        startTime = time.time()
        sub = insertionByMinDist_reduceLatenessAndEarliness_fast(route).getVisitingIdx()
        insertionByMinDistScoreTime[idx] = time.time() - startTime
        s,_,_,_ = score(visitedStopIdxs,sub,route.dist,g=1000)
        insertionByMinDistScore[idx] = s
        print(f"{idx=},{s=}, time: {insertionByMinDistScoreTime[idx]}")
    

    df = pd.DataFrame()
    df["insertionByMinDistScore_fy"] = pd.Series(insertionByMinDistScore)
    df["insertionByMinDistScoreTime_fy"] = pd.Series(insertionByMinDistScoreTime)
    df["routeScore"] = pd.Series(routeScore)
    df.to_csv(fileName)
    return df



def score_randomSelected(routeDict,fileName='../result/score_random.csv'):
    randomScore = np.zeros(len(routeDict))
    randomScoreTime = np.zeros(len(routeDict))
    routeScore = []
    for idx,(route_id, route) in enumerate(routeDict.items()):
        visitedStops,visitedStopIdxs = route.getVisitedStops(computeTime = True, repeatStation = True)
        randomVisitedIdxs = [visitedStopIdxs[0]]
        shuffleLst = visitedStopIdxs[1:-1]
        import random
        random.seed(2)
        random.shuffle(shuffleLst)
        sub = randomVisitedIdxs + shuffleLst + randomVisitedIdxs
        routeScore.append(route.route_score)
        import time
        startTime = time.time()

        randomScoreTime[idx] = time.time() - startTime
        s,_,_,_ = score(visitedStopIdxs,sub,route.dist,g=1000)
        randomScore[idx] = s
        print(f"{idx=},{s=}, time: {randomScoreTime[idx]}")
     
    df = pd.DataFrame()
    df["randomScore"] = pd.Series(randomScore)
    df["randomScoreTime"] = pd.Series(randomScoreTime)
    df["routeScore"] = pd.Series(routeScore)
    df.to_csv(fileName)
    return df
    

# #from routingAlgo_lzx2 import zone_tsp
# def score_lzx2zoneTSP(routeDict,fileName='../result/score_lzx_zoneTSP.csv'):
#     zoneTSPScore = np.zeros(len(routeDict))
#     zoneTSPScoreTime = np.zeros(len(routeDict))
#     routeScore = []
#     for idx,(route_id, route) in enumerate(routeDict.items()):
#         visitedStops,visitedStopIdxs = route.getVisitedStops(computeTime = True, repeatStation = True)
#         routeScore.append(route.route_score)
#         import time
#         startTime = time.time()
#         sub = zone_tsp(route).getVisitingIdx()
#         zoneTSPScoreTime[idx] = time.time() - startTime
#         s,_,_,_ = score(visitedStopIdxs,sub,route.dist,g=1000)
#         zoneTSPScore[idx] = s
#         print(f"{idx=},{s=}, time: {zoneTSPScoreTime[idx]}")
    

#     df = pd.DataFrame()
#     df["zoneTSPScore"] = pd.Series(zoneTSPScore)
#     df["zoneTSPScoreTime"] = pd.Series(zoneTSPScoreTime)
#     df["routeScore"] = pd.Series(routeScore)
#     df.to_csv(fileName)
#     return df





# def compareSlowAndFast():
#     import readData as rd
#     route10 = rd.loadOrCreateSmallTest()
    
#     from routingAlgo import insertionByMinDist_reduceLatenessAndEarliness
#     from route import insertionByMinDist_reduceLatenessAndEarliness_fast
#     from fy_score import score
    
#     for idx,route in enumerate(route10):
#         visitedStops,visitedStopIdxs = route.getVisitedStops(computeTime = True, repeatStation = True)
#         import time
#         startTime = time.time()
#         sub = insertionByMinDist_reduceLatenessAndEarliness(route).getVisitingIdx()
#         slowTime = time.time() - startTime
#         slowScore,_,_,_ = score(visitedStopIdxs,sub,route.dist,g=1000)
        
#         startTime = time.time()
#         sub = insertionByMinDist_reduceLatenessAndEarliness_fast(route).getVisitingIdx()
#         fastTime = time.time() - startTime
#         fastScore,_,_,_ = score(visitedStopIdxs,sub,route.dist,g=1000)
        
#         print(f'{idx=}, {slowScore=:0.4f}, {fastScore=:0.4f}, {slowTime=:0.2f}, {fastTime=:0.2f}')




if __name__ == "__main__":
    
    from os import path
    import readData as rd
    BASE_DIR = path.dirname(path.dirname(path.abspath(__file__)))
    buildInputDir = path.join(BASE_DIR, 'data/model_build_inputs')
    routeDict = rd.loadOrCreate('./history.pkl', buildInputDir)
    # df = score_zwb(routeDict)
    # df = score_lzx(routeDict)
    # df = score_fy(routeDict)
    # df = score_randomSelected(routeDict)
    # df = score_lzx2zoneTSP(routeDict)
    # compareSlowAndFast()

