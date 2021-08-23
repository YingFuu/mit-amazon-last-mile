# -*- coding: utf-8 -*-
"""
Created on Sat May 29 15:50:08 2021

@author: iwenc
"""
import random
from typing import List, Dict
import numpy as np

import readData as rd


# encode sets as integers: i'th bit of the integer will represent the state of i'th stop
# (i.e. do we take it in the subset or not).
# For example, 35(10) = (100011)2 will represent stops {1, 2, 6}.

flag = [0]
for i in range(32):
    flag.append(1 << i)
    
# print(f'{flag=}')


def toString(bitset, maxLen=32):
    result = []
    for i in range(1,maxLen):
        if (bitset & flag[i]) != 0:
            result.append(i)
    return result


def bitsetTest():
    bitsetA = 0 # empty set
    
    bitsetA |= flag[1]  # add 1 into bit set
    print(f'add 1 to emptyset: {bitsetA=}: {toString(bitsetA)}')
    
    bitsetA |= flag[3]  # add 3 into bit set
    print(f'after add 3: {bitsetA=}: {toString(bitsetA)}')
    
    bitsetA |= flag[6]  # add 6 into bit set
    print(f'after add 6: {bitsetA=}: {toString(bitsetA)}')
    
    bitsetA |= flag[3]  # add 3 into bit set
    print(f'after add 3: {bitsetA=}: {toString(bitsetA)}')
    
    print(f'remove 1 from {toString(bitsetA)}')
    bitsetA &= (~flag[1]) # remove 1 from bit set if exists
    print(f'  ans: {toString(bitsetA)}')
    
    print(f'remove 2 from {toString(bitsetA)}')
    bitsetA &= (~flag[2]) # remove 2 from bit set if exists
    print(f'  ans: {toString(bitsetA)}')
    
    print(f'3 in {toString(bitsetA)}?: {(bitsetA & flag[3]) != 0}')
        
    a = flag[1] | flag[3] | flag[5] | flag[6]
    b = flag[5] | flag[3] | flag[7]
    c = a & (~b)  # a - b
    print(f'{toString(a)} - {toString(b)} = {toString(c)}, expecting: 1, 6')



def powSet(a = [1,3,6,5]):
    result = [0 for i in range(2**len(a))]
    count = 1  # result[0:1] = {0} empty set
    for i in a:
        nextCount = count * 2;
        for j in range(count, nextCount):
            result[j] = result[j-count] | flag[i]
        count = nextCount
    return result



def tspDP(start, midSet, end, dist):
    
    powSetLen = 2**(end-1)
    startArrSize = (end+1) * powSetLen
    # (start, end, midSet) ==> start * startArrSize + end * powSetLen + midSet
    
    table = [None for i in range(end * startArrSize)]
    
    def f(start, midSet, end):   # 从 start 出发可以用任意次序访问 midSet
        # print(f'f({start},{toString(midSet)},{end})')
        
        idx = start * startArrSize + end * powSetLen + midSet
        
        if table[idx] != None:
            return table[idx]
         
        if midSet == 0:
           table[idx] = (dist[start][end], 0)
           # print(f'-- base: {start}->{end}')
           return table[idx]
        
        mLst = toString(midSet)  # list
        i = mLst[-1]      
        # print(f"{mLst = }")
        # print(f"{i = }")
        midSetExcludeLast = midSet & (~flag[i])  # bitset
        # print(f"{midSetExcludeLast = }")
        # mLstExcludeLast = toString(midSetExcludeLast) # list
        # print(f"{toString(midSetExcludeLast) = }")
        
        minf = None
        minA = None
        for a in powSet(mLst[0:-1]):
            # print(f"{a = }, {toString(a)=}")
            b = midSetExcludeLast & (~a)
            # print(f"{b = }, {toString(b)=}")
            r1,_ = f(start, a, i)
            r2,_ = f(i, b, end)
            r = r1+r2
            if minf==None or r<minf:
                minf = r
                minA = a
            
        table[idx] = (minf, minA)
        return table[idx]

    minf,_ = f(start, midSet, end)        

    # 找到中间的点的顺序，不打印头和尾巴
    def searchPath(start, midSet, end):
        if midSet == 0:
            return (start,)
        
        idx = start * startArrSize + end * powSetLen + midSet
        (minf, minA) = table[idx]
        
        mLst = toString(midSet)  # list
        i = mLst[-1]
        midSetExcludeLast = midSet & (~flag[i])  # bitset
        b = midSetExcludeLast & (~minA)
        return searchPath(start, minA, mLst[-1]) + searchPath(mLst[-1], b, end)
    
    path = searchPath(start, midSet, end)
    return (minf, path+(end,))


def tspDP_seq(stopIdxLst: List[int], routeDist: rd.DistMatrix) -> List[int]:
    """
    计算以 stopIdxLst[0] 为起点 stopIdxList[-1] 为终点，访问 stopIdxLst 中所有站一次的最短路径
        n=10, time=0.0360;
        n=11, time=0.0710;
        n=12, time=0.2230;
        n=13, time=0.4800;
        n=14, time=1.4210
    :param stopIdxLst: 每个元素表示一个 stop 的索引
    :param routeDist: NDArray((n,n), float), [i,j] 表示索引为 i 的 stop 到 j 的距离
    :return: 最短路的距离，及一条具体的最短路
    """
    if len(stopIdxLst) > 31:
        raise RuntimeError(f'DP 最多只能处理 31 个点，无法处理 {len(stopIdxLst)}，从时间及空间考虑请尽量不超过 14 个点')
    if len(stopIdxLst) <= 1:
        return stopIdxLst.copy()

    dist = np.zeros((len(stopIdxLst), len(stopIdxLst)))

    for i in range(len(stopIdxLst)):
        for j in range(len(stopIdxLst)):
            stopI = stopIdxLst[i]
            stopJ = stopIdxLst[j]
            dist[i][j] = routeDist[stopI][stopJ]

    end = len(stopIdxLst)-1
    midSet = 2**(end-1)-1
    d, path = tspDP(0, midSet, end, dist)
    
    result = [stopIdxLst[i] for i in path]
    return d, result


def improve_route_by_sliding_DP(stop_idx_list: List[int], dist: rd.DistMatrix, windowLen: int=10) -> List[int]:
    """
    给定一条 tsp 路径 stop_idx_list。不改变起点与终点，以长度为 windowLen 的窗口，从起点开始每次把窗口内（不包括起点与终点）
    子路径用 DP 替换成最优子路径，每次优化完窗口向右滑动一个单位。
    :param stop_idx_list: 待优化的路径 stop_idx_list[i] = j 表示第 i 个访问的站点的索引为 j
    :param dist: 站点之间的距离矩阵。dist[i,j] 表示索引为 i 的站点到 j 的距离
    :param windowLen: 优化的子路径的长度
    :return: 优化后的路径
    """
    new_stop_idx_list = stop_idx_list.copy()
    for i in range(len(stop_idx_list) - (windowLen - 1)):
        _,r = tspDP_seq(new_stop_idx_list[i:i + windowLen], dist)
        new_stop_idx_list[i:i + windowLen] = r

    return new_stop_idx_list


if __name__ == '__main__':
#    bitsetTest()

    # dist = np.array([[0, 10, 4, 5, 2, 3],
    #                  [10, 0, 7, 9, 2, 1],
    #                  [4, 7, 0, 5, 1, 2],
    #                  [5, 9, 5, 0, 2, 3],
    #                  [2, 2, 1, 2, 0, 2],
    #                  [3, 1, 2, 3, 2, 0]])
    #
    # r = tspDP_seq([0, 1, 2, 4, 3, 5], dist)
    # print(f"{r=}")

    random.seed(3)
    for n in range(10,15):
        import time
        startTime = time.time()
        dist = np.random.rand(n,n)
        stopIdxLst = [i for i in range(n)]
        d, path = tspDP_seq(stopIdxLst, dist)
        totalTime = time.time() - startTime
        print(f'{n=}, {totalTime=:.4f}, {d=:.4f}, {path=}')


