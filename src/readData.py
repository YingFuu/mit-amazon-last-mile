# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 18:08:30 2021

关于 station / zone
   1. 把 stop 按照 zone_id 分组，每组内的 lat 与 lng 的范围最大为 17.x 与 51.x
        zone_id max_lat     min_lat     max_lng     min_lng     range_lat   range_lng
        B-8.2D  47.808831   30.200522   -70.791241  -122.3522   17.608309   51.560959
        B-8.1C  47.820279   30.207763   -70.794282  -122.352221 17.612516   51.557939
        B-8.1D  47.802942   30.201807   -70.796796  -122.350537 17.601135   51.553741   
   2. 把 stop 按照 station_code 前3个字母分组，然后再按照 zone_id 分组，每组内 stop 的 lat 与 lng 的范围最大为 0.9x 与 0.9x
        station3    zone_id max_lat     min_lat     max_lng     min_lng     range_lat	range_lng
        DLA         E-1.1D  34.30302    33.390829   -117.595451 -118.517089 0.912191    0.921638
        DLA         E-2.3C  34.325202   33.413202   -117.608757 -118.495348 0.912       0.886591
        DLA         E-2.1C  34.329079   33.420196   -117.609889 -118.471709 0.908883    0.86182
   3. 把 stop 按照 station_code 及 zone_id 分组，每组内的 lat 与 lng 的范围最大为 0.18109 与 0.4149
        station_code    zone_id max_lat     min_lat     max_lng     min_lng     range_lat   range_lng
        DLA9            G-23.1B 33.759168   33.578078   -117.578062 -117.992962 0.18109     0.4149
        DCH2            G-8.3D  42.047251   41.946294   -87.710839  -87.746872  0.100957    0.036033
        DCH2            G-7.2D  42.033582   41.933612   -87.718378  -87.736198  0.09997     0.01782
   4. 把 stop 按照 station_code 及 topZone 分组，每组内的 lat 与 lng 的范围最大为 0.4x 与 0.6x
        station_code    topZone max_lat     min_lat     max_lng     min_lng     range_lat   range_lng
        DBO3            E       42.481387   42.011721   -71.50801   -71.899074  0.469666    0.391064
        DBO3            C       42.122996   41.710285   -71.135987  -71.492818  0.412711    0.356831
        DCH4            B       42.535941   42.140033   -87.805314  -88.412194  0.395908    0.60688
所以 zone_id 是每个 station 自己设置的，两个 station 会有同名的 zone 相聚很远

Route class 表示一条历史路径，一共6112条历史路径，每条路径
1. 出发的 station （仓库，提取货物的地方）
   station_code：是station的名字，一共有17个 DAU1、DBO1、DBO2、DBO3、DCH1、DCH2、DCH3、DCH4、DLA3、DLA4、DLA5、DLA7、DLA8、DLA9、DSE2、DSE4、DSE5
   date: 配送的日期
   depature_time: 从station出发的时间
   executor_capacity: 配送车的容积，单位立方厘米
   route_score: 路径质量，'High', 'Medium', 'Low'
2. 一系列的 Stop （最少33个，最多238个），第一个访问的 stop 一定是 station，每条路只有一个station。每个 stop 记录:
   stop_id: AA、AB、... ZZ 在路径中的标识, data/model_build_inputs/actual_sequences.json
            记录一条路径中的每个 stop_id 被访问的次序。发现 stop_id 在同一条路径中是升序，但是不一定连续。
   lat,lng: GPS 维度与经度,一定不是 none
   stop_type: 'Station' 或者 'Dropoff' （一共有 898,415 个）
   zone_id: Dropoff 的 zone_id 形式:'D-24.2D'。 其中减号之前表示 top level zone。一共有8962个zone，
            20个top level zone: {'RTED3', 'S', 'T', 'M', 'B', 'A', 'E', 'L', 'R', 'P', 'G', 'Q', 'J', 'C', 'D', 'O', 'F', 'LOA1', 'H', 'K'}。
            Station 的 zone_id 一定是 None. 有一些 Dropoff stop 的 zone_id 是 None
                891,900 个有 zone_id
                  6,515 个 None
3. 每个 Dropoff stop 会配送1至多个 package (Station 没有package)，每个 package
   package_id:
   scan_status: 
   start_time,end_time: 期望的配送时间窗口。 要么都为 None 表示没有时间窗，要么都不是 None 且 end_time > start_time。
            1,343,182 个 package 没有时间窗
              113,993 个有时间窗 （最短 3.5 小时，平均 9 小时，最大 24 小时）
        Dropoff stops （共 898,415 个)，其中
              839,243 个，没有时间窗
               38,516 个，有一个时间窗
               18,086 个，有多个相同的时间窗
                2,570 个，有多个不同的时间窗。ratio = (minEnd - maxStart) / (maxEnd - minStart) 重叠部分的时间窗占的比例
                    ratio min: 0.0
                    ratio mean: 0.698
                    ratio max: 0.996
                    ratio < 0.1 count: 7
   planned_service_time_seconds： 最小0.7秒，最大8007秒，平均68.24秒
   depth,height,width:           估计的大小，单位厘米

@author: iwenc
"""
from os import path
import os
import json
import numpy as np
import datetime as dt
import math
import pickle
import copy
import math
import collections

from typing import List, Tuple, Any, Dict
from nptyping import NDArray

DistMatrix = NDArray[(Any,Any),np.float32]
HistTransProbDict = Dict[str, Dict[str, Dict[str, float]]]
ZoneToRankProbDict = Dict[str, Dict[str, Dict[int, float]]]

class Package:
    def __init__(self,package_id,scan_status,start_time,end_time,
                 planned_service_time_seconds,depth,height, width,
                 departure_datetime):
        self.package_id = package_id
        self.scan_status = scan_status
        self.start_time = start_time
        self.end_time = end_time
        self.planned_service_time_seconds = planned_service_time_seconds
        self.depth = depth
        self.height = height
        self.width = width
        
        if self.start_time != None:
            self.start_time_since_departure = (
                self.start_time - departure_datetime).total_seconds()
        else:
            self.start_time_since_departure = None
        if self.end_time != None:
            self.end_time_since_departure = (
                self.end_time - departure_datetime).total_seconds()
        else:
            self.end_time_since_departure = None
    
    def hasTimeWindow(self):
        return self.start_time != None or self.end_time != None
    
    def earliness(self, arrival_time_since_departure_from_station):
        '''
        计算早到了多少。
        如果没有时间窗或者没有早到，返回 None；
        否则返回 self.start_time_since_departure - arrival_time_since_departure_from_station

        Parameters
        ----------
        arrival_time_since_departure_from_station : float
            配送车到达客户的时间

        Returns
        -------
        float
            早到多少秒。如果没有早到，就是 None
        '''
        if self.start_time_since_departure == None:
            return None
        if arrival_time_since_departure_from_station >= self.start_time_since_departure:
            return None
        return self.start_time_since_departure - arrival_time_since_departure_from_station
    
    def lateness(self, arrival_time_since_departure_from_station):
        '''
        计算迟到了多久。
        如果没有时间窗或者没有迟到，返回 None；
        否则返回 arrival_time_since_departure_from_station - self.end_time_since_departure

        Parameters
        ----------
        arrival_time_since_departure_from_station : float
            配送车到达客户的时间

        Returns
        -------
        float
            迟到多少秒。如果没有迟到，就是 None
        '''
        if self.end_time_since_departure == None:
            return None
        if arrival_time_since_departure_from_station <= self.end_time_since_departure:
            return None
        return arrival_time_since_departure_from_station - self.end_time_since_departure
    
    def __str__(self):
        return f'package-id: {self.package_id}; scan-status: {self.scan_status}; time-window: {self.start_time} - {self.end_time}; dimensions (D H W): {self.depth} {self.height} {self.width}'

class Stop:
    def __init__(self,stop_id,lat,lng,stop_type,zone_id):
        self.stop_id = stop_id
        self.lat = lat
        self.lng = lng
        self.stop_type = stop_type  # 'Station' 或者 'Dropoff'
        self.zone_id = zone_id
        if zone_id is None:
            self.topZone = None
        else:
            self.topZone = zone_id.split('-')[0]
        self.packages = []
        
        self.idx = None            # 在 route.stops 中的索引
        self.arrival_time = None   # 到达stop的时间，从仓库出发的时间算起的秒数
        self.departure_time = None # 离开stop的时间，从仓库出发的时间算起的秒数

    
    def copy(self):
        # s = Stop(self.stop_id, self.lat, self.lng, self.stop_type, self.zone_id)
        # s.idx = self.idx
        # s.arrival_time = self.arrival_time
        # s.departure_time = self.departure_time
        # return s
        return copy.copy(self)
    
    def add_package(self, package):
        self.packages.append(package)
    
    def isDropoff(self):
        return self.stop_type == 'Dropoff'
        
    def geoDistTo(self, stop):
        '''
        calculate the distance of two locations given their latitude and longtitudes, respectively.
        
        Haversine_formula
        formula:	a = sin²(Δφ/2) + cos φ1 ⋅ cos φ2 ⋅ sin²(Δλ/2)
        c = 2 ⋅ atan2( √a, √(1−a) )
        d = R ⋅ c
        where	φ is latitude, λ is longitude, R is earth’s radius (mean radius = 6,371km);
       
        Parameters
        ----------
        stop : other stop
        Returns
        -------
        Great cycle distance of two locations on earth, in metres.
    
        '''
        
        lat1 = self.lat
        lng1 = self.lng
        lat2 = stop.lat
        lng2 = stop.lng
        
        Rearth = 637100 # radius of the earth in m
        
        phi1 = lat1*math.pi/180
        phi2 = lat2*math.pi/180
        
        deltaPhi = (lat2-lat1)*math.pi/180
        deltaLambda = (lng2-lng1)*math.pi/180
        
        a = math.sin(deltaPhi/2)*math.sin(deltaPhi/2) + math.cos(phi1)*math.cos(phi2)*math.sin(deltaLambda/2)*math.sin(deltaLambda/2)
        c = 2*math.asin(math.sqrt(a))
        
        return Rearth*c  
    
    def computeTimeWindowAndServiceTime(self):
        '''
        每个stop/station 统计它包含的package的时间窗的
            min_start_time: 最早开始时间
            max_start_time: 最迟开始时间
            min_end_time: 最早结束时间
            max_end_time： 最迟结束时间
        如果没有package或package都没有时间窗，则它们都是None; 否则是从离开 station 算起的秒数
        每个stop 的 total_service_time_seconds 是它包含的 package 的总服务时间（仓库为0）
        Returns
        -------
        float
            min_start_time: 最早开始时间
        float
            max_start_time: 最迟开始时间
        float
            min_end_time: 最早结束时间
        float
            max_end_time： 最迟结束时间
        float
            total_service_time_seconds: 包含的 package 的总服务时间（仓库为0）
        '''
        if not hasattr(self, 'min_start_time'):
            total_service_time_seconds = 0
            hasTW = False
            min_start_time = math.inf
            max_start_time = -math.inf
            min_end_time = math.inf
            max_end_time = -math.inf
            for p in self.packages:
                total_service_time_seconds += p.planned_service_time_seconds
                if not p.hasTimeWindow():
                    continue
                hasTW = True
                if p.start_time_since_departure < min_start_time:
                    min_start_time = p.start_time_since_departure
                if p.start_time_since_departure > max_start_time:
                    max_start_time = p.start_time_since_departure
                if p.end_time_since_departure < min_end_time:
                    min_end_time = p.end_time_since_departure
                if p.end_time_since_departure > max_end_time:
                    max_end_time = p.end_time_since_departure
            if hasTW:
                self.min_start_time = min_start_time
                self.max_start_time = max_start_time
                self.min_end_time = min_end_time
                self.max_end_time = max_end_time
            else:
                self.min_start_time = None
                self.max_start_time = None
                self.min_end_time = None
                self.max_end_time = None
            self.total_service_time_seconds = total_service_time_seconds
        return self.min_start_time,self.max_start_time,self.min_end_time,self.max_end_time,self.total_service_time_seconds
    
    
    def hasTimeWindow(self):
        self.computeTimeWindowAndServiceTime()
        return self.min_start_time != None
    
    
    def __str__(self):
        lst = [str(p) for p in self.packages]
        pStr = ','.join(lst)
        return f'stop-id: {self.stop_id}; lat: {self.lat}; lng: {self.lng}; type: {self.stop_type}; \
zone-id: {self.zone_id}; idx: {self.idx}; arrival_time: {self.arrival_time}; \
departure_time: {self.departure_time}; packages: [{pStr}]'


# return midnight of a given datetime
def midnight(t):
    return dt.datetime(year=t.year,month=t.month,day=t.day,hour=0,minute=0,second=0,microsecond=0,tzinfo=t.tzinfo)
# return one day later of a given datetime
def nextDay(t):
    return t+dt.timedelta(days=1)


class Zone:
    def __init__(self, zone_id, idx, stops):
        self.zone_id = zone_id
        self.idx = idx
        self.stops = stops
    
    def computeDistTo(self, otherZone, dist, k=5):
        '''
        计算 this zone 到 otherZone 之间的距离。

        Parameters
        ----------
        otherZone : Zone
            另外一个 zone
        dist k : int, optional
            The default is 5.

        Returns
        -------
        float
            即从 this.stops 到 otherZone.stops 最距离中最短的 k 个距离的平均值
        '''
        z1Stops = self.stops
        z2Stops = otherZone.stops
        
        distArr = np.zeros(len(z1Stops) * len(z2Stops))
        count = 0
        for s1 in z1Stops:
            for s2 in z2Stops:
                if s1.idx == s2.idx:
                    raise RuntimeError
                distArr[count] = dist[s1.idx,s2.idx]
                count += 1
    
        k = min(k, count)            
        neariestK = np.sort(distArr)[:k]
        return np.mean(neariestK)    

    def computeCenteroid(self):
        '''
        计算每个 zone 包括的 stop 的几何中心

        Returns
        -------
        List[(float,float)]
            [zIdx] = (lat, lng) The centeroid of zone self.uniqueZoneId[zIdx]
        '''
        if not hasattr(self, 'centeroid_lat'):
            if len(self.stops) == 0:
                self.centeroid_lat = None
                self.centeroid_lng = None
            else:                
                latSum = 0
                lngSum = 0
                for s in self.stops:
                    latSum += s.lat
                    lngSum += s.lng
                self.centeroid_lat = latSum/len(self.stops)
                self.centeroid_lng = lngSum/len(self.stops)
        return self.centeroid_lat,self.centeroid_lng
    
    
    def computeTimeWindowAndServiceTime(self):
        '''
        计算每个 zone 包括的 stop 的时间窗口

        Returns
        -------
        List[(float,float)]
            [zIdx] = (minStart,maxStart,minEnd,maxEnd) zone self.uniqueZoneId[zIdx] 的最早、最迟开始时间，最早、最迟结束时间
        '''
        if not hasattr(self, 'min_start_time'):
            total_service_time_seconds = 0
            hasTW = False
            min_start_time = math.inf
            max_start_time = -math.inf
            min_end_time = math.inf
            max_end_time = -math.inf
            for s in self.stops:
                s.computeTimeWindowAndServiceTime()
                total_service_time_seconds += s.total_service_time_seconds
                if not s.hasTimeWindow():
                    continue
                hasTW = True
                if s.min_start_time < min_start_time:
                    min_start_time = s.min_start_time
                if s.max_start_time > max_start_time:
                    max_start_time = s.max_start_time
                if s.min_end_time < min_end_time:
                    min_end_time = s.min_end_time
                if s.max_end_time > max_end_time:
                    max_end_time = s.max_end_time
            if hasTW:
                self.min_start_time = min_start_time
                self.max_start_time = max_start_time
                self.min_end_time = min_end_time
                self.max_end_time = max_end_time
            else:
                self.min_start_time = None
                self.max_start_time = None
                self.min_end_time = None
                self.max_end_time = None
            self.total_service_time_seconds = total_service_time_seconds
        return self.min_start_time,self.max_start_time,self.min_end_time,self.max_end_time,self.total_service_time_seconds

    def hasTimeWindow(self):
        self.computeTimeWindowAndServiceTime()
        return self.min_start_time != None
    
    def nearest_stop_in_next_zone(self, next_zone, exclude_stop: Stop, dist: DistMatrix) -> Stop:
        """
        找到 self 的配送点中（有超过1个时排除 exclude_stop之外) 找一个 s1 及 next_zone 的配送点中的一个 s2。
        s1 到 s2 的距离是所有从 self 的配送点到 next_zone 的配送点中最近的。
        :param next_zone: 下一个要访问的区域
        :param exclude_stop: self.stops 中不能作为 s1 的 stop 的索引
        :param dist: 距离矩阵，[s1.idx,s2.idx] 是 stop s1 到 s2 的距离
        :return: (s1, s2)
        """
        min_dist = math.inf
        best_stop1 = None
        best_stop2 = None
        for s1 in self.stops:
            if len(self.stops) >= 2 and s1.idx == exclude_stop.idx:
                continue
            for s2 in next_zone.stops:
                d = dist[s1.idx, s2.idx]
                if d < min_dist:
                    min_dist = d
                    best_stop1 = s1
                    best_stop2 = s2
        return best_stop1, best_stop2
    
    

def most_frequent(lst: list):
    """
    找到列表中出现次数最多的元素
    :param lst:
    :return:
    """
    # return max(set(List), key = List.count) # 耗时 O(n^2)
    return collections.Counter(lst).most_common(1)[0][0]  # 耗时 O(n log n)

class ZoneSegment:
    def __init__(self, seg_idx, zone_id, zone_idx, stop_count=0):
        self.seg_idx = seg_idx
        self.zone_id = zone_id
        self.zone_idx = zone_idx
        self.stop_count = stop_count

    def __repr__(self):
        return f'ZoneSegment({self.seg_idx}, {self.zone_id}, {self.zone_idx}, {self.stop_count})'

class Route:
    def __init__(self,station_code,departure_datetime,executor_capacity,route_score,invalid_sequence_score):
        self.contact='Ying Fu (dhlsfy@163.com), Zhixin Luo and Wenbin Zhu (i@zhuwb.com), (c) 2021-06-17'
        self.station_code = station_code
#        self.date = date
#        self.departure_time = departure_time
        self.departure_datetime = departure_datetime
        self.executor_capacity = executor_capacity
        self.route_score = route_score
        self.invalid_sequence_score = invalid_sequence_score
        self.stops: List[Stop] = []
        self.visit_order = []
        self.stopId2idx = {}
        self.visitOrder2Idx = {}
        self.dist = np.array([[]])

        self.station = None
        self.drop_off_stops = None
    
    def add_stop(self, stop):
        stop.idx = len(self.stops)
        self.stops.append(stop)
        self.stopId2idx[stop.stop_id] = stop.idx
        
    def add_visit_order(self,visit_order):
        self.visit_order = visit_order
        for idx,vo in enumerate(visit_order):
            self.visitOrder2Idx[vo] = idx
    
    def isHighQuality(self):
        return self.route_score == 'High'
    
    def getVisitedStops(self, computeTime = False, repeatStation = False) -> Tuple[List[Stop], List[int]]:
        '''
        按照访问顺序返回路径上的所有stop
        如果 computeTime = True, 为每个站点计算到达时间和离开时间
            station的离开时间为0，即车辆开始工作的时间，
               之后每个站点的到达时间是前一个站点的离开时间加行使距离
               每个站点的离开时间是到达时间加上处理该站点所有包裹的时间（假设早于时间窗口到达一个站点时不等待）
            station的到达时间是完成所有任务以后车回到station的时间。
            所有到达和离开时间都是从离开station算起的秒数
        如果 repeatStation = True, 会把station做为一个站点插入路径的尾部

        Parameters
        ----------
        computeTime : bool, optional
            是否要计算每个站点的到达与离开时间. The default is False.
        repeatStation : bool, optional
            是否要把station当作最后一个站点插入到路径尾部. The default is False.

        Returns
        -------
        visitedStops : list of Stop. 
            按顺序访问的站点，第一个一定是station
        visitedStopIdxs : list of int. 
            按顺序访问的每个站点的索引
        '''
        visitedStops = [None for i in range(len(self.stops))]
        visitedStopIdxs = [None for i in range(len(self.stops))]
        for stop in self.stops:
            visitIdx = self.visit_order[stop.idx]
            visitedStops[visitIdx] = stop
            visitedStopIdxs[visitIdx] = stop.idx
            stop.visitIdx = visitIdx
        
        if computeTime:
            station = visitedStops[0]
            station.departure_time = 0
            # print(f'{station.departure_time=}')
            for vo in range(1,len(visitedStops)):
                pre = visitedStops[vo-1]
                cur = visitedStops[vo]
                # print(f'{pre.departure_time=}, {self.getDistance(pre,cur)=}')
                arrival_time = pre.departure_time + self.getDistance(pre,cur)
                cur.arrival_time = arrival_time
                
                for package in cur.packages:
                    # print(f'{package.planned_service_time_seconds=}')
                    arrival_time += package.planned_service_time_seconds
                cur.departure_time = arrival_time
            
            last = visitedStops[-1]
            station.arrival_time = last.departure_time + self.getDistance(last, station)
        
        if repeatStation:
            stationCopy = visitedStops[0].copy()
            visitedStops.append(stationCopy)
            visitedStopIdxs.append(stationCopy.idx)
        
        return visitedStops,visitedStopIdxs
    
    def getDatetime(self, secondsAfterDepature):
        return self.departure_datetime + dt.timedelta(seconds=secondsAfterDepature)
    
    def getDistance(self, stopA, stopB):
        return self.dist[stopA.idx, stopB.idx]
    
    def getVisitOrder(self, stop):
        return self.visit_order[stop.idx]

    def getStopByID(self, stop_id):
        idx = self.stopId2idx[stop_id]
        return self.stops[idx]
    
    def getStopByIdx(self, idx):
        return self.stops[idx]
    
    def getStopByVisitOrder(self, visit_order):
        idx = self.visitOrder2Idx[visit_order]
        return self.stops[idx]
    
    def add_dist(self, dist):
        self.dist = dist
    
    def splitStopAndComputeInfo(self):
        '''
        把所有stop分成 station, 有时间窗的stop列表，无时间窗的列表。
        每个stop/station 统计它包含的package的时间窗的
            min_start_time: 最早开始时间
            max_start_time: 最迟开始时间
            min_end_time: 最早结束时间
            max_end_time： 最迟结束时间
        如果没有package或package都没有时间窗，则它们都是None; 否则是从离开 station 算起的秒数
        每个stop 的 total_service_time_seconds 是它包含的 package 的总服务时间（仓库为0）

        Returns
        -------
        station : Stop
            仓库
        twList : List（Stop)
            有时间窗的 stop 列表
        noTWList : TYPE
            无时间窗的 stop 列表

        '''
        twList = []
        noTWList = []
        for s in self.stops:
            s.computeTimeWindowAndServiceTime()
            
            if s.stop_type == 'Station':
                station = s
                continue
            
            if s.hasTimeWindow() != None:
                twList.append(s)
            else:
                noTWList.append(s)
        return station,twList,noTWList

    
    #################################### zone sequence #################################
    # station_code 当作 station 的 zone_id
    # stop.zone_id 是 dropoff stop 的 zone_id
    ####################################################################################
    
    def computeZoneIdSequence(self) -> List[str]:
        """
        把连续是相同 zone 的 stop 合并: S A A B B A B B B C C A S ==> S(1) A(2) B(2) A(1) B(3) C(2) A(1) S(1)
        -------
        list[str]
        :return: self.visitedZones 访问的 zone 的 zone_id
        """
        if not hasattr(self, 'visitedZones'):
            zone_id_seq = [self.station_code]
            visited_stops, _ = self.getVisitedStops(computeTime=True, repeatStation=True)
            zone_id_list = [s.zone_id for s in visited_stops if s.zone_id is not None]
            cur_zone_id = None
            for zone_id in zone_id_list:
                if zone_id != cur_zone_id:
                    zone_id_seq.append(zone_id)
                    cur_zone_id = zone_id
            zone_id_seq.append(self.station_code)
            self.visitedZones = zone_id_seq

        return self.visitedZones


    def computeVisitedZoneIdxNoDuplicate(self, first_occurence:bool = True) -> List[int]:
        """
        把连续是相同 zone 的 stop 合并: S A A B B A B B B C C A S ==> S(1) A(2) B(2) A(1) B(3) C(2) A(1) S(1)
        first_occurence = True: 每个zone（除station以外） 只保留第一访问的
        first_occurence = False: 每个zone 只保留配送点最多的
        -------
        list[int]
        :param first_occurence:
        :return: self.visitedZones 访问的 zone 的 zone_id
        """
        if not hasattr(self, 'visitedZoneIdxNoDuplicate'):
            if first_occurence:
                self.visitedZoneIdxNoDuplicate = self._compute_first_seg_zone_idx_seq()
            else:
                self.visitedZoneIdxNoDuplicate = self._compute_longest_seg_zone_idx_seq()

        return self.visitedZoneIdxNoDuplicate

    def _compute_first_seg_zone_idx_seq(self) -> List[int]:
        visitedStops, _ = self.getVisitedStops(repeatStation=True)
        zones, zoneId2zone = self.computeZones()

        visitedZoneIdxNoDuplicate = []
        zoneIdSet = set()
        for s in visitedStops:
            if s.stop_type == 'Station':
                idx = zoneId2zone[self.station_code].idx
                visitedZoneIdxNoDuplicate.append(idx)
                continue

            if s.zone_id != None and (s.zone_id not in zoneIdSet):
                idx = zoneId2zone[s.zone_id].idx
                visitedZoneIdxNoDuplicate.append(idx)
                zoneIdSet.add(s.zone_id)

        return visitedZoneIdxNoDuplicate

    def _compute_longest_seg_zone_idx_seq(self, debug=False) -> List[str]:
        zones, zone_id2zone = self.computeZones()
        zone_segs = []

        # 把相同 zone_id 的 stop 合并成 ZoneSegment
        visited_stops, _ = self.getVisitedStops(computeTime=True, repeatStation=True)
        zone_idx_list = [zone_id2zone[s.zone_id].idx for s in visited_stops if s.zone_id is not None]  # no station
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

        # 计算每个 zone_idx 对应的最长 seg_len
        max_seg_len_list = np.zeros(len(zones), dtype=int)
        for zone_seg in zone_segs:
            max_seg_len_list[zone_seg.zone_idx] = max(zone_seg.stop_count, max_seg_len_list[zone_seg.zone_idx])

        if debug:
            print(f'{max_seg_len_list=}')

        zone_idx_seq = []
        counted_zone = np.zeros(len(zones), dtype=bool)
        for zone_seg in zone_segs:
            if counted_zone[zone_seg.zone_idx]:
                continue
            max_seg_len = max_seg_len_list[zone_seg.zone_idx]
            if zone_seg.stop_count == max_seg_len:
                zone_idx_seq.append(zone_seg.zone_idx)
                counted_zone[zone_seg.zone_idx] = True
        zone_idx_seq.append(zone_idx_seq[0])

        return zone_idx_seq


    def computeZones(self) -> Tuple[List[Zone], Dict[str, Zone]]:
        '''
        按照 stops 的输入顺序，而不是访问顺序，把路径中出现过的不重复的 zone_id 依次从 0 开始编号 （station_code 总是作为第 0 个)
        把 stops 按 zone_id 分组，按次序创建每个zone

        Returns
        -------
        list[Zone]
            zones[i] = 第 i 个 zone
        Dict[str, Zone]
            zoneId2zone[zone_id]: id 为 zone_id 的 zone
        '''

        if not hasattr(self, 'zones'):
            zoneId2stops = {}    # zone_id 或者 station_code 到 stop lists
            zoneId2stops[self.station_code] = []
            uniqueZoneIdList = [self.station_code]              
            for s in self.stops:
                if s.stop_type == 'Station':
                    zoneId2stops[self.station_code] = [s]
                    continue
                if s.zone_id != None:
                    if s.zone_id in zoneId2stops:
                        zoneId2stops[s.zone_id].append(s)
                    else:
                        zoneId2stops[s.zone_id] = [s]
                        uniqueZoneIdList.append(s.zone_id)                        
            
            zones = []
            zoneId2zone = {}
            for idx,zoneId in enumerate(uniqueZoneIdList):
                stops = zoneId2stops[zoneId]
                zone = Zone(zoneId,idx,stops)
                zones.append(zone)
                zoneId2zone[zoneId] = zone
        
            self.zones = zones
            self.zoneId2zone = zoneId2zone
        return self.zones, self.zoneId2zone
    


    def computeHistoricalZoneTransitionMatrix(self, zone2zone2prob):
        '''
        根据历史统计信息记录从一个 zone 到另一个 zone 的概率
        从一个 zone 出发的所有概率只和 <= 1 (没有 normalize 到 1)
        Parameters
        ----------
        zone2zone2prob : Dict[str, Dict[str,float]]
            zone2zone2prob[zone_id1][zone_id2] = c
            根据历史统计连续访问 zone_id1,zone_id2 的次数占访问 zone_id1 次数的比例

        Returns
        -------
        numpy.array((zoneCount, zoneCount))
            probMatrix[zIdx1,zIdx2] 是 Zone self.zones[zIdx1] 到 self.zones[zIdx2] 历史转移概率

        '''
        if not hasattr(self, 'probMatrix'):
            zones,_ = self.computeZones()
            zoneCount = len(zones)
            probMatrix = np.zeros((zoneCount,zoneCount))
            for i in range(zoneCount):
                zoneId1 = zones[i].zone_id
                for j in range(zoneCount):
                    if i == j:
                        continue
                    zoneId2 = zones[j].zone_id
                    d2prob = zone2zone2prob.get(zoneId1, {})
                    probMatrix[i,j] = d2prob.get(zoneId2, 0)
            self.probMatrix = probMatrix
        return self.probMatrix
        




    def computeZoneDistMatrix(self):
        '''
        计算 zone 到 zone 之间的距离矩阵        

        Returns
        -------
        numpy.array shape=(zoneCount,zoneCount) dtype=float
            self.zoneDistMatrix[i,j] 是从 self.zones[i] 到 self.zones[j] 的距离
        '''
        if not hasattr(self, 'zoneDistMatrix'):
            zones,_ = self.computeZones()
            zoneCount = len(zones)
            zoneDistMatrix = np.zeros((zoneCount,zoneCount))
            for i in range(zoneCount):
                zone1 = zones[i]
                for j in range(zoneCount):
                    if i == j:
                        continue
                    zone2 = zones[j]
                    zoneDistMatrix[i,j] = zone1.computeDistTo(zone2, self.dist)
    
            self.zoneDistMatrix = zoneDistMatrix
        return self.zoneDistMatrix

    def computeIdxOfNeariestZoneFromStation(self, k=5):
        '''
        计算从 statoin 出发到达的最近 k 个zone的索引(不包括 station)

        Parameters
        ----------
        k : int, optional
            The default is 5.

        Returns
        -------
        numpy.ndarray(n, dtype=int)
            idxOfNeariestFromStation[i] = zIdx, 从 station 出发 self.zones[zIdx] 是第 i 近的
        '''
        zones,_ = self.computeZones()
        zoneCount = len(zones)
        distFromStation = np.zeros(zoneCount-1)
        for i in range(1,zoneCount):
            distFromStation[i-1] = zones[0].computeDistTo(zones[i], self.dist)
        count = min(k, zoneCount)
        return (distFromStation.argsort()[:count] + 1)

    def computeIdxOfNeariestZoneToStation(self, k=5):
        '''
        计算能到达 station 的最近 k 个zone的索引(不包括 station)

        Parameters
        ----------
        k : int, optional
            The default is 5.

        Returns
        -------
        numpy.ndarray(n, dtype=int)
            idxOfNeariestToStation[i] = zIdx, self.zones[zIdx] 是到 station 第 i 近的
        '''
        zones,_ = self.computeZones()
        zoneCount = len(zones)
        distToStation = np.zeros(zoneCount-1)
        for i in range(1,zoneCount):
            distToStation[i-1] = zones[i].computeDistTo(zones[0], self.dist)
        count = min(k, zoneCount)
        return (distToStation.argsort()[:count] + 1) 

    def rank(array):
        '''
        计算 array 中每个元素的排序。

        Parameters
        ----------
        array : 
            待排序的数组
        Returns
        -------
        ranks : numpy.ndarray shape=len(array) dtype = int
            ranks[i] = j 表示 array[i] 是第 j 小的元素
        '''
        temp = array.argsort()
        ranks = np.empty_like(temp)
        ranks[temp] = np.arange(len(array))
        return ranks
        
    
    def computeZoneDistRank(self):
        '''
        计算为每对 zone1 到 zone2, 计算 zone2 是 zone1 能到的 zone 中第 几近的。
        Returns
        -------
        numpy.ndarry shape=(zoneCount,zoneCount) dtype = int
            self.zoneDistRank[i,j] = r, 表示从 self.zones[j] 是从 self.zones[i] 出发能到达的第 r 近的 zone
        '''
        if not hasattr(self, 'zoneDistRank'):
            m = self.computeZoneDistMatrix()
            r = np.zeros(m.shape, dtype=int)
            for i in range(m.shape[0]):
                r[i] = Route.rank(m[i])
            self.zoneDistRank = r
        return self.zoneDistRank

    def computeHistoricalConditionalZoneTransitionMatrixByRank(self, zone2rankProb, maxRankKept=5):
        '''
        self.probMatrixByRank[zIdx1,zIdx2] 是 Zone self.uniqueZoneIdList[zIdx1] 到 self.uniqueZoneIdList[zIdx2] 历史转移概率（考虑了 zIdx2 是 zIdx1 第几近）
        Parameters
        ----------
        zone2rankProb : Dict[str,numpy.ndarray(n, dtype=float)]
            zone2rankProb[zid][r] = p 历史统计信息中，从 zone zid 出发下一个 zone 是第 r 近的概率为 p
        maxRankKept : int, optional
            从一个 zone 出发只保留到最近的 maxRankKept zone 的概率。 The default is 5.

        Returns
        -------
        numpy.ndarray((n,n) dtype=float) n 为zone的数量
            probMatrixByRank[i,j] = p 表示 self.zones[i] 到 self.zones[j] 的概率为 p
        '''
        if not hasattr(self, 'probMatrixByRank'):
            zones,_ = self.computeZones()
            zoneDistRank = self.computeZoneDistRank()
    
            zoneCount = len(zones)
            
            probMatrixByRank = np.zeros((zoneCount,zoneCount))
            for i in range(zoneCount):
                zoneId1 = zones[i].zone_id
                rankProb = zone2rankProb.get(zoneId1, np.zeros(0))
                for j in range(zoneCount):
                    if i == j:
                        continue
                    nextRank = zoneDistRank[i,j]
                    if nextRank <= maxRankKept and nextRank < len(rankProb):
                        probMatrixByRank[i,j] = rankProb[nextRank]
            self.probMatrixByRank = probMatrixByRank
        return self.probMatrixByRank
        

    def computeHistoricalUnconditionalZoneTransitionMatrixByRank(self, rankProb, maxRankKept=5):
        '''
        Parameters
        ----------
        rankProb : numpy.ndarray(n, dtype=float)
            rankProb[r] = p: 表示根据历史数据估计的，下一个 zone 为（离上一个zone） 第 r 近的概率为 p
        maxRankKept : int, optional
            从一个 zone 出发只保留到最近的 maxRankKept zone 的概率。 The default is 5.
        Returns
        -------
        numpy.ndarray((n,n) dtype=float) n 为zone的数量
            nextZoneProbByRank[i,j] = rankProb[r]： sefl.zones[j] 为离 self.zones[i] 第 r 近的zone； 
            访问完 self.zones[i] 之后访问 self.zones[j] 的概率为 rankProb[r]
        '''
        if not hasattr(self, 'nextZoneProbByRank'):
            zones,_ = self.computeZones()
            zoneDistRank = self.computeZoneDistRank()
    
            zoneCount = len(zones)
            
            nextZoneProbByRank = np.zeros((zoneCount,zoneCount))
            for zIdx1 in range(zoneCount):
                for zIdx2 in range(zoneCount):
                    if zIdx1 == zIdx2:
                        continue
                    nextRank = zoneDistRank[zIdx1,zIdx2]
                    if nextRank <= maxRankKept and nextRank < len(rankProb):
                        nextZoneProbByRank[zIdx1,zIdx2] = rankProb[nextRank]
            self.nextZoneProbByRank = nextZoneProbByRank
        return self.nextZoneProbByRank

    def get_station_and_drop_off_stops(self) -> Tuple[Stop, List[Stop]]:
        """
        :return: (station, drop_off_stops)
            分别对应一个 station 及 dropoff stop 的列表
        """
        if (self.station is None) or (self.drop_off_stops is None):
            self.drop_off_stops = []  # 所有 dropff stops
            for s in self.stops:
                if s.stop_type == 'Station':
                    self.station = s
                    continue
                self.drop_off_stops.append(s)
        return self.station, self.drop_off_stops

    def fill_missing_zone_by_knn(self, top_k=5) -> Tuple[List[Zone], Dict[str, Zone]]:
        """
        为缺 zone 的 stop 在该路径中找最近 top_k=5个有 zone 的 stop，用这 topK 个最近邻居出现最多的 zone 来填充
        填好的存为 stop.filled_zone_idx

        :param top_k: int, 默认为 5
        :return: Tuple[List[Zone], Dict[str, Zone]]  第一个为 zone 的列表，以 zone.idx 作为索引； 第二个为 zone_id 到 zone 的字典
        """
        if not hasattr(self, 'zones_filled'):
            zones, zone_id2zone = self.computeZones()

            zones_filled = []
            zone_id2zone_filled = {}
            for zone in zones:
                zone_copy = Zone(zone.zone_id, zone.idx, zone.stops.copy())
                zones_filled.append(zone_copy)
                zone_id2zone_filled[zone_copy.zone_id] = zone_copy

            count = min(len(self.stops), top_k)

            stops_with_zone_id = []  # 有 zone_id 的 drop off stops
            for s in self.stops:
                if not s.isDropoff():
                    s.filled_zone_idx = 0
                elif s.zone_id is not None:
                    s.filled_zone_idx = zone_id2zone.get(s.zone_id).idx
                    # print(f'{s.zone_id=} ==> {s.filled_zone_idx}')
                    stops_with_zone_id.append(s)

            for (i, s) in enumerate(self.stops):
                if s.isDropoff() and (s.zone_id is None): # 没有 zone_id 的 drop off stops
                    dist_to_stops_with_zone_id = [
                        self.getDistance(s, next_stop) for next_stop in stops_with_zone_id] # if s != stop
                    nearest_k_stop_idx = np.argsort(np.array(dist_to_stops_with_zone_id))[:top_k]
                    nearest_k_stops = np.array(stops_with_zone_id)[nearest_k_stop_idx]
                    # print(f'first stop: {nearest_k_stop_idx[0]}, {nearest_k_stops[0].stop_id}')
                    nearest_k_zone_idx = [zones[s.filled_zone_idx].idx for s in nearest_k_stops]

                    s.filled_zone_idx = most_frequent(nearest_k_zone_idx)

            for s in self.stops:
                if s.isDropoff() and (s.zone_id is None):  # 没有 zone_id 的 drop off stops
                    zone = zones_filled[s.filled_zone_idx]
                    zone.stops.append(s)

                    # print(f"======= add stop: {s.idx} into zone: {zone.idx}")
                    # print(f'zone.stops: {[s.idx for s in zones[s.filled_zone_idx].stops]}')
                    # print(f'zone_filled.stops: {[s.idx for s in zones_filled[s.filled_zone_idx].stops]}')

            self.zones_filled = zones_filled
            self.zone_id2zone_filled = zone_id2zone_filled

        return self.zones_filled, self.zone_id2zone_filled

    def verify_visit_sequence(self, stop_idxs: List[int]) -> None:
        """
        检查一天路径是否合法（从station出发回到station，访问且只访问每个dropp off stop一次）
        :param stop_idxs:  按访问次序的 stop_idx
        :raise: RuntimeError 如果路径不合法
        """
        if len(stop_idxs) != len(self.stops) + 1:
            used = np.zeros(len(self.stops))
            for i in stop_idxs:
                if i >= len(self.stops):
                    print(f' {i=} >= {len(self.stops)}')
                    continue
                used[i] = 1
            for i in range(len(self.stops)):
                if used[i] < 1:
                    print(f'  missing: {i}')
            raise RuntimeError(f'路径的长度 {len(stop_idxs)} != 总点数: {len(self.stops)} + 1, {stop_idxs=}')
        for i in stop_idxs:
            if i < 0 or i >= len(self.stops):
                raise RuntimeError(f'{stop_idxs[i]=} 超出范围 [0, {len(self.stops)}), {stop_idxs=}')
        if self.stops[stop_idxs[0]].isDropoff():
            raise RuntimeError(f'开始不是station：{str(self.stops[stop_idxs[0]])}, {stop_idxs=}')
        if self.stops[stop_idxs[-1]].isDropoff():
            raise RuntimeError(f'结束不是station：{str(self.stops[stop_idxs[-1]])}, {stop_idxs=}')
        visited_stop_idx_set = set()
        for i in range(1,len(stop_idxs)):
            stop_idx = self.stops[stop_idxs[i]].idx
            if stop_idx in visited_stop_idx_set:
                raise RuntimeError(f'stop_idxs[{i}] = {stop_idxs[i]} 重复访问， , {stop_idxs=}')
            else:
                visited_stop_idx_set.add(stop_idx)
        for i in range(0, len(self.stops)):
            if not i in visited_stop_idx_set:
                raise RuntimeError(f'missing stop_idx: {i}')

    def visitOrderToDict(self, visited_stop_idxs):
        length = len(visited_stop_idxs)
        if visited_stop_idxs[-1] == visited_stop_idxs[0]:
            length -= 1

        sequence = {}
        for idx, stopIdx in enumerate(visited_stop_idxs[0:length]):
            stop_id = self.getStopByIdx(stopIdx).stop_id
            sequence[stop_id] = idx
        return sequence

    def __str__(self):
        lst = [str(s) for s in self.stops]
        stopStr = ','.join(lst)
        return f'station_code: {self.station}; date: {self.date}; depature_time: {self.depature_time}; \
executor_capacity: {self.executor_capacity} cm^3; route_score: {self.route_score}; \
invalid_sequence_score: {self.invalid_sequence_score}; stops: [{stopStr}]; \
visit_order: {self.visit_order}; dist: {self.dist}'


RouteDict = Dict[str, Route]


# in Pythobn 3.7 or later order of key in dict is the same as insertion
def readTrain(datadir,historical=True) -> RouteDict:
    
    # Read input data
    print('Reading Input Data')
    if historical:
        actual_sequences_path=path.join(datadir, 'actual_sequences.json')
        with open(actual_sequences_path, newline='') as in_file:
            actual_sequences = json.load(in_file) # a dictionary of route_id to visiting sequence
        print(f'{len(actual_sequences)=}')
        
        invalid_sequence_scores_path=path.join(datadir, 'invalid_sequence_scores.json')
        with open(invalid_sequence_scores_path, newline='') as in_file:
            invalid_sequence_scores = json.load(in_file) # a dictionary of route_id to invalid_sequence_score
        print(f'{len(invalid_sequence_scores)=}')
    
    if historical:
        package_data_path=path.join(datadir, 'package_data.json')
    else:
        package_data_path=path.join(datadir, 'new_package_data.json')
    with open(package_data_path, newline='') as in_file:
        package_data = json.load(in_file) # a dictionary of route_id to packages delivered
    print(f'{len(package_data)=}')
    
    if historical:
        route_data_path=path.join(datadir, 'route_data.json')
    else:
        route_data_path=path.join(datadir, 'new_route_data.json')        
    with open(route_data_path, newline='') as in_file:
        route_data = json.load(in_file) # a dictionary of route_id to route
    print(f'{len(route_data)=}')
    
    if historical:
        travel_times_path=path.join(datadir, 'travel_times.json')
    else:
        travel_times_path=path.join(datadir, 'new_travel_times.json')        
    with open(travel_times_path, newline='') as in_file:
        travel_times = json.load(in_file) # a dictionary of route_id to distances among stops
    print(f'{len(travel_times)=}')



    routeDict = {}    

    for route_id,route in route_data.items(): # `RouteID_<hex-hash>`: an alphanumeric string that uniquely identifies each route.
        stops = route['stops'] # a dictionary of stop-id to stop
        
        if historical:
            sequences = actual_sequences[route_id] 
            # if len(sequences) != 1:
            #     raise RuntimeError
            # exactly one key 'actual'
            sequence = sequences['actual'] # a dictionary of <stop-id> to <uint-number>
        
            invalid_sequence_score = invalid_sequence_scores[route_id]
        else:
            # 把 station 放到最开始， 其它stop按输入顺序
            sequence = {}
            stationIdx = len(stops)
            for idx,stop_id in enumerate(stops):        
                stop = stops[stop_id]    
                stop_type = stop['type'] # categorical variable denoting the type of stop {`Station` | `Dropoff`}. The delivery vehicle acquires all packages at the station and delivers them at subsequent drop-off locations.
                if stop_type == 'Station':
                    sequence[stop_id] = 0
                    stationIdx = idx
                elif idx < stationIdx:
                    sequence[stop_id] = idx+1
                else:
                    sequence[stop_id] = idx
                            
            invalid_sequence_score = math.nan
        
        packages = package_data[route_id] # a dictionary of <stop-id> to list of packages
        # if sequence.keys() != packages.keys():
        #     print(f'{sequence.keys()=}, {packages.keys()=}')
        #     raise RuntimeError
        # stop-id lists in package_data for a route is eactly the same as sequences
        
        station_code = route["station_code"] # an alphanumeric string that uniquely identifies the delivery station (or depot) at which the route began.
        date = route['date_YYYY_MM_DD']  # <YYYY-MM-DD>, the date the delivery vehicle departed from the station. 
        departure_time = route['departure_time_utc'] # <hh:mm:ss>, the time the delivery vehicle departed from the station, specified in UTC.
        executor_capacity = route['executor_capacity_cm3'] # <unit32>, the volume capacity of the delivery vehicle, specified in cm^3.
        if historical:
            route_score = route['route_score'] # categorical variable denoting the quality of the observed stop sequence {`High` | `Medium` | `Low`}. The quality score is based both on the level of time window adherence and the amount of backtracking in the observed sequence. Backtracking occurs when a delivery vehicle delivers packages within some neighborhood or geographical area, leaves the neighborhood or geographical area, then returns later during the route. Backtracking is inefficient and should be limited when possible.
        else:
            route_score = 'High'
        stops = route['stops'] # a dictionary of stop-id to stop
        # if len(sequence) != len(stops):
        #     print(f'{len(sequence)=}, {len(stops)=}')
        #     raise RuntimeError
        # if sequence.keys() != stops.keys():
        #     print(f'{sequence.keys()=}, {stops.keys()=}')
        #     raise RuntimeError    
        # stop-id lists in route_data for a route is eactly the same as sequences
        routeObj = Route(
            station_code,
            dt.datetime.fromisoformat(' '.join((date,departure_time))),
            #dt.date.fromisoformat(date), dt.time.fromisoformat(departure_time),
            executor_capacity,
            route_score, invalid_sequence_score)
        routeDict[route_id] = routeObj
    
        travel_times_matrix = travel_times[route_id]
        # if sequence.keys() != travel_times_matrix.keys():
        #     print(f'{sequence.keys()=}, {travel_times_matrix.keys()=}')
        #     raise RuntimeError
        # stop-id lists in travel_times for a route is eactly the same as sequences
    
        visit_order = []
        for stop_id, visitIdx in sequence.items(): # <stop-id>, an identifier code for each stop within a route {`AA` | `AB` | ... | `ZZ`}. Stop identifier codes may be shared among routes. Do not assume, however, that stop identifiers shared by multiple routes refer to the same stop. 
            visit_order.append(visitIdx)
            stop = stops[stop_id]    
            lat = stop['lat'] # <float-number>, latitude of stop in WGS 84 projection system
            lng = stop['lng'] # <float-number>, longitude of stop in WGS 84 projection system
            stop_type = stop['type'] # categorical variable denoting the type of stop {`Station` | `Dropoff`}. The delivery vehicle acquires all packages at the station and delivers them at subsequent drop-off locations.
            zone_id = stop['zone_id'] #  a unique identifier denoting the geographical planning area into which the stop falls. The numeral before the dash denotes a high-level planning zone. The text after the dash denotes the subzone within the high-level zone.        
            if isinstance(zone_id, float):
                if math.isnan(zone_id):
                    zone_id = None
                else:
                    raise RuntimeError

            
            stopObj = Stop(stop_id,lat,lng,stop_type,zone_id)
            routeObj.add_stop(stopObj)            
        
            packageLst = packages[stop_id] # a dictionary <PackageId> to package, represents a list of packages delivered to a stop
            # a stop may have more than one packages, for example 
            # route_id='RouteID_00143bdd-0a6b-49ec-bb35-36593d303e77',stop_id='AD', len(packageLst)=3
            # if len(packageLst) != 1:
            #     print(f'{route_id=},{stop_id=}, {len(packageLst)=}')
            #     print(f'{packageLst=}')
            #     raise RuntimeError
            for packageId, package in packageLst.items():
                if historical:
                    scan_status = package['scan_status'] # categorical variable denoting the delivery status of a package {`DELIVERED` | `DELIVERY_ATTEMPTED` | `REJECTED`}. If a package’s delivery was attempted but not successful, delivery may be reattempted later in the route. 
                else:
                    scan_status = None
                time_window = package['time_window'] # the interval of time in which package delivery is acceptable, defined by 
                start_time = time_window['start_time_utc'] # <YYYY-MM-DD hh:mm:ss>, `NaN`, no time window was specified. 
                end_time = time_window['end_time_utc'] # <YYYY-MM-DD hh:mm:ss>`NaN`, no time window was specified. 
                planned_service_time_seconds = package['planned_service_time_seconds'] # <uinit-number>, The duration of time expected to deliver the package once the delivery person has arrived at the package’s delivery location, specified in seconds. Service time may include time required to park and hand-off the package at the drop-off location. 
                dimensions = package['dimensions'] # the approximate depth, height, and width of the package {`depth_cm`, `height_cm`, and `width_cm`}, specified in centimeters.
                depth_cm = dimensions['depth_cm'] # <float-number>
                height_cm = dimensions['height_cm'] # <float-number>
                width_cm = dimensions['width_cm'] # <float-number>
    
                def parseDatetime(t):
                    if isinstance(t, float):
                        if math.isnan(t):
                            return None
                        raise RuntimeError
                    return dt.datetime.fromisoformat(t)
                
                packageObj = Package(packageId,scan_status,
                                     parseDatetime(start_time),
                                     parseDatetime(end_time),
                                     planned_service_time_seconds,
                                     depth_cm,height_cm,width_cm,
                                     routeObj.departure_datetime)
                stopObj.add_package(packageObj)
        
        routeObj.add_visit_order(visit_order)
    
        dist = np.zeros((len(stops), len(stops)))
        
        for src_id, stop_travel_times in travel_times_matrix.items(): # <stop-id>, an identifier code for each stop within a route {`AA` | `AB` | ... | `ZZ`}. Stop identifier codes may be shared among routes. Do not assume, however, that stop identifiers shared by multiple routes refer to the same stop. 
            srcIdx = routeObj.stopId2idx[src_id]
            for dest_id, d in stop_travel_times.items():
                destIdx = routeObj.stopId2idx[dest_id]
                dist[srcIdx,destIdx] = d
    
        routeObj.add_dist(dist)         
    
    return routeDict


# =============================================================================
# 统计历史信息
# =============================================================================

# station2Routes[s]: 表示从 station s出发的高质量路径的列表
def groupRouteByStation(routeDict) -> Dict[str, List[Route]]:
    station2Routes = {}
    for i, (route_id, route) in enumerate(routeDict.items()):
        if route.route_score == "High":
            if route.station_code in station2Routes.keys():
                station2Routes[route.station_code].append(route)
            else:
                station2Routes[route.station_code] = [route]

            if (i % 100 == 0):
                print(f"current route = {i}")
    return station2Routes


# station2zone2zone2prob[s][zoneId1][zoneId2] = p, 三级嵌套的字典
# 表示根据 station s 的高质量历史路径统计
#    zoneId1 的下一个 zone 是 zoneId2 的概率为 p
def computeTransMatrix(routeDict) -> HistTransProbDict:
    station2Routes = groupRouteByStation(routeDict)
    station2zone2zone2prob = {}
    for (station, routeLst) in station2Routes.items():
        zone2zone2count = {}  # 统计 (zone1,zone2) 出现的次数
        startZone2count = {}  # 每个 start zone 出现的次数
        for r, route in enumerate(routeLst):
            zoneIdSeq = route.computeZoneIdSequence()
            for z in range(len(zoneIdSeq) - 1):
                startZoneId = zoneIdSeq[z]
                nextZoneId = zoneIdSeq[z + 1]
                if startZoneId in zone2zone2count:
                    zone2count = zone2zone2count[startZoneId]
                else:
                    zone2count = {}  # 从start zone出发，每个 destination zone 出现的次数
                    zone2zone2count[startZoneId] = zone2count

                zone2count[nextZoneId] = zone2count.get(nextZoneId, 0) + 1
                startZone2count[startZoneId] = startZone2count.get(startZoneId, 0) + 1

            # if r % 100 == 0:
            #     print(f"current route: {r}")

        for startZone in startZone2count:
            zone2count = zone2zone2count[startZone]
            total = startZone2count[startZone]
            for nextZone in zone2count:
                zone2count[nextZone] = zone2count[nextZone] / total
        #            station2zone2zone2prob[startZone] = dict(sorted(zone2count.items(), key=lambda item: item[1],reverse=True))

        station2zone2zone2prob[station] = zone2zone2count

    # [s][z1][z2]: station s 中 zone z1 的后继是 zone z2 的概率
    return station2zone2zone2prob


# station2zone2rankProb[s][zoneId1][rank] = p,  两级嵌套的字典，里面存 np.array
# 表示根据 station s 的高质量历史路径统计
#    zoneId1 的下一个 zone 是离它第 rank 近的概率为 p
def computeZoneToRankTransMatrix(routeDict) -> ZoneToRankProbDict:
    station2Routes = groupRouteByStation(routeDict)
    station2zone2rankProb = {}
    for (station, routeLst) in station2Routes.items():
        zone2rank2count = {}  # 统计 (zone1,zone2) 出现的次数
        startZone2count = {}  # 每个 start zone 出现的次数
        for r, route in enumerate(routeLst):
            zoneIdSeq = route.computeZoneIdSequence()
            zoneDistRank = route.computeZoneDistRank()  # [zIdx1][zIdx2] = rank

            zones, zoneId2zone = route.computeZones()
            zoneIdx = [zoneId2zone[zid].idx for zid in zoneIdSeq]

            for z in range(len(zoneIdSeq) - 1):
                startZoneId = zoneIdSeq[z]
                if startZoneId in zone2rank2count:
                    rank2count = zone2rank2count[startZoneId]
                else:
                    rank2count = {}  # 从start zone出发，每个 destination zone 出现的次数
                    zone2rank2count[startZoneId] = rank2count

                nextRank = zoneDistRank[zoneIdx[z], zoneIdx[z + 1]]
                rank2count[nextRank] = rank2count.get(nextRank, 0) + 1
                startZone2count[startZoneId] = startZone2count.get(startZoneId, 0) + 1

            # if r % 100 == 0:
            #     print(f"current route: {r}")

        zone2rankProb = {}
        for startZone in startZone2count:
            rank2count = zone2rank2count[startZone]
            total = startZone2count[startZone]

            maxKey = int(max(rank2count.keys()))
            rankProb = np.zeros(maxKey + 1)
            for rank in rank2count:
                rankProb[int(rank)] = rank2count[rank] / total
            zone2rankProb[startZone] = rankProb

        station2zone2rankProb[station] = zone2rankProb

    # [s][z1][z2]: station s 中 zone z1 的后继是 zone z2 的概率
    return station2zone2rankProb


# station2rankProb[s][rank] = p, 字典中存 np.array
# 表示根据 station s 的高质量历史路径统计
#    下一个 zone 是离上一个zone第 rank 近的概率为 p
def computeToRankProb(routeDict) -> Dict[str, Dict[int, float]]:
    station2Routes = groupRouteByStation(routeDict)
    station2rankProb = {}
    for (station, routeLst) in station2Routes.items():
        rank2count = {}  # 统计 (zone1,zone2) 出现的次数
        pairCount = 0;
        for r, route in enumerate(routeLst):
            zoneIdSeq = route.computeZoneIdSequence()
            zoneDistRank = route.computeZoneDistRank()  # [zIdx1][zIdx2] = rank

            _, zoneId2zone = route.computeZones()
            zoneIdx = [zoneId2zone[zid].idx for zid in zoneIdSeq]

            for z in range(len(zoneIdSeq) - 1):
                nextRank = zoneDistRank[zoneIdx[z], zoneIdx[z + 1]]
                rank2count[nextRank] = rank2count.get(nextRank, 0) + 1
                pairCount += 1

            # if r % 100 == 0:
            #     print(f"current route: {r}")

        maxKey = int(max(rank2count.keys()))
        rankProb = np.zeros(maxKey + 1)
        for rank in rank2count:
            rankProb[int(rank)] = rank2count[rank] / pairCount

        station2rankProb[station] = rankProb

    # [s][z1][z2]: station s 中 zone z1 的后继是 zone z2 的概率
    return station2rankProb


def savePkl(obj, fileName):
    dirName = path.dirname(fileName)
    if len(dirName) > 0:
        os.makedirs(dirName, exist_ok=True)
    with open(fileName, "wb") as file:
        pickle.dump(obj,file)

def loadPkl(fileName):
    with open(fileName, "rb") as file:
        return pickle.load(file)


##############################################
# 默认目录结构
##############################################
BASE_DIR = path.dirname(path.dirname(path.abspath(__file__)))
buildInputDir = path.join(BASE_DIR, 'data/model_build_inputs')
buildOutputDir = path.join(BASE_DIR, 'data/model_build_outputs')
historyFile = path.join(buildOutputDir, "model.pkl")
smallTestFile = path.join(buildOutputDir, "small_test10.pkl")
applyInputDir = path.join(BASE_DIR, 'data/model_apply_inputs')
applyOutputDir = path.join(BASE_DIR, 'data/model_apply_outputs')

def loadOrCreateAll(fileName=historyFile, dirName=buildInputDir)\
        -> Tuple[RouteDict, HistTransProbDict, ZoneToRankProbDict, Dict[str, Dict[int, float]]]:
    if not fileName.endswith(".pkl"):
        raise RuntimeError
    if path.exists(fileName):
        print(f'load from: {fileName}')
        return loadPkl(fileName)
    else:
        print(f'read from: {dirName}')
        routeDict = readTrain(dirName)
        station2zone2zone2prob = computeTransMatrix(routeDict)
        station2zone2rankProb = computeZoneToRankTransMatrix(routeDict)
        station2rankProb = computeToRankProb(routeDict)
        obj = (routeDict, station2zone2zone2prob, station2zone2rankProb, station2rankProb)
        print(f'save to: {fileName}')
        savePkl(obj, fileName)
        return obj


#######################################################
# 本地调试用
#######################################################

def loadOrCreateAll10(fileName=historyFile, testFile=smallTestFile, dirName=buildInputDir)\
        -> Tuple[RouteDict, HistTransProbDict, ZoneToRankProbDict, Dict[str, Dict[int, float]]]:
    if path.exists(testFile):
        print(f'load from {testFile}')
        return loadPkl(testFile)
    else:
        routeDict, station2zone2zone2prob, station2zone2rankProb, station2rankProb = loadOrCreateAll(fileName, dirName)
        routeDict10 = {}
        for idx,(route_id, route) in enumerate(routeDict.items()):
            if idx >= 10:
                break
            routeDict10[route_id] = route

        obj = (routeDict10, station2zone2zone2prob, station2zone2rankProb, station2rankProb)
        print(f'save to {testFile}')
        savePkl(obj, testFile)
        return obj


def loadOrCreate(fileName='../result/history.pkl', dirName=buildInputDir) -> RouteDict:
    if not fileName.endswith(".pkl"):
        raise RuntimeError
    if path.exists(fileName):
        print(f'load from: {fileName}')
        return loadPkl(fileName)
    else:
        print(f'read from: {dirName}')
        routeDict = readTrain(dirName)
        print(f'save to: {fileName}')
        savePkl(routeDict, fileName)
        return routeDict

def loadOrCreateSmallTest(fileName='../result/history.pkl', testFile='../result/history10.pkl') -> RouteDict:
    if path.exists(testFile):
        print(f'load from {testFile}')
        return loadPkl(testFile)
    else:
        BASE_DIR = path.dirname(path.dirname(path.abspath(__file__)))
        buildInputDir = path.join(BASE_DIR, 'data/model_build_inputs')
        routeDict = loadOrCreate(fileName, buildInputDir)

        route10 = []
        for idx, (route_id, route) in enumerate(routeDict.items()):
            if idx == 10:
                break
            route10.append(route)
        print(f'save to {testFile}')
        savePkl(route10, testFile)
        return route10

# data/model_build_inputs/actual_sequences.json
# The stop_id in every route are in ascending order
def checkStopIDIncreasing():
    BASE_DIR = path.dirname(path.dirname(path.abspath(__file__)))
    actual_sequences_path = path.join(BASE_DIR, 'data/model_build_inputs/actual_sequences.json')
    
    # Read input data
    print('Reading Input Data')
    with open(actual_sequences_path, newline='') as in_file:
        actual_sequences = json.load(in_file,object_pairs_hook=(lambda x: x)) # a dictionary of route_id to visiting sequence
    print(f'{len(actual_sequences)=}')

    for route_id,sequences in actual_sequences: # `RouteID_<hex-hash>`: an alphanumeric string that uniquely identifies each route.
        if len(sequences) != 1:
            raise RuntimeError
        key,sequence = sequences[0]
        if key != 'actual':
            raise RuntimeError
        
        prev = ''
        for stop_id,num in sequence:
            if stop_id <= prev:   
                print(sequence)
                print(f'{prev=}, {stop_id=}')
                raise RuntimeError
            prev = stop_id
    
    print('all stop_id in routes are increasing')


def stats(buildInputDir):
    routeDict = loadOrCreate('./history.pkl', buildInputDir)
    
    station2routeCount = {}
    stopCount2routeCount = {}
    executor_capacity2RC = {}
    route_score2RC = {}
    topZoneSet = set()
    zoneSet = set()
    for route_id, route in routeDict.items():
        station_code = route.station_code
        station2routeCount[station_code] = station2routeCount.get(station_code,0) + 1
        
        stops = route.stops
        stopCount = len(stops)
        stopCount2routeCount[stopCount] = stopCount2routeCount.get(stopCount,0)+1
        
        executor_capacity = route.executor_capacity
        executor_capacity2RC[executor_capacity] = executor_capacity2RC.get(executor_capacity,0)+1
        
        route_score = route.route_score
        route_score2RC[route_score] = route_score2RC.get(route_score,0)+1
        
        for stop in route.stops:
            if stop.zone_id != None:
                zones = stop.zone_id.split('-')
                topZoneSet.add(zones[0])
                zoneSet.add(stop.zone_id)
            
    print(f'number of stations: {len(station2routeCount)}')
    print(f'{station2routeCount=}')
    print(f'number of different stop counts: {len(stopCount2routeCount)}')
    print(f'max stop count in a route: {max(stopCount2routeCount.keys())}')
    print(f'min stop count in a route: {min(stopCount2routeCount.keys())}')
    print(f'{executor_capacity2RC=}')
    print(f'{route_score2RC=}')
    print(f'number of top level zones: {len(topZoneSet)}')
    print(f'top level zones: {topZoneSet}')
    print(f'number of zones: {len(zoneSet)}')



if __name__ == '__main__':
    
#    checkStopIDIncreasing()
    
    # stats(buildInputDir)
    
    routeDict = readTrain(applyInputDir, historical=False)
    