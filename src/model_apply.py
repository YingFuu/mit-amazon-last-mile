from os import path
import json
from tsp_solver.greedy import solve_tsp
import readData as rd
from zoneTSP import zone_tsp_solver

# Get Directory
BASE_DIR = path.dirname(path.dirname(path.abspath(__file__)))
buildOutputDir=path.join(BASE_DIR, 'data/model_build_outputs')

# Read input data
print('Reading Input Data')
# Model Build output
historyFile = path.join(buildOutputDir, "model.pkl")
routeDict, station2zone2zone2prob, station2zone2rankProb, station2rankProb = rd.loadOrCreateAll(historyFile,None)

# Prediction Routes (Model Apply input)
prediction_routes_path = path.join(BASE_DIR, 'data/model_apply_inputs')
routeDict = rd.readTrain(prediction_routes_path, historical=False)




proposed = {}
for route_id,route in routeDict.items():
    zone2zone2prob = station2zone2zone2prob[route.station_code]
    try:
        visited_stop_idxs = zone_tsp_solver(route, zone2zone2prob, sliding_window_len=10)
        route.verify_visit_sequence(visited_stop_idxs)
    except:
        print(f'error occurred resort to simple tsp')
        _, visitedStopIdxs = route.getVisitedStops(computeTime=True, repeatStation=True)
        visited_stop_idxs = solve_tsp(route.dist, endpoints=(visitedStopIdxs[0], visitedStopIdxs[0]))
    # print(f'found sequence: {visited_stop_idxs}')

    ansDict = {}
    ansDict['proposed'] = route.visitOrderToDict(visited_stop_idxs)
    proposed[route_id] = ansDict

# Write output data
output_path=path.join(BASE_DIR, 'data/model_apply_outputs/proposed_sequences.json')
with open(output_path, 'w') as out_file:
    json.dump(proposed, out_file)
    print("Success: The '{}' file has been saved".format(output_path))

print('model submitted by Ying Fu (dhlsfy@163.com), Zhixin Luo and Wenbin Zhu (i@zhuwb.com), (c) 2021-06-17')
print('Done!')
