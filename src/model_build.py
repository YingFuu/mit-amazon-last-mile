from os import path
import sys, json, time
import readData as rd


# Get Directory
BASE_DIR = path.dirname(path.dirname(path.abspath(__file__)))
buildInputDir = path.join(BASE_DIR, 'data/model_build_inputs')
buildOutputDir = path.join(BASE_DIR, 'data/model_build_outputs')

# simply copy input files into output dir
# import shutil
# shutil.copyfile(path.join(buildInputDir,'actual_sequences.json'),
#                 path.join(buildOutputDir,'actual_sequences.json'))
# shutil.copyfile(path.join(buildInputDir,'invalid_sequence_scores.json'),
#                 path.join(buildOutputDir,'invalid_sequence_scores.json'))
# shutil.copyfile(path.join(buildInputDir,'package_data.json'),
#                 path.join(buildOutputDir,'package_data.json'))
# shutil.copyfile(path.join(buildInputDir,'route_data.json'),
#                 path.join(buildOutputDir,'route_data.json'))
#shutil.copyfile(path.join(buildInputDir,'travel_times.json'),
#                path.join(buildOutputDir,'travel_times.json'))


historyFile = path.join(buildOutputDir, "model.pkl")
rd.loadOrCreateAll(historyFile, buildInputDir)
print('model submitted by Ying Fu (dhlsfy@163.com), Zhixin Luo and Wenbin Zhu (i@zhuwb.com), (c) 2021-06-17')

# print('Saving Solved Model State')
# output={
#     'Model':'Hello from the model_build.py script!',
#     'sort_by':'lat'
# }
#
# # Write output data
# model_path=path.join(BASE_DIR, 'data/model_build_outputs/model.json')
# with open(model_path, 'w') as out_file:
#     json.dump(output, out_file)
#     print("Success: The '{}' file has been saved".format(model_path))
