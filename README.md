# mit-amazon-last-mile
source code for last mile routing research challenge

## Files
src/    FY,Luo,Tiny 写的各种程序源代码，及程序暂时需要保存的文件
    readData.py     读取历史数据, 学习统计规律等
    zoneTSP.py      基于历史统计生成 zone 级别的路径，然后 zone 内用 tsp 生成 stop 级别的路径
    fy_dp.py        用 DP 求解小规模 tsp 的最优解
    model_build.py  比赛用接口文件，读入历史数据，把学习的模型保存到 ../data/model_build/outputs/model.pkl
    model_apply.py  比赛用接口文件，从 ../data/model_build/outputs 读入训练好的模型，../data/model_apply/inputs
                    读入新任务数据。生成预测，保存到 ../data/model_apply/outputs/proposed_sequences.json
   	fy_score.py     根据官方评分标准写的评分程序
	  zoneTSP_exp.py  用历史数据，比较 zone_tsp, zone_tsp+in_zone_sliding, zone_tsp+post_optimize, tsp 的效果
