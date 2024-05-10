运行train.py
输入学生id即可得到该学生最合适top-3（可更改top-k）教师id

运行train_exp.py
即可得到论文中该模型所有参数指标以及参数图



f1score.py
所有实验参数代码文件

Logger.py
日志文件

model.py
本论文所设计的模型

mydataset.py
导入数据集文件

predictions.txt
保存本模型预测每个学生对每个教师评分的文件，以二维数组表示，大小为900（学生数量）x1600（教师数量）

对于数据集的介绍详见论文第4.1节



