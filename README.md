## 基于随机森林算法的kdd99数据集分类器
参考链接：https://github.com/lishuo3331/kdd99/upload/master

数据集: http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html

## 本机环境
1.pycharm 2019.3.3

2.python 3.6(ubuntu 18.04)

## 文件解析
Find_sparse_feature.py  寻找稀疏特征

Feat_untils.py 合并稀疏特征+独热编码+attack_type合并为五类

Make_selected_feat.py 特征筛选

Make_training_df.py 生成训练集

Dump_best_rf.py full data set上训练随机随机森林模型，并且保存模型以便随后使用

Rf_gs.py  gridsearch  自动优化超参数，适用于小数据集

Make_test_df.py 生成测试集

Make_predictions.py   对随机森林算法进行打分

data.py中设置数据
   __FILE10PCCSV = r"../data/train10pctonumber.csv"

   __FILE10PC = r"../data/train10pc"
 
   __FILE = r"../data/train"

按照上述顺序运行即可
