'''
使用测试集给模型打分
'''
import pickle
import numpy as np
from scoring import cost_based_scoring as cbs

with open('../data/cls_df.pkl', 'rb') as f:
    df = pickle.load(f)
with open(r'../data/selected_feat_names.pkl', 'rb') as f:
    selected_feat_names = pickle.load(f)
X = df[selected_feat_names].values   #得到cls_df对应特征的值
print("data loaded")

# rf 打分
with open('../data/rf.pkl', 'rb') as f:
    rf = pickle.load(f)
y_rf = rf.predict(X)   #根据特征集对攻击类型进行判断
print("rf results:")
np.set_printoptions(threshold=np.inf)#输出全部元素
print(y_rf)

