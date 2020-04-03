'''
使用测试集给模型打分
'''
import pickle
import numpy as np
from scoring import cost_based_scoring as cbs

with open('../data/test_df.pkl', 'rb') as f:
    df = pickle.load(f)
with open(r'../data/selected_feat_names.pkl', 'rb') as f:
    selected_feat_names = pickle.load(f)
X = df[selected_feat_names].values
y = df['attack_type'].values  # ground truth
print("data loaded")

# rf 打分
with open('../data/rf.pkl', 'rb') as f:
    rf = pickle.load(f)
y_rf = rf.predict(X)   #攻击类型猜测值
print("rf results:")
index = 0;
wrong_number= 0
for a in y:
    if(y[index] == y_rf[index]):
        index +=1
    else:
        print('index: ',index,'true value: ',y[index],' pre value: ',y_rf[index])
        wrong_number +=1
        index +=1
print('%.4f%%'%((1-wrong_number/index)*100))
# np.set_printoptions(threshold=np.inf)#输出全部元素
# print(y_rf)


# cbs.score(y, y_rf, True)   #打分函数

# # ada boost
# with open('../data/ada.pkl', 'rb') as f:
#     ada = pickle.load(f)
# y_ada = ada.predict(X)
# print("ada results:")
# cbs.score(y, y_ada, True)
#
# # et
# with open('../data/et.pkl', 'rb') as f:
#     et = pickle.load(f)
# y_et = et.predict(X)
# print("et results:")
# cbs.score(y, y_et, True)
#
# # voting
# with open('../data/voting.pkl', 'rb') as f:
#     voting = pickle.load(f)
# y_voting = voting.predict(X)
# print("voting results:")
# cbs.score(y, y_voting, True)
#
# # stacking
# with open('../data/stacking.pkl', 'rb') as f:
#     stacking = pickle.load(f)
# y_stacking = stacking.predict(X)
# print("stacking results:")
# cbs.score(y, y_stacking, True)
