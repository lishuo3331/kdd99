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

#ROC line
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc  ###计算roc和auc
from sklearn import model_selection

# Import some data to play with


##变为2分类
# X, y = X[y != 2], y[y != 2]

# Add noisy features to make the problem harder
random_state = np.random.RandomState(0)
n_samples, n_features = X.shape
# X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]

# shuffle and split training and test sets
# X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=.3,random_state=0)

# Learn to predict each class against the other
# svm = svm.SVC(kernel='linear', probability=True,random_state=random_state)

###通过decision_function()计算得到的y_score的值，用在roc_curve()函数中
y_score = rf.predict(X)

# Compute ROC curve and ROC area for each class
fpr,tpr,threshold = roc_curve(y, y_score) ###计算真正率和假正率
roc_auc = auc(fpr,tpr) ###计算auc的值

plt.figure()
lw = 2
plt.figure(figsize=(10,10))
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('rf_kdd99')
plt.legend(loc="lower right")
plt.show()




# y_rf = rf.predict(X)   #攻击类型猜测值
# print("rf results:")
# index = 0;
# wrong_number= 0
# for a in y:
#     if(y[index] == y_rf[index]):
#         index +=1
#     else:
#         print('index: ',index,'true value: ',y[index],' pre value: ',y_rf[index])
#         wrong_number +=1
#         index +=1
# print('%.4f%%'%((1-wrong_number/index)*100))
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
