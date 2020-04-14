import pickle
from sklearn.ensemble import RandomForestClassifier

with open('../data/training_df.pkl', 'rb') as f:
    df = pickle.load(f)
with open(r'../data/selected_feat_names.pkl', 'rb') as f:
    selected_feat_names = pickle.load(f)
print("data loaded")

# train on full data set
y = df["attack_type"]
X = df[selected_feat_names]


rfc = RandomForestClassifier(n_jobs=-1, n_estimators=35, criterion="entropy")
rfc.fit(X, y)

'''

'''
estimator = rfc.estimators_[1]

from sklearn.tree import export_graphviz
# 导出为dot 文件
export_graphviz(estimator, out_file='tree.dot',
                feature_names = selected_feat_names,
                class_names = "attack_type",
                rounded = True, proportion = False,
                precision = 2, filled = True)

# 用系统命令转为PNG文件(需要 Graphviz)
from subprocess import call
call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=100'])

# 在jupyter notebook中展示
from IPython.display import Image
Image(filename = 'tree.png')


print("training finished")

# save model for later use
with open(r'../data/rf.pkl', 'wb') as f:
    pickle.dump(rfc, f)
print("model dumped")
