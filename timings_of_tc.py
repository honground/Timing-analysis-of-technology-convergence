import numpy as np
import pandas as pd

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

import random
random.seed(9001)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
import tensorflow as tf
import math

#"2009-12-31T00:00:00Z","2010-03-31T00:00:00Z","2010-06-31T00:00:00Z","2010-09-31T00:00:00Z","2010-12-31T00:00:00Z"]
timestamp = ["2017-12-31T00:00:00Z"]
titles = 'Artificial intelligence'
#path = '/Users/suckwonhong/Desktop/wiki_2/'
path = 'wiki_res_201712/'
i=timestamp[0]

dat_dlp = pd.read_csv(path+"final_AI_wikipedia_sim_2018" + "_" + i.split('-')[0] + i.split('-')[1] + ".csv")

dat_dlp = pd.read_csv("final_AI_wikipedia_sim_2018" + "_" + i.split('-')[0] + i.split('-')[1] + ".csv")

dat_dlp = dat_dlp.drop(columns = 'Unnamed: 0')

y = dat_dlp['horizon']
X = dat_dlp.loc[:, ~dat_dlp.columns.isin(['source','target','horizon','pa_y'])]
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)


###knn
# fix random seed for reproducibility
seed = 7
np.random.seed(seed)


from sklearn.model_selection import GridSearchCV
param_range = list(range(2,6))
param_grid = {"gamma": [0.1,1,10],
              "degree": param_range}
svm = SVC(kernel = 'poly',class_weight='balanced')
grid_search = GridSearchCV(svm, param_grid=param_grid, cv=3, iid=False)
grid_search.fit(X, y)
grid_search.best_score_
grid_search.best_params_


##classifier definition
svm = SVC(kernel = 'poly',class_weight='balanced', degree=4)
knn = KNeighborsClassifier(n_neighbors = 13, p =2)
RF = RandomForestClassifier()
lr = LogisticRegression(random_state=0, solver='liblinear')
"""
xgb6 = XGBClassifier(
    learning_rate =0.007,
    n_estimators=1000,
    max_depth = 3,
    min_child_weight = 5,
    gamma=0.4,
    subsample=0.55,
    colsample_bytree=0.85,
    reg_alpha=0.005,
    objective= 'binary:logistic',
    nthread=4,
    scale_pos_weight=1,
    seed=27
)
"""

cv = StratifiedKFold(n_splits=5,shuffle=False)
conf_mat = np.zeros((2,2))

pred = pd.DataFrame(columns = ['target','convergence'])

for train,test in cv.split(X,y):
    #print(train)
    prediction = lr.fit(X[train], y.iloc[train]).predict(X[test])
    #target_tech = target_tech + test
    print(test)
    cm = []
    cm = confusion_matrix(y.iloc[test], prediction)
    print(cm)
    #res=np.hstack((test.reshape(-1,1),np.array(pred_res)[test].reshape(-1,1)))
    #df=pd.DataFrame(res)
    #df.columns =['target','convergence']
    #pred = pd.concat([pred,df])
    conf_mat = conf_mat + cm


###knn grid search
from sklearn.model_selection import GridSearchCV
param_grid = {'n_neighbors':np.arange(1,round(math.sqrt(X.shape[0]*0.8)))}
knn = KNeighborsClassifier(p=2)
knn_cv= GridSearchCV(knn,param_grid,cv=3)
knn_cv.fit(X,y)
knn_cv.best_score_
knn_cv.best_params_


###random forest grid search
from sklearn.model_selection import GridSearchCV
param_range = list(range(64,100))
param_grid = {"max_depth": [10,50,None],
              "max_features": [1,5,10],
              "bootstrap": [True, False],
              "n_estimators": param_range}
RF = RandomForestClassifier()
grid_search = GridSearchCV(RF, param_grid=param_grid, cv=3, iid=False)
grid_search.fit(X, y)
grid_search.best_score_
grid_search.best_params_




from sklearn.cross_validation import ShuffleSplit
from sklearn.metrics import accuracy_score
from collections import defaultdict
RF = RandomForestClassifier(n_estimators=96, bootstrap=True, max_depth=10, max_features=5)
scores = defaultdict(list)
names = np.array(['ra', 'aa', 'pa_x', 'jc',  'cn', 'sorensen',
       'salton', 'lhn', 'hpi', 'hdi'])

cv = StratifiedKFold(n_splits=3,shuffle=False)
for train,test in cv.split(X,y):
    #print(train)
    r= RF.fit(X[train], y.iloc[train])
    acc=accuracy_score(y.iloc[test], r.predict(X[test]))
    for i in range(X.shape[1]):
        X_t = X[test].copy()
        np.random.shuffle(X_t[:,i])
        shuff_acc = accuracy_score(y.iloc[test], r.predict(X_t))
        scores[names[i]].append((acc-shuff_acc)/acc)

print ([(round(np.mean(score), 7), feat) for feat, score in scores.items()])




##svm grid search
from sklearn.model_selection import GridSearchCV
param_range = list(range(2,6))
param_grid = {"gamma": [0.1,1,10],
              "degree": param_range}
svm = SVC(kernel = 'poly',class_weight='balanced')
grid_search = GridSearchCV(svm, param_grid=param_grid, cv=3, iid=False)
grid_search.fit(X, y)
grid_search.best_score_
grid_search.best_params_




###nnet
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import InputLayer, Input
from tensorflow.python.keras.layers import Reshape, MaxPooling2D
from tensorflow.python.keras.layers import Conv2D, Dense, Flatten
from tensorflow.python.keras.utils import np_utils
from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import minmax_scale
from sklearn.ensemble import RandomForestClassifier
from tensorflow.python.keras.layers import LeakyReLU
from sklearn.utils import class_weight
from sklearn.model_selection import StratifiedKFold

e = LabelEncoder()
e.fit(y)
Y = e.transform(y)
skf = StratifiedKFold(n_splits=5, shuffle=False)

conf_mat = np.zeros((2,2))

for train, test in skf.split(X, Y):
    model = Sequential()
    model.add(Dense(4, input_dim=10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
    model.fit(X[train], Y[train], epochs=10, batch_size=2)
    prediction = model.predict(X[test])
    prediction = prediction > 0.5
    prediction = prediction.astype(int)
    cm=confusion_matrix(Y[test],prediction)
    conf_mat = conf_mat + cm
