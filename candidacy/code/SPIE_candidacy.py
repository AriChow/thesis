from scipy.io import loadmat
import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV


def principal_components(X, whiten=False):
    c = int(np.min(X.shape))
    pca = PCA(whiten=whiten)
    maxvar = 0.95
    data = X
    pca.fit(X)
    var = pca.explained_variance_ratio_
    s1 = 0
    for i in range(len(var)):
        s1 += var[i]
    s = 0
    for i in range(len(var)):
        s += var[i]
        if (s * 1.0 / s1) >= maxvar:
            break
    pca = PCA(n_components=i + 1)
    pca.fit(data)
    return pca

home = os.path.expanduser('~')
data_home = home + '/Google Drive/research/GE_Project/data/'

data = loadmat(data_home + 'hand_crafted_features.mat')
labels = np.loadtxt(data_home + 'GE_noise_labels1.txt')
labels = labels.astype('int')
X = data['features']
y = labels

skf = StratifiedKFold(n_splits=5)
Cs = np.logspace(-4, 4, 3)
for train, test in skf.split(X, y):
    X_train = X[train, :]
    y_train = y[train]
    X_test = X[test, :]
    y_test = y[test]
    logistic = LogisticRegression()
    # Parameters of pipelines can be set using ‘__’ separated parameter names:
    pca = principal_components(X_train)
    X_train = pca.transform(X_train)
    X_test = pca.transform(X_test)
    estimator = GridSearchCV(logistic,
                             dict(logistic__C=Cs))
    estimator.fit(X_train, y_train)


    print()
