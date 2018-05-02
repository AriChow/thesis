from imblearn.over_sampling import SMOTE
import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV
import cv2
from mahotas.features import haralick
import pickle
from sklearn.preprocessing import StandardScaler

def haralick_all_features(names, distance=1):
    # if os.path.exists(self.data_location + 'features/' + self.type1 + '/haralick_' + self.data_name + '.npz'):
    # 	f = np.load(open(self.data_location + 'features/' + self.type1 + '/haralick_' + self.data_name + '.npz', 'rb'))
    # 	return f.f.arr_0[idx, :]
    # else:
    f = []
    for i in range(len(names)):
        I = cv2.imread(names[i])
        if I is None or I.size == 0 or np.sum(I[:]) == 0 or I.shape[0] == 0 or I.shape[1] == 0:
            h = np.zeros((1, 13))
        else:
            I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
            h = haralick(I, distance=distance, return_mean=True, ignore_zeros=False)
            h = np.expand_dims(h, 0)
        if i == 0:
            f = h
        else:
            f = np.vstack((f, h))
    # np.savez(open(self.data_location + 'features/' + self.type1 + '/haralick_' + self.data_name + '.npz', 'wb'), f)
    return f

def principal_components(X, whiten=False):
    c = int(np.min(X.shape))
    pca = PCA(whiten=whiten)
    maxvar = 0.98
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
data_home = home + '/Documents/research/thesis/candidacy/GE_data_files/'


f = open(data_home + 'E_cad_CK15_pck26_names.txt', 'r')
names = f.readlines()
names1 = []
for n in names:
    names1.append(n.strip('\n'))

labels = np.loadtxt(data_home + 'E_cad_CK15_pck26_labels.txt')

# images_home = home + '/Documents/research/GE_project/images/'
# names = []
# y = []
# for i in range(len(names1)):
#     name = names1[i]
#     label = labels[i, :]
#     patient = name[-3:]
#     group = name[-10:-7]
#     ecad = images_home + group + '/AFRemoved/E_cad_AFRemoved_' + patient + '.tif'
#     ck15 = images_home + group + '/AFRemoved/CK15_AFRemoved_' + patient + '.tif'
#     pck26 = images_home + group + '/AFRemoved/pck26_AFRemoved_' + patient + '.tif'
#     names.append(ecad)
#     names.append(ck15)
#     names.append(pck26)
#     y.append(label[0])
#     y.append(label[1])
#     y.append(label[2])
# y = np.asarray(y, dtype='int')
# for i in range(len(y)):
#     if y[i] == -1:
#         y[i] = 1
#     else:
#         y[i] = 0
#
# X = haralick_all_features(names)
# pickle.dump([names, X, y], open('../results/GE_data_py27.pkl', 'wb'))

# names, X, y = pickle.load(open('../results/GE_data.pkl', 'rb'))
#
# skf = StratifiedKFold(n_splits=5)
# params = {'C': np.logspace(-4, 4, 5)}
# probs = np.zeros(len(names))
# for train_ids, test_ids in skf.split(X, y):
#     X_train = X[train_ids]
#     y_train = y[train_ids]
#     X_test = X[test_ids]
#     y_test = y[test_ids]
#     slr = StandardScaler()
#     slr.fit(X_train)
#     X_train = slr.transform(X_train)
#     X_test = slr.transform(X_test)
#     # sm = SMOTE(random_state=42)
#     # X_res, y_res = sm.fit_sample(X_train, y_train)
#     # pca = principal_components(X_res)
#     # X_res = pca.transform(X_res)
#     logreg = LogisticRegression(class_weight='balanced')
#     clf = GridSearchCV(logreg, params)
#     clf.fit(X_train, y_train)
#     # X_test = pca.transform(X_test)
#     prob = clf.predict_proba(X_test)
#     for i in range(prob.shape[0]):
#         probs[test_ids[i]] = prob[i, 1]
#
# pickle.dump([names, probs], open('../results/GE_probs.pkl', 'wb'))

names, probs = pickle.load(open('../results/GE_probs_smote_pca.pkl', 'rb'))



import matplotlib.pyplot as plt

a = 0.01 * np.arange(101)
b = np.arange(0, 100) * 0.01

plt.figure()
hist, _ = np.histogram(probs, a)
plt.bar(b, hist, align='edge', width=0.01)
# plt.xticks(np.linspace(0, 1, 100))
plt.xlabel('Probability')
plt.ylabel('Frequency')
plt.title('Distribution of QoI scores for all images')
plt.savefig('../results/all_probs_smote_pca.eps')
plt.close()

names_ecad = []
probs_ecad = []
names_ck15 = []
probs_ck15 = []
names_pck26 = []
probs_pck26 = []
for i in range(len(probs)):
    if 'E_cad' in names[i]:
        names_ecad.append(names)
        probs_ecad.append(probs[i])
    elif 'CK15' in names[i]:
        names_ck15.append(names)
        probs_ck15.append(probs[i])
    elif 'pck26' in names[i]:
        names_pck26.append(names)
        probs_pck26.append(probs[i])

plt.figure()
hist, _ = np.histogram(probs_ecad, a)
plt.bar(b, hist, align='edge', width=0.01)
plt.xlabel('Probability')
plt.ylabel('Frequency')
plt.title('Distribution of QoI scores for E_cad')
plt.savefig('../results/E_cad_probs_smote_pca.eps')
plt.close()

plt.figure()
hist, _ = np.histogram(probs_ck15, a)
plt.bar(b, hist, align='edge', width=0.01)
plt.xlabel('Probability')
plt.ylabel('Frequency')
plt.title('Distribution of QoI scores for CK15')
plt.savefig('../results/CK15_probs_smote_pca.eps')
plt.close()

plt.figure()
hist, _ = np.histogram(probs_pck26, a)
plt.bar(b, hist, align='edge', width=0.01)
plt.xlabel('Probability')
plt.ylabel('Frequency')
plt.title('Distribution of QoI scores for E_cad')
plt.savefig('../results/pck26_probs_smote_pca.eps')
plt.close()

print()
