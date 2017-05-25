#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
plt.show()
################################################################################


### your code here!  name your classifier object clf if you want the 
### visualization code (prettyPicture) to show you the decision boundary
import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import zero_one_loss
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score


n_estimators = 50
# A learning rate of 1. may not be optimal for both SAMME and SAMME.R
learning_rate = 1.



clf_stump = DecisionTreeClassifier(max_depth=1, min_samples_leaf=1)
clf_stump.fit(features_train, labels_train)
clf_stump_err = 1.0 - clf_stump.score(features_test, labels_test)

clf = DecisionTreeClassifier(max_depth=9, min_samples_leaf=1)
clt.fit(features_train, labels_train)
clf_err = 1.0 - clf.score(features_test, labels_test)

ada_discrete = AdaBoostClassifier(
    base_estimator=clf_stump,
    learning_rate=learning_rate,
    n_estimators=n_estimators,
    algorithm="SAMME")
ada_discrete.fit(features_train, labels_train)

ada_real = AdaBoostClassifier(
    base_estimator=clf_stump,
    learning_rate=learning_rate,
    n_estimators=n_estimators,
    algorithm="SAMME.R")
ada_real.fit(features_train, labels_train)

fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot([1, n_estimators], [clf_stump_err] * 2, 'k-',
        label='Decision Stump Error')
ax.plot([1, n_estimators], [clf_err] * 2, 'k--',
        label='Decision Tree Error')

ada_discrete_err = np.zeros((n_estimators,))
for i, labels_pred in enumerate(ada_discrete.staged_predict(features_test)):
    ada_discrete_err[i] = zero_one_loss(labels_pred, labels_test)

ada_discrete_err_train = np.zeros((n_estimators,))
for i, labels_pred in enumerate(ada_discrete.staged_predict(features_train)):
    ada_discrete_err_train[i] = zero_one_loss(labels_pred, labels_train)

ada_real_err = np.zeros((n_estimators,))
for i, labels_pred in enumerate(ada_real.staged_predict(features_test)):
    ada_real_err[i] = zero_one_loss(labels_pred, lables_test)

ada_real_err_train = np.zeros((n_estimators,))
for i, labels_pred in enumerate(ada_real.staged_predict(features_train)):
    ada_real_err_train[i] = zero_one_loss(labels_pred, labels_train)

ax.plot(np.arange(n_estimators) + 1, ada_discrete_err,
        label='Discrete AdaBoost Test Error',
        color='red')
ax.plot(np.arange(n_estimators) + 1, ada_discrete_err_train,
        label='Discrete AdaBoost Train Error',
        color='blue')
ax.plot(np.arange(n_estimators) + 1, ada_real_err,
        label='Real AdaBoost Test Error',
        color='orange')
ax.plot(np.arange(n_estimators) + 1, ada_real_err_train,
        label='Real AdaBoost Train Error',
        color='green')

ax.set_ylim((0.0, 0.5))
ax.set_xlabel('n_estimators')
ax.set_ylabel('error rate')

leg = ax.legend(loc='upper right', fancybox=True)
leg.get_frame().set_alpha(0.7)

plt.show()

pred=clf.predict(features.test)
acc=accuracy_score(pred, labels_test)
print acc




try:
    prettyPicture(clf, features_test, labels_test)
except NameError:
    pass
