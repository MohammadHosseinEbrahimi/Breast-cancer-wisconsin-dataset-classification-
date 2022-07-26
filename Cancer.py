"""
This script loads the breast cancer wisconsin dataset (classification). 
The breast cancer dataset is a classic and binary classification dataset.

The script tries to test some machine learning approaches to classify the 
cancerous tissue from the noncancerous one. 

Written by Mohammadhossein Ebrahimi
26 July 2022
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_breast_cancer
import sklearn

cancer = load_breast_cancer()
dataa =pd.concat([pd.DataFrame(cancer.data) ,pd.DataFrame(cancer.target)], 
                 axis=1, ignore_index=True)
dataa.reset_index()
dataa.columns= ['mean radius', 'mean texture', 'mean perimeter', 'mean area',
'mean smoothness', 'mean compactness', 'mean concavity',
'mean concave points', 'mean symmetry', 'mean fractal dimension',
'radius error', 'texture error', 'perimeter error', 'area error',
'smoothness error', 'compactness error', 'concavity error',
'concave points error', 'symmetry error', 'fractal dimension error',
'worst radius', 'worst texture', 'worst perimeter', 'worst area',
'worst smoothness', 'worst compactness', 'worst concavity',
'worst concave points', 'worst symmetry', 'worst fractal dimension',
'target']
d=[np.count_nonzero(dataa.target==0, axis=None),
   np.count_nonzero(dataa.target==1, axis=None)]
ser = pd.Series(data=d, index=['malignant', 'benign'])
ser.rename('target')
X=dataa.drop('target', axis=1)
y=dataa['target']
X_train, X_test, y_train, y_test = train_test_split (X, y, random_state=0)


##acuracy of a dummy classifier (majority class)
from sklearn.dummy import DummyClassifier
Dumm= DummyClassifier(strategy="most_frequent")
Dumm.fit(X_train, y_train)
y_predict=Dumm.predict(X_test)
print('Accuracy of K nearest neighbour classifier on training set: {:.2f}'
     .format(Dumm.score(X_train, y_train)))
print('Accuracy of K nearest neighbour classifier on test set: {:.2f}'
     .format(Dumm.score(X_test, y_test)))

# K nearest neighbour
f=KNeighborsClassifier (n_neighbors=5)
f.fit(X_train, y_train)
means = dataa.mean()[:-1].values.reshape(1, -1)
pre=f.predict(means)
test_predict=f.predict(X_train)
score= f.score(X_test, y_test)
print('Accuracy of K nearest neighbour classifier on training set: {:.2f}'
     .format(f.score(X_train, y_train)))
print('Accuracy of K nearest neighbour classifier on test set: {:.2f}'
     .format(f.score(X_test, y_test)))


# Support vector machine
from sklearn.svm import SVC
ff= SVC(kernel='rbf', gamma=0.0001).fit(X_train, y_train)
ff.score(X_test, y_test)
y_predicted = ff.predict(X_test)
score2= sklearn.metrics.accuracy_score(y_test, y_predicted)
cm = sklearn.metrics.confusion_matrix(y_test, y_predicted)
#confusion matrix
from sklearn.metrics import confusion_matrix
svm_predicted = ff.predict(X_test)
confusion = confusion_matrix(y_test, svm_predicted)
#Precision-recall curves
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
precision, recall, thresholds = precision_recall_curve(y_test, y_predicted)
closest_zero = np.argmin(np.abs(thresholds))
closest_zero_p = precision[closest_zero]
closest_zero_r = recall[closest_zero]
plt.figure()
plt.xlim([0.0, 1.01])
plt.ylim([0.0, 1.01])
plt.plot(precision, recall, label='Precision-Recall Curve')
plt.plot(closest_zero_p, closest_zero_r, 'o', markersize = 12, 
         fillstyle = 'none', c='r', mew=3)
plt.xlabel('Precision', fontsize=16)
plt.ylabel('Recall', fontsize=16)
plt.axes().set_aspect('equal')
plt.show()
#Receiver operating characteristic curve, or ROC curve
#AUC is the area under the curve for determing the model performance
from sklearn.metrics import roc_curve, auc
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_predicted)
roc_auc_lr = auc(fpr_lr, tpr_lr)
plt.figure()
plt.xlim([-0.01, 1.00])
plt.ylim([-0.01, 1.01])
plt.plot(fpr_lr, tpr_lr, lw=3, 
         label='LogRegr ROC curve (area = {:0.2f})'.format(roc_auc_lr))
plt.xlabel('False Positive Rate', fontsize=16)
plt.ylabel('True Positive Rate', fontsize=16)
plt.title('ROC curve (1-of-10 digits classifier)', fontsize=16)
plt.legend(loc='lower right', fontsize=13)
plt.plot([0, 1], [0, 1], color='navy', lw=3, linestyle='--')
plt.axes().set_aspect('equal')
plt.show()
print('Accuracy of SVM classifier on training set: {:.2f}'
     .format(ff.score(X_train, y_train)))
print('Accuracy of SVM classifier on test set: {:.2f}'
     .format(ff.score(X_test, y_test)))



#Neural netwrk
from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,  hidden_layer_sizes=(5, 2), 
                    random_state=1)
clf.fit(X_train, y_train)
y_predicted_NN = clf.predict(X_test)
cm = sklearn.metrics.confusion_matrix(y_test, y_predicted_NN)
score_NN=sklearn.metrics.accuracy_score(y_test, y_predicted_NN)
#confusion matrix
NN_predicted = clf.predict(X_test)
confusion_NN = confusion_matrix(y_test, NN_predicted)
print('Accuracy of NN classifier on training set: {:.2f}'
     .format(clf.score(X_train, y_train)))
print('Accuracy of NN classifier on test set: {:.2f}'
     .format(clf.score(X_test, y_test)))




#random forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
clf2 = RandomForestClassifier(n_estimators = 10,
                            random_state=0).fit(X_train, y_train)
print('Accuracy of RF classifier on training set: {:.2f}'
     .format(clf2.score(X_train, y_train)))
print('Accuracy of RF classifier on test set: {:.2f}'
     .format(clf2.score(X_test, y_test)))
