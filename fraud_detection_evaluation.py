
"""
@author: ahmad
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve,auc
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import recall_score, precision_score

df = pd.read_csv('fraud_data.csv')

X = df.iloc[:,:-1]
y = df.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


grid_values = {'C':[0.01, 0.1, 1, 10, 100]}
grid_clf = GridSearchCV(LogisticRegression(), param_grid = grid_values, scoring='recall')
grid_clf.fit(X_train, y_train)
lr_predicted = grid_clf.decision_function(X_test)

precision, recall, thresholds = precision_recall_curve(y_test, lr_predicted)

plt.figure()
plt.plot(precision, recall, label='Precision-Recall Curve')
plt.xlabel('Precision', fontsize=16)
plt.ylabel('Recall', fontsize=16)
plt.show()

fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_predicted)
roc_auc_lr = auc(fpr_lr, tpr_lr)

plt.figure()
plt.plot(fpr_lr, tpr_lr, lw=3, label='LogRegr ROC curve (area = {:0.2f})'.format(roc_auc_lr))
plt.xlabel('False Positive Rate', fontsize=16)
plt.ylabel('True Positive Rate', fontsize=16)
plt.title('ROC curve (1-of-10 digits classifier)', fontsize=16)
plt.legend(loc='lower right', fontsize=13)
plt.plot([0, 1], [0, 1], color='navy', lw=3, linestyle='--')
plt.show()

accuracy = grid_clf.score(X_test, y_test)
y_test_predicted = grid_clf.predict(X_test)
recall = recall_score(y_test, y_test_predicted)
precision = precision_score(y_test, y_test_predicted)

print('accuracy: {}'.format(accuracy))
print('recall: {}'.format(recall))
print('precision: {}'.format(precision))