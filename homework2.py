from matplotlib import cm
import sklearn
from sklearn.model_selection import train_test_split 
import numpy as np
import pandas as pd
from scipy.io import arff
import matplotlib.pyplot as plt
import matplotlib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import seaborn as sns

data = arff.loadarff('pd_speech.arff')
df = pd.DataFrame(data[0])

X = df.iloc[:, :-1]
y = df.iloc[:, -1]
y = y.astype('int')

skf = StratifiedKFold(n_splits=10)
cm_knn = np.zeros((2,2)) # Initializing the Confusion Matrix
cm_bay = np.zeros((2,2)) # Initializing the Confusion Matrix

clf_knn = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
clf_bay = GaussianNB()
  
for train_index, test_index in skf.split(X, y):
    # KNeighbours
    x_train_knn, x_test_knn = X.iloc[train_index], X.iloc[test_index]
    y_train_knn, y_test_knn = y[train_index], y[test_index]
    clf_knn.fit(x_train_knn, y_train_knn) 
    y_predict_knn = clf_knn.predict(x_test_knn)
    cm_knn = np.add(cm_knn, confusion_matrix(y_test_knn, y_predict_knn))

    # Gaussian
    x_train_bay, x_test_bay = X.iloc[train_index], X.iloc[test_index]
    y_train_bay, y_test_bay = y[train_index], y[test_index]
    clf_bay.fit(x_train_bay, y_train_bay) 
    y_predict_bay = clf_bay.predict(x_test_bay)
    cm_bay = np.add(cm_bay, confusion_matrix(y_test_bay, y_predict_bay))

sns.set(font_scale=1.4)
sns.heatmap(cm_knn, annot=True, annot_kws={"size": 16})
""" 
sns.set(font_scale=1.4)
sns.heatmap(cm_bay, annot=True, annot_kws={"size": 16}) """

plt.show()



