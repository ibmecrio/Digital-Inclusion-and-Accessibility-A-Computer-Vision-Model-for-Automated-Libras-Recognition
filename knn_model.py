import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

landmarks_arr = np.array(pd.read_csv('landmarks_training.csv'))
landmarks_test = np.array(pd.read_csv('landmarks_test.csv'))

X = landmarks_arr[:, 1:-1]
y = landmarks_arr[:, -1]

label_encoder = LabelEncoder()

y_num = label_encoder.fit_transform(y)

# Uniform weights over more neighbors give a meaningful confidence score:
# predict_proba returns the fraction of the 21 neighbors that voted for the
# class, which actually measures neighbor agreement. (distance weighting was
# avoided here because the nearest neighbor dominates and pins confidence
# near 100%, defeating the purpose of a confidence index.)
knn_clf = KNeighborsClassifier(n_neighbors=21, weights='uniform')
knn_clf.fit(X, y_num)

joblib.dump(knn_clf, 'models/knn_model.joblib')
joblib.dump(label_encoder, 'models/label_encoder.joblib')

print("Model and label encoder created!")