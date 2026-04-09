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

knn_clf = KNeighborsClassifier(n_neighbors=5)
knn_clf.fit(X, y_num)

joblib.dump(knn_clf, 'knn_model.joblib')
joblib.dump(label_encoder, 'label_encoder.joblib')

print("Model and label encoder created!")