import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

landmarks_arr = np.array(pd.read_csv('landmarks_training.csv'))

X = landmarks_arr[:, 1:-1]
y = landmarks_arr[:, -1]

label_encoder = LabelEncoder()
y_num = label_encoder.fit_transform(y)

# SVM is distance/scale sensitive, so the scaler is bundled into the
# pipeline and applied automatically on predict().
svm_clf = make_pipeline(
    StandardScaler(),
    # probability=True enables predict_proba (needed for the confidence score).
    SVC(kernel='rbf', C=10, gamma='scale', probability=True)
)
svm_clf.fit(X, y_num)

joblib.dump(svm_clf, 'models/svm_model.joblib')
joblib.dump(label_encoder, 'models/label_encoder.joblib')

print("Model and label encoder created!")
