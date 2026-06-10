import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

landmarks_arr = np.array(pd.read_csv('landmarks_training.csv'))

X = landmarks_arr[:, 1:-1]
y = landmarks_arr[:, -1]

label_encoder = LabelEncoder()
y_num = label_encoder.fit_transform(y)

# Linear baseline. Scaling helps convergence, so it's bundled into the
# pipeline and applied automatically on predict().
logreg_clf = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000)
)
logreg_clf.fit(X, y_num)

joblib.dump(logreg_clf, 'models/logistic_regression_model.joblib')
joblib.dump(label_encoder, 'models/label_encoder.joblib')

print("Model and label encoder created!")
