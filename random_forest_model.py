import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

landmarks_arr = np.array(pd.read_csv('landmarks_training.csv'))

X = landmarks_arr[:, 1:-1]
y = landmarks_arr[:, -1]

label_encoder = LabelEncoder()
y_num = label_encoder.fit_transform(y)

# Trees don't need feature scaling.
rf_clf = RandomForestClassifier(
    n_estimators=300,
    n_jobs=-1,
    random_state=42
)
rf_clf.fit(X, y_num)

joblib.dump(rf_clf, 'models/random_forest_model.joblib')
joblib.dump(label_encoder, 'models/label_encoder.joblib')

print("Model and label encoder created!")
