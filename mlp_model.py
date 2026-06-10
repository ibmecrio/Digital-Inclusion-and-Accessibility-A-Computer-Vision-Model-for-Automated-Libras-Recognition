import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder

landmarks_arr = np.array(pd.read_csv('landmarks_training.csv'))

X = landmarks_arr[:, 1:-1]
y = landmarks_arr[:, -1]

label_encoder = LabelEncoder()
y_num = label_encoder.fit_transform(y)

# Neural nets train better on scaled inputs, so the scaler is bundled
# into the pipeline and applied automatically on predict().
mlp_clf = make_pipeline(
    StandardScaler(),
    MLPClassifier(
        hidden_layer_sizes=(128, 64),
        max_iter=500,
        random_state=42
    )
)
mlp_clf.fit(X, y_num)

joblib.dump(mlp_clf, 'models/mlp_model.joblib')
joblib.dump(label_encoder, 'models/label_encoder.joblib')

print("Model and label encoder created!")
