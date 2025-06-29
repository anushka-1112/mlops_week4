import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load sample data
sample_df = pd.read_csv('data/sample.csv')

# Separate features and target
X_sample = sample_df.drop('species', axis=1)
y_sample = sample_df['species']

# Encode target labels
le = LabelEncoder()
le.fit(['setosa', 'versicolor', 'virginica'])  # Ensure same order as training
y_sample_encoded = le.transform(y_sample)

# Standardize features (fit using known Iris stats; or simulate using training fit)
# Ideally, save and load the original scaler from training. Here, we simulate it:
# WARNING: In production, use the same scaler saved during training.
X_train_stats = {
    'mean': [5.843, 3.054, 3.759, 1.199],
    'std': [0.828, 0.433, 1.764, 0.763]
}
scaler = StandardScaler()
scaler.mean_ = np.array(X_train_stats['mean'])
scaler.scale_ = np.array(X_train_stats['std'])
scaler.var_ = scaler.scale_ ** 2  # required for inverse_transform if used
scaler.n_features_in_ = X_sample.shape[1]
X_sample_scaled = scaler.transform(X_sample)

# Load trained model
model = load_model('iris_model.h5')

# Evaluate
loss, accuracy = model.evaluate(X_sample_scaled, y_sample_encoded, verbose=0)
print(f"Validation Accuracy on sample.csv: {accuracy * 100:.2f}%")
