import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV

# Generate realistic heart dataset (no external file needed)
np.random.seed(42)

data_size = 1000

age = np.random.randint(20, 80, data_size)
trestbps = np.random.randint(90, 180, data_size)
chol = np.random.randint(150, 300, data_size)
thalach = np.random.randint(90, 200, data_size)
oldpeak = np.random.uniform(0, 4, data_size)

# Create DataFrame
df = pd.DataFrame({
    'age': age,
    'trestbps': trestbps,
    'chol': chol,
    'thalach': thalach,
    'oldpeak': oldpeak
})

# Create realistic target (rule-based)
df['target'] = (
    (df['age'] > 50).astype(int) +
    (df['trestbps'] > 140).astype(int) +
    (df['chol'] > 240).astype(int) +
    (df['thalach'] < 140).astype(int) +
    (df['oldpeak'] > 2).astype(int)
)

df['target'] = (df['target'] >= 3).astype(int)

# Features
X = df[['age', 'trestbps', 'chol', 'thalach', 'oldpeak']]
y = df['target']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# Models
lr = CalibratedClassifierCV(LogisticRegression(max_iter=1000), method='sigmoid')
dt = CalibratedClassifierCV(DecisionTreeClassifier(max_depth=5), method='sigmoid')
rf = CalibratedClassifierCV(RandomForestClassifier(n_estimators=100, max_depth=5), method='sigmoid')

# Train
lr.fit(X_train, y_train)
dt.fit(X_train, y_train)
rf.fit(X_train, y_train)

# Save
os.makedirs("models", exist_ok=True)

pickle.dump(lr, open("models/lr.pkl", "wb"))
pickle.dump(dt, open("models/dt.pkl", "wb"))
pickle.dump(rf, open("models/rf.pkl", "wb"))
pickle.dump(scaler, open("models/scaler.pkl", "wb"))

print("Realistic healthcare models trained successfully.")