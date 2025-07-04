import pandas as pd
from sklearn import svm
import joblib

# Load dataset
dataset = pd.read_csv('data.csv', header=None)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Train the model using SVM
model = svm.SVC(kernel='linear')
model.fit(X, y)

# Save it using joblib
joblib.dump(model, 'model.pkl')
print("✅ SVM model saved in current environment.")
