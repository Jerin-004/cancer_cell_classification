from flask import Flask, request, render_template
from sklearn.datasets import load_breast_cancer
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load and train model only if not already saved
MODEL_PATH = 'model.pkl'

if not os.path.exists(MODEL_PATH):
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.33, random_state=42
    )
    model = GaussianNB()
    model.fit(X_train, y_train)
    joblib.dump(model, MODEL_PATH)

model = joblib.load(MODEL_PATH)
feature_names = load_breast_cancer().feature_names

@app.route('/')
def home():
    return render_template('index.html', feature_names=feature_names)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [float(request.form[f'feature{i}']) for i in range(30)]
        prediction = model.predict([features])[0]
        result = "Malignant" if prediction == 0 else "Benign"
        return render_template('index.html', prediction_text=f'Result: {result}', feature_names=feature_names)
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}', feature_names=feature_names)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=False)
