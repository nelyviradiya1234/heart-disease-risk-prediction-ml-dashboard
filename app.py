from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load models
lr = pickle.load(open("models/lr.pkl", "rb"))
dt = pickle.load(open("models/dt.pkl", "rb"))
rf = pickle.load(open("models/rf.pkl", "rb"))
scaler = pickle.load(open("models/scaler.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Inputs
        age = float(request.form['feature1'])
        bp = float(request.form['feature2'])
        chol = float(request.form['feature3'])
        hr = float(request.form['feature4'])
        oldpeak = float(request.form['feature5'])

        features = np.array([[age, bp, chol, hr, oldpeak]])
        features = scaler.transform(features)

        model_choice = request.form['model']

        if model_choice == "Logistic Regression":
            model = lr
        elif model_choice == "Decision Tree":
            model = dt
        else:
            model = rf

        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1] * 100

        # Risk levels
        if probability < 40:
            result = f"Low Risk ✅ (Probability: {probability:.2f}%)"
            color = "green"
        elif probability < 70:
            result = f"Moderate Risk ⚠️ (Probability: {probability:.2f}%)"
            color = "orange"
        else:
            result = f"High Risk 🔴 (Probability: {probability:.2f}%)"
            color = "red"

        return render_template('index.html', prediction_text=result, color=color)

    except:
        return render_template('index.html', prediction_text="Invalid Input!", color="black")

if __name__ == "__main__":
    app.run(debug=True)