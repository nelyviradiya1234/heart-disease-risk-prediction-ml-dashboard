# ❤️ Heart Disease Risk Prediction ML Dashboard

An interactive machine learning dashboard for predicting heart disease
risk using **Logistic Regression, Decision Tree, and Random Forest**
models.

The project provides a simple web interface where users can enter
health-related parameters and receive a probability-based risk
classification.

## 📌 Project Overview

This project demonstrates an end-to-end machine learning workflow for
healthcare risk prediction:

-   Generate a heart-related dataset using five input features
-   Preprocess features using `StandardScaler`
-   Train multiple classification models
-   Calibrate model probabilities using `CalibratedClassifierCV`
-   Save trained models using Pickle
-   Allow users to select a machine learning model
-   Generate a predicted risk probability
-   Classify the result into Low, Moderate, or High Risk
-   Provide interactive web interfaces using Streamlit and Flask

## 🧠 Machine Learning Models

The project uses three classification algorithms:

1.  **Logistic Regression**
2.  **Decision Tree**
3.  **Random Forest**

The models are calibrated using sigmoid calibration to provide
probability estimates.

## 📊 Input Features

The dashboard uses the following five input features:

  Feature          Description
  ---------------- -----------------------------
  Age              Age of the person
  Blood Pressure   Resting blood pressure
  Cholesterol      Cholesterol level
  Max Heart Rate   Maximum heart rate
  Oldpeak          ST depression-related value

## 🚦 Risk Classification

The predicted probability is converted into three risk levels:

  Probability     Risk Level
  --------------- ------------------
  Less than 40%   🟢 Low Risk
  40% -- 69.99%   🟠 Moderate Risk
  70% or higher   🔴 High Risk

## 🖥️ Dashboard

The Streamlit dashboard provides:

-   Patient input fields
-   Model selection
-   Prediction button
-   Probability-based risk result
-   Simple and user-friendly interface

A Flask-based web interface is also included with an HTML frontend.

## 🛠️ Technologies Used

-   **Python**
-   **Pandas**
-   **NumPy**
-   **Scikit-learn**
-   **Streamlit**
-   **Flask**
-   **HTML**
-   **CSS**
-   **Pickle**

## 📁 Project Structure

``` text
heart-disease-risk-prediction-ml-dashboard/
│
├── models/
│   ├── lr.pkl
│   ├── dt.pkl
│   ├── rf.pkl
│   └── scaler.pkl
│
├── templates/
│   └── index.html
│
├── app.py
├── streamlit_app.py
├── train_models.py
├── README.md
└── TEXT.txt
```

## ⚙️ How It Works

### 1. Data Generation

The training script generates 1,000 records using randomly generated
values for age, blood pressure, cholesterol, maximum heart rate, and
oldpeak.

The target variable is created using a rule-based combination of these
features.

### 2. Data Preprocessing

The input features are standardized using `StandardScaler`.

### 3. Model Training

The following models are trained:

-   Logistic Regression
-   Decision Tree
-   Random Forest

The trained models and scaler are saved as `.pkl` files inside the
`models` directory.

### 4. Prediction

When a user enters patient information, the dashboard:

1.  Collects the five input values
2.  Applies the saved scaler
3.  Loads the selected model
4.  Calculates the predicted probability
5.  Displays the corresponding risk level

## 🚀 Installation

Clone the repository:

``` bash
git clone https://github.com/nelyviradiya1234/heart-disease-risk-prediction-ml-dashboard.git
```

Move into the project directory:

``` bash
cd heart-disease-risk-prediction-ml-dashboard
```

Install the required Python packages:

``` bash
pip install numpy pandas scikit-learn streamlit flask
```

## ▶️ Run the Streamlit Dashboard

Run:

``` bash
streamlit run streamlit_app.py
```

The Streamlit application will open in your browser.

## ▶️ Run the Flask Application

Run:

``` bash
python app.py
```

The Flask application will start locally.

## 🔄 Retrain the Models

To generate and train the models again, run:

``` bash
python train_models.py
```

This creates or updates:

``` text
models/lr.pkl
models/dt.pkl
models/rf.pkl
models/scaler.pkl
```

## ⚠️ Important Note

This project is an **educational machine learning demonstration** and
should not be used as a medical diagnosis or as a substitute for
professional medical advice.

The training data in the provided project is generated synthetically
using rule-based target creation rather than being sourced from a
clinical dataset. Therefore, the predictions should not be interpreted
as medically validated risk assessments.

## 🎯 Learning Objectives

This project demonstrates practical skills in:

-   Machine Learning
-   Classification
-   Data preprocessing
-   Feature scaling
-   Model calibration
-   Model comparison
-   Probability-based prediction
-   Python development
-   Streamlit dashboard development
-   Flask web development
-   Model serialization with Pickle

## 👩‍💻 Author

**Nely Viradiya**

GitHub:\
https://github.com/nelyviradiya1234

## 📄 License

This project is intended for educational and portfolio purposes.
