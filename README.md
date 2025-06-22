# 🔮 Customer Churn Prediction App
### App Link : https://churn-pred-ann.streamlit.app/

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.34.0-red)](https://streamlit.io/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

A machine learning-powered web application to predict customer churn using deep learning models.

---

## 📊 Project Overview

This project predicts whether a customer will leave (churn) based on various customer attributes such as:

- 🌍 Geography
- 👤 Gender
- 🎂 Age
- 💳 Credit Score
- 📆 Tenure
- 🏦 Balance
- 🛒 Number of Products
- 💼 Estimated Salary
- ✅ Is Active Member
- 💳 Has Credit Card

---

## 🧪 ML Pipeline

### 1️⃣ Data Preprocessing
- Cleaned dataset
- Label Encoding (Gender)
- One Hot Encoding (Geography)
- Feature Scaling with `StandardScaler`

### 2️⃣ Model Building
- Deep Learning model built with **TensorFlow/Keras**
- Used multiple dense layers
- Applied callbacks like EarlyStopping to avoid overfitting
- Achieved high accuracy on test data

### 3️⃣ Deployment
- Interactive web app built with **Streamlit**
- Model deployed for real-time predictions
---

## 🚀 Tech Stack

- 🐍 **Python 3.x**
- 📊 **Pandas, NumPy**
- 🎯 **Scikit-learn**
- 🧠 **TensorFlow / Keras**
- 🌐 **Streamlit**

---

## 🌐 Application Features

✅ User-friendly input interface  
✅ Real-time churn prediction  
✅ Probability progress visualization  
✅ Clean UI with clear prediction result  
✅ Cloud-ready deployment

---

## 🏁 Run locally

### Clone Repository

```bash
git clone https://github.com/vishalrathourr/ann-churn-prediction.git

cd customer-churn-prediction-app
```
### Install Requirements

```bash
pip install -r requirements.txt
```

### Launch Streamlit App
```bash
streamlit run app.py
```
