# AutoML Lite

A web-based Automated Machine Learning application that allows users to upload CSV datasets, automatically detect machine learning tasks, train different machine learning models, evaluate their performance, visualize results, and receive AI-assisted explanations.

## Live Demo

https://automl-lite-sjva.onrender.com

##  Features

-  Upload CSV datasets
-  Preview uploaded data
-  Automatic Classification/Regression detection
-  Target column selection
-  Multiple machine learning models
-  Model evaluation metrics
-  Data and model visualizations
-  AI Copilot for interpreting results
-  Live deployment with Render

##  Machine Learning Models

### Classification

- Logistic Regression
- Random Forest Classifier
- Support Vector Machine (SVM)

### Regression

- Linear Regression
- Random Forest Regressor
- Support Vector Regression (SVR)

## Technologies

- Python
- Flask
- Flask-CORS
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- HTML
- CSS
- JavaScript
- Gunicorn
- Render

## Project Structure

```text
automl-lite/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── datasets/
├── models/
│
├── static/
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── script.js
│
├── templates/
│   └── index.html
│
└── utils/
    ├── ai_copilot.py
    ├── copilot.py
    ├── data_handler.py
    ├── ml_models.py
    └── visualization.py