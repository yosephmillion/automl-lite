# AutoML Lite — Machine Learning Website with AI Copilot

AutoML Lite is a locally runnable machine learning web application that allows users to upload CSV datasets, inspect their data, automatically configure a machine learning task, train models, view evaluation metrics, and receive AI-assisted recommendations.


## Features

- CSV-only dataset upload
- CSV validation and error handling
- Automatic header detection
- Manual header correction
- Dataset preview of the first five rows
- Automatic target-variable selection
- Manual target-variable selection
- Automatic Classification/Regression detection
- Manual task-type override
- Descriptive statistics
- Data distribution visualization
- Multiple machine learning models
- 80/20 train-test split
- Classification metrics
- Regression metrics
- Confusion matrix visualization
- Actual vs. predicted visualization
- Feature importance analysis
- AI Copilot recommendations
- Responsive modern interface

## Machine Learning Models

### Classification

- Logistic Regression
- Random Forest
- SVM
- Decision Tree
- KNN
- Naive Bayes

### Regression

- Linear Regression
- Random Forest Regressor
- SVR
- Decision Tree Regressor
- Ridge
- Lasso

## Technologies

### Frontend

- HTML5
- CSS3
- JavaScript

### Backend

- Python
- Flask
- Pandas
- NumPy
- Scikit-learn
- Matplotlib

## How It Works

1. Upload a CSV dataset.
2. AutoML Lite detects whether the dataset contains a header.
3. The user can confirm or change the header setting.
4. The last column is selected as the default target.
5. The user can manually select another target.
6. The application detects Classification or Regression.
7. The user can override the detected task.
8. Available machine learning models are displayed.
9. The selected model is trained using an 80/20 train-test split.
10. Evaluation metrics and visualizations are displayed.
11. The AI Copilot analyzes the results and provides recommendations.

## Installation

Clone the repository:

```bash
git clone https://github.com/yosephmillion/automl-lite.git
cd automl-lite
```

---
Yoseph Million Mekuria

RespAI 

*August 16, 2025*
