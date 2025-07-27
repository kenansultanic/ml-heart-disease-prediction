# Heart Disease Prediction using Machine Learning

This project demonstrates the application of several machine learning algorithms for **binary classification**.  
The dataset consists of **medical records of patients** and is used to predict the presence of heart disease.  
The target variable is **HeartDisease** (0 – no disease, 1 – has disease), while other columns represent demographic and clinical characteristics of patients, such as age, blood pressure, cholesterol levels, ECG results, chest pain type, and more.

## Project Workflow

The analysis includes the following steps:

1. **Exploratory Data Analysis (EDA) and Visualization**
2. **Data Preprocessing** (data cleaning, encoding categorical variables, scaling)
3. **Training and Evaluation of Multiple Classification Models**
4. **Comparing Model Performance** using metrics such as:
   - Accuracy
   - Precision
   - Recall
   - F1-score
   - ROC-AUC
   - Confusion Matrix
   - Probability Distributions

## Algorithms Used

The focus is on comparing three algorithms:

- **Logistic Regression**
- **Support Vector Machines (SVM)**
- **k-Nearest Neighbours (KNN)**

The goal is to compare their performance and identify the most efficient approach for predicting heart disease based on the available patient features.

## Project Structure

The project is organized into multiple folders for better clarity and modularity:

- **`data/`** – contains the CSV file (`heart.csv`) with the original data used for analysis and model training.
- **`logics/`** – contains the `functions.py` file, which implements helper functions for statistical analyses and visualizations, such as:
  - Model evaluation metrics
  - Confusion matrix visualization
  - ROC curves
  - Probability distribution plots
