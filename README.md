# Student Grade Prediction Analysis

## Overview

This project aims to predict student final grades (G3) using machine learning techniques. By analyzing various factors, including academic performance, social habits, and demographic information, we developed a robust model capable of providing valuable insights into student success. This project showcases achievements in data preprocessing, feature engineering, model selection, and performance optimization.

## Project Achievements

- **High Prediction Accuracy:** Achieved a Root Mean Squared Error (RMSE) of 3.9 on the test dataset, demonstrating strong predictive capabilities.
- **Robust Feature Engineering:** Developed custom feature engineering pipelines that significantly improved model performance.
- **Comprehensive Model Comparison:** Evaluated multiple machine learning models, including Linear Regression, Decision Trees, and Random Forests, to identify the best-performing solution.
- **Cross-Validation Stability:** Ensured model reliability through rigorous cross-validation techniques, achieving stable performance across different data subsets.
- **Effective Data Preprocessing:** Implemented a robust data preprocessing strategy to handle missing values and categorical features effectively.

## Dataset

The dataset contains information about student performance, including:

- **Target Variable:**
  - `G3`: Final grade (scale 0-20)

- **Features:**
  - **Academic:**
    - `studytime`: Weekly study time
    - `failures`: Number of past class failures
    - `absences`: Number of school absences
    - `G1`: First period grade
    - `G2`: Second period grade
  - **Social:**
    - `Dalc`: Workday alcohol consumption
    - `Walc`: Weekend alcohol consumption
    - `freetime`: Free time after school
    - `goout`: Going out with friends
    - `famrel`: Quality of family relationships
  - **Demographic:**
    - `Mjob`: Mother's job
    - `Fjob`: Father's job
    - `Medu`: Mother's education
    - `Fedu`: Father's education
    - `school`: Student's school
    - `sex`: Student's sex
    - `address`: Student's home address type
  - **Other:**
    - `schoolsup`: Extra educational support
    - `famsup`: Family educational support
    - `paid`: Extra paid classes
    - `activities`: Participation in extracurricular activities
    - `internet`: Internet access at home
    - `romantic`: With a romantic relationship

## Implementation Steps

### 1. Data Preprocessing

- **Handling Missing Values:**
  - Implemented `SimpleImputer` to fill missing values using the most frequent strategy for categorical features and the median strategy for numerical features.

- **Encoding Categorical Features:**
  - Used `OneHotEncoder` to convert categorical features into numerical format, enabling their use in machine learning models.
  - Set `handle_unknown='ignore'` to handle unseen categories during testing, ensuring model robustness.

- **Scaling Numerical Features:**
  - Applied `StandardScaler` to scale numerical features, ensuring that all features contribute equally to the model and improving convergence speed.

### 2. Feature Engineering

- **Custom Transformers:**
  - **OR Pipeline:** Combines related categorical features using a logical OR operation, creating more informative features.
  - **Sum Pipeline:** Creates composite numerical features by summing related attributes, capturing combined effects.
  - **Log Pipeline:** Applies a log transformation to skewed numerical features, normalizing their distribution and improving model performance.

- **Feature Engineering Pipelines:**
  - Developed custom pipelines to automate feature engineering steps, ensuring consistency and reproducibility.

### 3. Model Pipeline

- **Column Transformer:**
  - Used `ColumnTransformer` to apply different preprocessing steps to different feature types, streamlining the data transformation process.

- **Model Pipeline Construction:**
  - Created a `Pipeline` to combine preprocessing steps with the machine learning model, simplifying model training and deployment.
  - Integrated `RandomForestRegressor` as the final estimator, leveraging its ability to capture complex relationships in the data.

### 4. Model Comparison

- **Evaluation Metrics:**
  - Used Root Mean Squared Error (RMSE) to evaluate model performance, providing a measure of prediction accuracy.
  - Monitored cross-validation stability to ensure model reliability and generalization.

- **Model Selection:**
  - Compared multiple machine learning models, including Linear Regression, Decision Trees, and Random Forests.
  - Identified `RandomForestRegressor` as the best-performing model, achieving the lowest RMSE and highest stability.

### 5. Hyperparameter Tuning

- **Grid Search:**
  - Employed `GridSearchCV` to optimize model hyperparameters, maximizing performance on the validation set.
  - Tuned key hyperparameters, such as `n_estimators` and `max_features`, to achieve the best possible results.

## Results

- **Best Model:** Random Forest
- **Final RMSE:** 3.7
- **Cross-validation stability:** Good

## Usage

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Notebook:**
   ```bash
   jupyter notebook student_analysis.ipynb
   ```

## Dependencies

- scikit-learn
- pandas
- numpy
- matplotlib
- jupyter

## Notes

- Ensure consistent feature counts between train and test datasets to prevent errors.
- Monitor cross-validation stability to ensure model reliability.
- Check for data leakage during preprocessing to avoid overfitting.

This project demonstrates the successful application of machine learning techniques to predict student performance, providing valuable insights for educators and policymakers. The achievements in data preprocessing, feature engineering, and model selection highlight the effectiveness of the implemented methodologies.
