# iPhone Sales Price Prediction using Machine Learning

Submitted by: Anika Fariha Chowdhury
Student ID: 24264037
Course: BAN644-Applied Machine Learning for Business Analytics with Python
Project Type: Regression Problem
Dataset: iphone_sales_dataset.csv

## Project Overview

This project focuses on building a machine learning regression model to predict iPhone sales price based on different business-related features. The dataset includes information such as country, iPhone model, storage, color, quantity, sale date, and payment method.

The main objective of this project is to analyze sales data, build regression models, compare their performance, optimize the best-performing model, and generate business insights from the final result.

## Dataset

The dataset used in this project is:

`iphone_sales_dataset.csv`

The target variable is:

`Price`

The selected input features are:

- Country
- iPhone_Model
- Storage
- Color
- Quantity
- Payment_Method
- Sale_Month
- Sale_Day
- Sale_DayOfWeek

Columns such as `Order_ID`, `Customer_Name`, and the original `Sale_Date` were not directly used for prediction because they do not provide strong direct value for model training. However, useful date-based features were extracted from `Sale_Date`.

## Project Workflow

The project followed a complete machine learning workflow:

1. Import required libraries
2. Load the dataset
3. Understand dataset structure
4. Clean and preprocess data
5. Perform exploratory data analysis
6. Select features and target variable
7. Split data into training and testing sets
8. Build preprocessing pipeline
9. Train multiple regression models
10. Optimize the best model using GridSearchCV
11. Evaluate the final optimized model
12. Compare actual and predicted prices
13. Generate business insights and recommendations

## Exploratory Data Analysis

Exploratory Data Analysis was performed to understand the dataset better. The analysis included price distribution, average price by iPhone model, total quantity sold by country, average price by storage size, and payment method usage.

The EDA helped identify patterns in pricing and sales behavior. For example, iPhone model and storage capacity are expected to influence price, while country, quantity, payment method, and sale date features may also provide useful business insights.

## Data Preprocessing

The dataset was cleaned by removing duplicate rows and converting the `Sale_Date` column into a datetime format. From the sale date, new features were created:

- Sale_Month
- Sale_Day
- Sale_DayOfWeek

Categorical features were handled using `OneHotEncoder`, while numerical features were processed using `SimpleImputer` and `StandardScaler`. A machine learning pipeline was used to keep preprocessing and model training organized.

## Models Used

Three regression models were trained and compared:

| Model | MAE | MSE | RMSE | R2 Score |
|---|---:|---:|---:|---:|
| Linear Regression | 333.75 | 160955.58 | 401.19 | -0.467 |
| Decision Tree Regressor | 417.85 | 274389.45 | 523.82 | -1.501 |
| Random Forest Regressor | 342.32 | 156654.06 | 395.80 | -0.428 |

## Model Building Result

Among the initial models, Random Forest Regressor performed best overall because it had the lowest RMSE and the highest R2 score compared to Linear Regression and Decision Tree Regressor.

Linear Regression was simple and easy to interpret, but it could not capture complex relationships in the data. Decision Tree Regressor performed the worst and showed poor generalization. Random Forest performed better because it combines multiple decision trees and reduces overfitting, but it is less interpretable and requires more computation.

However, all models had negative R2 scores, which means they performed worse than a simple average-based prediction. This indicates that the dataset or selected features were not strong enough to produce highly accurate predictions.

## Model Optimization

The Random Forest model was optimized using GridSearchCV.

Best parameters selected:

```python
{
    "model__max_depth": None,
    "model__min_samples_leaf": 4,
    "model__min_samples_split": 10,
    "model__n_estimators": 100
}
