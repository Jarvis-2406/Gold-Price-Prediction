# Gold-Price-Prediction
## Project Objective
Gold Price Prediction Model using ML in Google Colab for GitHub Dataset Overview The dataset contains 2,290 rows and 6 columns, including:  
Date – The date of the recorded prices. 
SPX – S&amp;P 500 Index values.
GLD – Gold price (target variable).
USO – Crude oil price. SLV – Silver price. 
EUR/USD – Exchange rate of Euro to US Dollar.

# DATASET USED
- <a href="https://drive.google.com/file/d/1Yf_yHz1JoaCPEmlHqos3b3a4a4XxkVG6/view?usp=sharing">Dataset</a>

## Importing Libraries
To streamline our workflow, we will import all necessary libraries at the beginning. This approach saves time and prevents redundant imports throughout the process.

- **Pandas**: A powerful library built on NumPy, essential for handling structured data in DataFrames. It is widely used for data manipulation, cleaning, merging, reshaping, and aggregation.
- **NumPy**: A fundamental package for numerical computing in Python, known for its efficient operations on large multidimensional arrays and a vast collection of mathematical functions.
- **Matplotlib**: A versatile library for creating static, animated, and interactive visualizations in Python, supporting various output formats.
- **Seaborn**: Built on top of Matplotlib, Seaborn enhances data visualization by providing aesthetically appealing and informative statistical plots.

## Loading the Dataset
To load our dataset, we will utilize the `read_csv()` function from Pandas. Additionally, by specifying the `parse_dates` parameter, we ensure that the date column is automatically converted to a datetime object instead of being read as a generic string. This conversion facilitates efficient time-based operations and visualizations.

To inspect our dataset, we will use Pandas' built-in functions to check the data types of each column and detect any missing values.

## Data Preprocessing: Handling Missing Values
Missing values can significantly impact model performance. Some models, such as Linear Regression, do not perform well when missing data is present, while others, like Random Forest, can handle them to some extent. However, it is always advisable to address missing values before training any model. Pandas automatically recognizes missing values and represents them as `NaN`. We will analyze the dataset for null values and decide on the best imputation strategy.

## Data Wrangling
Data wrangling is a crucial step in any data science project, as it involves transforming raw data into a structured and insightful format. We will set the `Date` column as the index, which will be beneficial for time-series analysis. By examining daily changes in gold prices, we will attempt to identify trends. Since raw price data often appears noisy, smoothing techniques will be applied to extract meaningful insights.

## Identifying Trends in Gold Prices Using Moving Averages
To better understand the price trend, we will implement a smoothing technique known as Moving Averages. Specifically, we will compute a 20-day rolling average using Pandas' `rolling()` function. This technique helps to highlight trends by reducing short-term fluctuations in the data.

## Analyzing Column Distributions
To examine the distribution of numerical columns, we will create histograms using Matplotlib's `subplot()` function. Additionally, we will use Seaborn's `histplot()` function with `kde=True` to visualize the density estimate. To assess skewness, we will compute skewness values for each column. If necessary, transformations such as square root, logarithmic, or inverse transformations will be applied to normalize the data.

## Handling Outliers
Outliers can distort model performance, especially in regression-based models. For instance, in Linear Regression, extreme values can significantly increase the Mean Squared Error. Although some models, such as Decision Trees and ensemble methods like Random Forest, are more robust to outliers, it remains good practice to detect and address them.

## Visualizing Outliers Using Boxplots
Boxplots provide insights into data distribution, highlighting skewness and the presence of outliers. The box represents the interquartile range (25th to 75th percentile), the median is shown as a line inside the box, and whiskers extend to capture most of the data, excluding extreme outliers. We will set thresholds at the 5th and 95th percentiles to normalize extreme values beyond these limits.

## Modeling the Data
Before building our models, we will split the dataset into training and testing sets to evaluate model performance effectively. By setting an 80:20 split, we ensure that our model generalizes well on unseen data while minimizing overfitting.

## Scaling the Data
Feature scaling is an essential step before training machine learning models. Standardization (Z-score normalization) ensures that each feature has a mean of 0 and a standard deviation of 1, making different variables comparable. Scaling is particularly crucial for models that rely on distance-based calculations.

## Lasso Regression
Lasso Regression is a variation of Linear Regression that incorporates L1 regularization, effectively reducing overfitting and selecting important features. Using Scikit-Learn’s `make_pipeline()`, we will apply Lasso Regression with polynomial features (degree = 2) and utilize `GridSearchCV` to find optimal hyperparameters.

Results:
- **R-squared Score**: 0.9687
- **Best Hyperparameters**: `lasso__alpha = 0.0001`
- **Best Score from Grid Search**: 0.9677

GridSearchCV enables hyperparameter tuning, ensuring that the model does not overfit the training data and performs optimally on unseen data.

## RandomForest Regressor for Regression
Next, we will implement an ensemble-based model using the RandomForest Regressor. This method constructs multiple decision trees and aggregates their outputs to improve predictive accuracy and reduce variance.

Results:
- **Best Hyperparameters**: `max_depth = 7`, `n_estimators = 100`
- **Best Score**: 0.9787
- **R-squared Score**: 0.9696

Using `GridSearchCV`, we fine-tune hyperparameters such as the number of trees (`n_estimators`) and maximum tree depth (`max_depth`). The best-performing model consists of 100 decision trees with a maximum depth of 7, yielding optimal results.

By comparing the performance of Lasso Regression and RandomForest Regressor, we can determine the best model for predicting gold prices based on various features.


## Conclusion
In this analysis, we processed and visualized gold price data, addressing missing values, outliers, and feature scaling to enhance model performance. Trend analysis using moving averages revealed price fluctuations, while distribution plots and boxplots helped identify skewness and outliers. We implemented Lasso Regression and Random Forest Regressor, with Random Forest achieving the highest accuracy (R² = 0.9696), making it the preferred model for predicting gold prices due to its robustness in handling non-linearity and outliers. However, Lasso Regression remains valuable for feature selection and interpretability. Future improvements could involve exploring advanced models like XGBoost or ARIMA for time-series forecasting.












