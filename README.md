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
We will import all the libraries that we will be using throughout this article in one place so that do not have to import every time we use it this will save both our time and effort.

Pandas – A Python library built on top of NumPy for effective matrix multiplication and dataframe manipulation, it is also used for data cleaning, data merging, data reshaping, and data aggregation 
Numpy – A Python library that is used for numerical mathematical computation and handling multidimensional ndarray, it also has a very large collection of mathematical functions to operate on this array 
Matplotlib – It is used for plotting 2D and 3D visualization plots, it also supports a variety of output formats including graphs 
Seaborn – seaborn library is made on top of Matplotlib it is used for plotting beautiful plots. 

## Loading the Dataset
We will read the dataset using the pandas read_csv() function, we can also specify the parse_dates argument which will convert the data type of the Dates column in datetime dtype. One Thing to notice initially the Dates dtype is an object. But when We change it datetime dtype it can be useful in many plotting and other computation.

## 
