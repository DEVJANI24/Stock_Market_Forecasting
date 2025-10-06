# 📈 Stock Price Forecasting – TCS (Tata Consultancy Services)

This project predicts the stock prices of **Tata Consultancy Services (TCS)** using multiple **time series forecasting techniques**.  
It compares the performance of **ARIMA**, **SARIMA**, **Facebook Prophet**, and **LSTM** models on real-world stock market data fetched from Yahoo Finance.

---

## 📌 Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Models Implemented](#models-implemented)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Future Improvements](#future-improvements)
- [Author](#author)

---

## 🔎 Overview
Stock price prediction is a classic time series problem that involves analyzing historical trends to forecast future prices.  
In this project, we use different statistical and deep learning models to predict the **closing prices of TCS stock** between 2015 and 2025.

The notebook covers:
- Data collection using **Yahoo Finance**
- Data preprocessing and visualization
- Training and evaluating multiple forecasting models
- Comparing their performance using **RMSE (Root Mean Square Error)**

---

## 📊 Dataset
- **Ticker Symbol:** `TCS.NS`  
- **Company:** Tata Consultancy Services  
- **Time Period:** 2015-01-01 to 2025-01-01  
- **Data Source:** [Yahoo Finance](https://finance.yahoo.com/)  
- **Column Used:** `Close` (Daily closing price)  

---

## 🤖 Models Implemented

### 1. **ARIMA (AutoRegressive Integrated Moving Average)**
Used for univariate time series forecasting by combining autoregressive and moving average components.

### 2. **SARIMA (Seasonal ARIMA)**
An extension of ARIMA that models both trend and seasonality (yearly cycles in stock data).

### 3. **Facebook Prophet**
A powerful model developed by Meta, useful for capturing daily/weekly trends and seasonality in time series.

### 4. **LSTM (Long Short-Term Memory)**
A recurrent neural network model designed to learn long-term dependencies in sequential data.

---

## ✨ Features
✅ Automatic stock data download using `yfinance`  
✅ Data preprocessing (date formatting, scaling, and splitting)  
✅ Time series visualizations (line charts, forecast plots)  
✅ RMSE-based model performance comparison  
✅ Combined forecast visualization of all models  

---

## 🛠 Technologies Used
- **Programming Language:** Python  
- **Development Environment:** Jupyter Notebook  

**Libraries Used:**
- `pandas`, `numpy` → Data handling and manipulation  
- `matplotlib`, `seaborn` → Data visualization  
- `statsmodels` → ARIMA & SARIMA models  
- `prophet` → Facebook Prophet forecasting  
- `scikit-learn` → Metrics and scaling  
- `tensorflow.keras` → LSTM neural network  
- `yfinance` → Stock market data fetching  

-------------------------------------------

## ⚙️ Installation

### 1. Clone this repository
``bash
git clone https://github.com/your-username/stock-forecasting-tcs.git
cd stock-forecasting-tcs

**Install dependencies**
pip install -r requirements.txt
--------------------------------------------
***requirement.txt***
pandas
numpy
matplotlib
seaborn
statsmodels
prophet
scikit-learn
tensorflow
yfinance

------------------------------------------
**Usage**
Launch Jupyter Notebook:
>jupyter notebook

Open the notebook file:
updated.ipynb

-----------------------------------------
**Results**
## After training all models, the notebook prints the RMSE (Root Mean Square Error) for each:

--Model	Description	RMSE (Example)
  >ARIMA	Traditional statistical model	~668.96
  >SARIMA	Seasonal ARIMA model	~380.98
  >Prophet	Daily trend-based forecasting	~671.23
  >LSTM	Deep learning sequential model	~1508.19

## 📊  The final comparison plot displays:

Black line → Actual closing prices

Blue line → ARIMA forecast

Green line → SARIMA forecast

Red line → Prophet forecast

Orange line → LSTM forecast
-----------------------------------------

**Future Improvements**

  -Tune model hyperparameters for better accuracy

  -Add new models (e.g., XGBoost, Transformer-based forecasting)

  -Implement automated backtesting

  -Build a real-time dashboard using Streamlit or Flask

  -Add functionality for predicting other stocks dynamically
  
---------------------------------------------
**Author**

-Name: DEV JANI
-Degree: B.Sc. Computer Science, VNSGU University
-Project Type: Time Series Forecasting (Stock Market)

-------------------------------------------------
