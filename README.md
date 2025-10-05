
# 🪙 Gold Price Forecasting Web App
### 1. Project Description
This is an interactive Streamlit web application that utilizes time-series analysis and Linear Regression to predict the Indian retail price of gold (INR/gram).

Unlike simple historical models, this application integrates key macroeconomic and policy variables to create a robust pricing model. Crucially, the application allows users to simulate the impact of future policy changes (Import Duty, GST) and external risk factors on the 30-day forecast.

The primary goal is to provide a comprehensive tool for analyzing the complex drivers of India's gold market, moving beyond simple correlation to offer actionable, policy-sensitive predictions.

### 2. Methodology & Features
Core Methodology: Linear Regression with Lagged Features
The model uses a Linear Regression approach, specifically relying on lagged features. This means the model's highest predictor for today's price is the price from the previous day. This method effectively captures the daily randomness inherent in financial markets while allowing external factors to influence the prediction's slope.

#### Key Interactive Features
The application provides an interactive sidebar that allows users to manipulate three critical variables that drive the forecast:

Import Duty (%): Adjusts the gold import duty rate.

GST (%): Adjusts the Goods and Services Tax rate applied to gold.

International Political Risk Score (1-10): A simulated factor that proxies global geopolitical or economic uncertainty (higher risk generally pushes gold prices up).

The main interface displays a time-series plot of the historical data, the model's fit, and the simulated 30-day forecast, alongside a detailed daily forecast table.

### 3. Technical Stack
The application is built entirely in Python using the following libraries:

Library

Purpose

Streamlit

Front-end web application framework (GUI).

Pandas & NumPy

Data manipulation, cleaning, and numerical calculations.

Scikit-learn

Machine learning model implementation (LinearRegression).

Matplotlib

Data visualization and plotting.

###4. How to Run Locally
To run this application on your local machine, follow these steps:

#### Clone the Repository:

git clone donita-lemek/goldpriceprediction/
cd goldpriceprediction

Install Dependencies: Ensure you have Python installed, then install the required libraries using the requirements.txt file:

pip install -r requirements.txt

Data Files: Verify that the following five CSV files are present in the root directory:

Crude Oil WTI Futures Historical Data.csv

Gold Futures Historical Data.csv

gold_import_duty_daily.csv

gst_daily.csv

USD_INR Historical Data (1).csv

#### Run the App:

streamlit run streamlit_app.py

The application will automatically open in your web browser.

<p align="center">
&copy; 2025 Donita Lemek. All rights reserved.
</p>
