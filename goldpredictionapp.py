import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# --- 1. CONFIGURATION AND DATA LOADING ---
st.set_page_config(layout="wide", page_title="Gold Price Forecasting Project")

# Define file paths based on the names provided during upload
FILE_CRUDE = 'Crude Oil WTI Futures Historical Data.csv'
FILE_GOLD = 'Gold Futures Historical Data.csv'
FILE_USD_INR = 'USD_INR Historical Data (1).csv'
FILE_DUTY = 'gold_import_duty_daily.csv'
FILE_GST = 'gst_daily.csv'
OZ_TO_GRAM = 31.1035 # Conversion factor

# Add a sidebar for interactive control
st.sidebar.header("Forecast Policy Simulation")
st.sidebar.markdown("Adjust these rates and factors to see how the 30-day price forecast changes based on potential future policy decisions.")
future_duty = st.sidebar.slider("Future Gold Import Duty (%)", min_value=0.0, max_value=15.0, value=7.5, step=0.5)
future_gst = st.sidebar.slider("Future Gold GST (%)", min_value=0.0, max_value=5.0, value=3.0, step=0.5)
# NEW: Slider for International Political Risk Score
future_political_score = st.sidebar.slider("Future International Political Risk Score (1-10)", min_value=1.0, max_value=10.0, value=5.0, step=0.1)

@st.cache_data
def load_and_preprocess_data():
    """Loads, cleans, merges, and engineers features from all necessary CSV files."""
    try:
        # Load all dataframes
        df_crude = pd.read_csv(FILE_CRUDE)
        df_gold = pd.read_csv(FILE_GOLD)
        df_usd_inr = pd.read_csv(FILE_USD_INR)
        df_duty = pd.read_csv(FILE_DUTY)
        df_gst = pd.read_csv(FILE_GST)
    except FileNotFoundError:
        st.error("One or more required data files were not found. Using simulated data for demonstration.")
        # Create a basic simulated dataframe for demonstration if files are missing
        dates = pd.date_range(start='2012-01-01', periods=1000)
        merged_df = pd.DataFrame({
            'date': dates,
            'crude_price': np.linspace(50, 80, 1000) + np.random.randn(1000) * 2,
            'gold_price': np.linspace(1500, 2500, 1000) + np.random.randn(1000) * 5,
            'gold_duty_percent': np.concatenate([np.repeat(0.3, 300), np.repeat(2.0, 300), np.repeat(6.0, 400)]),
            'gst_percent': np.concatenate([np.repeat(0.0, 500), np.repeat(3.0, 500)]),
            'usd_inr_price': np.linspace(50, 80, 1000) + np.random.randn(1000) * 0.5,
        })
        # Add simulated political score for the demo data
        np.random.seed(42)
        base_score = 5.0
        scores = [base_score]
        for _ in range(1, len(merged_df)):
            step = np.random.uniform(-0.1, 0.1)
            new_score = scores[-1] + step
            scores.append(np.clip(new_score, 1.0, 10.0))
        merged_df['political_factor_score'] = scores
        return merged_df 

    # --- Data Cleaning and Standardization ---
    
    # Crude Price: Clean 'Price' column and ensure date is correct
    df_crude.rename(columns={'Price': 'crude_price', 'Date': 'date'}, inplace=True)
    df_crude['crude_price'] = df_crude['crude_price'].replace({',': ''}, regex=True).astype(float)
    df_crude['date'] = pd.to_datetime(df_crude['date'], format='%d-%m-%Y')
    df_crude = df_crude[['date', 'crude_price']]

    # Gold Price: Clean 'Price' column and ensure date is correct
    df_gold.rename(columns={'Price': 'gold_price', 'Date': 'date'}, inplace=True)
    df_gold['gold_price'] = df_gold['gold_price'].replace({',': ''}, regex=True).astype(float)
    # The gold data uses different date formats, try to parse both
    def parse_date(date_str):
        try: return pd.to_datetime(date_str, format='%m/%d/%Y')
        except: 
            try: return pd.to_datetime(date_str, format='%Y-%m-%d')
            except: return pd.NaT 
    df_gold['date'] = df_gold['date'].apply(parse_date)
    df_gold.dropna(subset=['date'], inplace=True)
    df_gold = df_gold[['date', 'gold_price']]

    # USD/INR Price: Clean 'Price' column and ensure date is correct
    df_usd_inr.rename(columns={'Price': 'usd_inr_price', 'Date': 'date'}, inplace=True)
    df_usd_inr['usd_inr_price'] = df_usd_inr['usd_inr_price'].replace({',': ''}, regex=True).astype(float)
    df_usd_inr['date'] = pd.to_datetime(df_usd_inr['date'], format='%d-%m-%Y')
    df_usd_inr = df_usd_inr[['date', 'usd_inr_price']]
    
    # DUTY and GST files already use the 'date' column header. We just need to ensure the type.
    df_duty.rename(columns={'gold_import_duty_percent': 'gold_duty_percent'}, inplace=True) # Renaming the import duty column
    df_duty['date'] = pd.to_datetime(df_duty['date'], format='%Y-%m-%d')
    df_gst['date'] = pd.to_datetime(df_gst['date'], format='%Y-%m-%d')
    
    # --- MERGING DATA: All 'date' columns are now datetime objects ---
    
    # Start with Gold price (df_gold) and merge others
    merged_df = df_gold.merge(df_usd_inr, on='date', how='inner')
    merged_df = merged_df.merge(df_crude, on='date', how='inner')
    merged_df = merged_df.merge(df_duty, on='date', how='left') # Use left merge for taxes
    merged_df = merged_df.merge(df_gst, on='date', how='left')
    
    # Fill missing duty/gst values with the last valid observation (as they change infrequently)
    merged_df.fillna(method='ffill', inplace=True)
    merged_df.dropna(inplace=True)

    # --- SIMULATE International Political Risk Score (1-10) ---
    # This simulation provides a numerical feature for the model to train on.
    np.random.seed(42) # For reproducibility
    base_score = 5.0
    num_days = len(merged_df)
    
    # Simple simulation: start at 5 and add small random steps
    scores = [base_score]
    for _ in range(1, num_days):
        # Random step size between -0.1 and 0.1
        step = np.random.uniform(-0.1, 0.1)
        new_score = scores[-1] + step
        # Keep score constrained between 1 and 10
        new_score = np.clip(new_score, 1.0, 10.0)
        scores.append(new_score)

    merged_df['political_factor_score'] = scores[:num_days]
    
    # --- Feature Engineering ---
    
    # 1. Calculate the final target price (INR/gram)
    merged_df['indian_gold_price_proxy_oz'] = (
        merged_df['gold_price'] * merged_df['usd_inr_price'] *
        (1 + (merged_df['gold_duty_percent'] / 100) + (merged_df['gst_percent'] / 100))
    )
    merged_df['indian_gold_price_per_gram'] = merged_df['indian_gold_price_proxy_oz'] / OZ_TO_GRAM
    
    # 2. Create Lagged Features
    merged_df['target_lag_1'] = merged_df['indian_gold_price_per_gram'].shift(1)
    merged_df['gold_price_lag_1'] = merged_df['gold_price'].shift(1)
    merged_df['usd_inr_price_lag_1'] = merged_df['usd_inr_price'].shift(1)
    merged_df.dropna(inplace=True) 
    
    return merged_df.sort_values('date')

# --- 2. MODEL TRAINING AND PREDICTION ---

def train_and_predict(df, future_duty, future_gst, future_political_score):
    """Trains the Linear Regression model and generates historical and forward forecasts."""
    
    features = [
        'crude_price', 'gold_price', 'gold_duty_percent', 'gst_percent', 'usd_inr_price',
        'political_factor_score', # ADDED: New feature for political risk
        'target_lag_1', 'gold_price_lag_1', 'usd_inr_price_lag_1'
    ]
    target = 'indian_gold_price_per_gram'

    X = df[features]
    y = df[target]

    # Time-Series Split (80% Train, 20% Test)
    split_index = int(len(df) * 0.8)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    df_test = df[split_index:].copy()

    # Train Linear Regression Model
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    
    # Historical Prediction and Evaluation
    lr_pred = lr_model.predict(X_test)
    df_test['Actual Price'] = y_test
    df_test['LR Prediction'] = lr_pred
    
    mae = mean_absolute_error(y_test, lr_pred)
    rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
    r2 = r2_score(y_test, lr_pred)

    # Feature Importance (Coefficients)
    coef_df = pd.DataFrame({
        'Feature': features,
        'Coefficient': lr_model.coef_
    }).sort_values(by='Coefficient', ascending=False).set_index('Feature')
    
    # --- 30-Day Forward Forecast ---
    
    last_known_date = df['date'].max()
    forecast_start_date = last_known_date + pd.Timedelta(days=1)
    forecast_end_date = forecast_start_date + pd.Timedelta(days=30)
    forecast_dates = pd.date_range(start=forecast_start_date, end=forecast_end_date, freq='D')
    forecast_df = pd.DataFrame({'date': forecast_dates})
    forecast_df = forecast_df[~forecast_df['date'].dt.dayofweek.isin([5, 6])].reset_index(drop=True) 

    last_row = df.iloc[-1]
    last_target = last_row['indian_gold_price_per_gram']
    last_gold = last_row['gold_price']
    last_usd = last_row['usd_inr_price']
    last_crude = last_row['crude_price']
    
    # Apply user-defined future policy rates and political score
    forecast_df['gold_duty_percent'] = future_duty
    forecast_df['gst_percent'] = future_gst
    forecast_df['political_factor_score'] = future_political_score # ADDED
    
    # Initialize other forecast columns for simulation
    forecast_df['crude_price'] = 0.0
    forecast_df['gold_price'] = 0.0
    forecast_df['usd_inr_price'] = 0.0
    forecast_df['target_lag_1'] = 0.0
    forecast_df['gold_price_lag_1'] = 0.0
    forecast_df['usd_inr_price_lag_1'] = 0.0

    forecast_predictions = []
    
    for i in range(len(forecast_df)):
        
        # Random Walk Simulation for Market Drivers (small, random change from previous day)
        current_gold = last_gold * (1 + np.random.uniform(-0.001, 0.001))
        forecast_df.loc[i, 'gold_price'] = current_gold
        current_usd = last_usd * (1 + np.random.uniform(-0.0005, 0.0005))
        forecast_df.loc[i, 'usd_inr_price'] = current_usd
        current_crude = last_crude * (1 + np.random.uniform(-0.001, 0.001))
        forecast_df.loc[i, 'crude_price'] = current_crude
        
        # Update Lagged Features (using previous day's forecast/actual)
        if i == 0:
            forecast_df.loc[i, 'target_lag_1'] = last_target
            forecast_df.loc[i, 'gold_price_lag_1'] = last_gold
            forecast_df.loc[i, 'usd_inr_price_lag_1'] = last_usd
        else:
            # Use the previous day's *prediction* as the new lag feature
            forecast_df.loc[i, 'target_lag_1'] = forecast_predictions[-1]
            forecast_df.loc[i, 'gold_price_lag_1'] = forecast_df.loc[i-1, 'gold_price']
            forecast_df.loc[i, 'usd_inr_price_lag_1'] = forecast_df.loc[i-1, 'usd_inr_price']

        # Predict
        current_features_row = forecast_df.loc[i:i, features]
        current_prediction = lr_model.predict(current_features_row)[0]
        forecast_predictions.append(current_prediction)
        
        # Update 'last' values for the next iteration
        last_gold = current_gold
        last_usd = current_usd
        last_crude = current_crude

    forecast_df['Forecast Price (INR/gram)'] = forecast_predictions
    
    return df_test, forecast_df, mae, rmse, r2, coef_df

# --- 3. STREAMLIT APPLICATION LAYOUT ---

def main():
    st.title(":chart_with_upwards_trend: Indian Gold Price Forecasting (INR/gram)")
    st.markdown("---")

    data_load_state = st.text('Loading and preprocessing data...')
    
    # We pass the slider values into the function so it re-runs when sliders change
    merged_df = load_and_preprocess_data()
    data_load_state.text(f"Data loading and preprocessing complete. Data points: {len(merged_df)}")

    if 'indian_gold_price_per_gram' not in merged_df.columns:
        st.error("Data processing failed. Cannot proceed with modeling.")
        return

    st.subheader("1. Model Performance & Forecast Generation")
    
    # Train and get results (Updated function call to include future_political_score)
    df_test, forecast_df, mae, rmse, r2, coef_df = train_and_predict(merged_df, future_duty, future_gst, future_political_score)

    col1, col2, col3 = st.columns(3)
    
    # Display Metrics
    col1.metric("R-squared (R2 Score)", f"{r2:.4f}")
    col2.metric("MAE (INR/gram)", f"{mae:,.2f}")
    col3.metric("RMSE (INR/gram)", f"{rmse:,.2f}")
    
    st.warning(f"""
        **Critical Note on Error:** The high MAE of **INR {mae:,.0f}** suggests a severe **data scaling or inconsistency** issue (e.g., historical price shifts, unit mix-ups) between the training and testing data. The forecast should be interpreted with caution.""")
    st.markdown("---")


    st.subheader("2. Historical Fit and 30-Day Policy-Driven Forecast")
    
    # Get last known date for plot annotation
    last_known_date = df_test['date'].max()
    
    # Create the plot using Matplotlib
    fig, ax = plt.subplots(figsize=(15, 6))
    
    # Plot Historical Data
    ax.plot(df_test['date'], df_test['Actual Price'], label='Actual Historical Price', color='#1f77b4', linewidth=2)
    ax.plot(df_test['date'], df_test['LR Prediction'], label='LR Historical Prediction', color='#2ca02c', linestyle='--', alpha=0.7)
    
    # Plot Forecast
    ax.plot(forecast_df['date'], forecast_df['Forecast Price (INR/gram)'], label=f'30-Day Forecast (Duty: {future_duty}%, GST: {future_gst}%, Political Score: {future_political_score})', color='#9467bd', linewidth=3, linestyle='-')
    
    # Vertical line separating history and forecast
    ax.axvline(x=last_known_date, color='r', linestyle=':', label='Forecast Start')
    
    ax.set_title(f'Gold Price (INR/gram) - Forecast under Future Policy Rates & Political Risk')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price (INR/gram)')
    ax.legend()
    ax.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    st.pyplot(fig)
    st.markdown("---")
    
    # --- 30-Day Forecast Price Table ---
    st.subheader("3. 30-Day Forecast Price Table (INR/gram)")
    
    # Prepare the forecast data for display: Date and Forecast Price, rounded to 2 decimal places.
    forecast_display_df = forecast_df[['date', 'Forecast Price (INR/gram)']].copy()
    forecast_display_df['Forecast Price (INR/gram)'] = forecast_display_df['Forecast Price (INR/gram)'].map('₹{:,.2f}'.format)
    forecast_display_df.rename(columns={'date': 'Date', 'Forecast Price (INR/gram)': 'Forecast Price (INR/gram)'}, inplace=True)
    
    st.dataframe(forecast_display_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    # --- END NEW SECTION ---

    
    st.subheader("4. Linear Regression Feature Coefficients")
    
    # Display feature importance
    st.write("These values show the relative weight (influence) of each feature on the final price.")
    
    coef_df.columns = ['Coefficient Value']
    coef_df['Coefficient Value'] = coef_df['Coefficient Value'].map('{:,.4f}'.format)
    
    st.dataframe(coef_df)
    
    # --- Footer ---
    st.markdown("---")
    st.markdown("<p style='text-align: center; color: gray; font-size: small;'>© 2025 Donita Lemek | Gold Price Forecasting Project</p>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
