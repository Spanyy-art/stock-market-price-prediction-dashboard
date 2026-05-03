import os
import pandas as pd
import yfinance as yf

CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def fetch_stock_data(ticker, year):
    """Fetches historical data and handles caching/multi-indexing."""
    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"
    cache_path = os.path.join(CACHE_DIR, f"{ticker}_{year}.csv")

    # Check if data is already cached to save time/bandwidth
    if os.path.exists(cache_path):
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
    else:
        # Download from Yahoo Finance
        df = yf.download(ticker, start=start_date, end=end_date)
        if df.empty:
            return pd.DataFrame()
        
        # yfinance often returns MultiIndex columns; we flatten them
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        df.to_csv(cache_path)
    
    return df

def prepare_features(df):
    """Calculates Moving Averages and cleans data."""
    if df.empty:
        return df
    
    # Calculate 7-day and 21-day Moving Averages
    df['MA7'] = df['Close'].rolling(window=7).mean()
    df['MA21'] = df['Close'].rolling(window=21).mean()
    
    # Drop rows with NaN values (from rolling windows)
    df = df.dropna()
    return df

def get_prediction_alignment(df, forecast_out=30):
    """Aligns features with a future target for Linear Regression."""
    # Create a 'Prediction' column shifted 'n' units up
    df['Prediction'] = df['Close'].shift(-forecast_out)
    
    # X: Features (Close, MA7, MA21, Volume)
    X = df[['Close', 'MA7', 'MA21', 'Volume']]
    
    # X_train: Everything except the last 'forecast_out' days
    X_train = X[:-forecast_out]
    
    # y: The prediction target (the future price)
    y = df['Prediction']
    y_train = y[:-forecast_out]
    
    return X_train, y_train, X[-forecast_out:]