import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def run_prediction_engine(X_train, y_train, X_forecast):
    """Trains a Linear Regression model and predicts the future."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Confidence Score (R-squared)
    confidence = r2_score(y_train, model.predict(X_train))
    
    # Forecast the next 30 days
    forecast_prediction = model.predict(X_forecast)
    
    return forecast_prediction, confidence

    # Features (X) and labels (y)
    X = np.array(range(len(data))).reshape(-1, 1)
    y = data['Close'].values

    # Remove last 30 days for training
    X_train = X[:-30]
    y_train = y[:-30]

    model = LinearRegression()
    model.fit(X_train, y_train)

    return model