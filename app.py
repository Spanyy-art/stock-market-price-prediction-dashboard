import sys
sys.path.insert(0, 'path_to_src_directory')
import os
import numpy as np
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import yfinance as yf
import plotly.graph_objects as go
import pandas as pd

# Import our custom modules
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import yfinance as yf
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from model import run_prediction_engine

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])

# App Layout
app.layout = dbc.Container([
    dbc.Row([
        html.H1(
            "Stock Market Prediction Dashboard",
            style={"textAlign": "center", "marginTop": "20px"}
        )
    ]),
    
    dbc.Row([
        dbc.Col([
            html.Label("Select Stock:"),
            dcc.Dropdown(
                id="stock-picker",
                options=[
                    {"label": "Apple (AAPL)", "value": "AAPL"},
                    {"label": "Google (GOOG)", "value": "GOOG"},
                    {"label": "Tesla (TSLA)", "value": "TSLA"},
                    {"label": "Microsoft (MSFT)", "value": "MSFT"},
                    {"label": "Amazon (AMZN)", "value": "AMZN"},
                    {"label": "NVIDIA (NVDA)", "value": "NVDA"},
                    {"label": "Meta (META)", "value": "META"},
                    {"label": "Netflix (NFLX)", "value": "NFLX"},
                    {"label": "Intel (INTC)", "value": "INTC"},
                    {"label": "AMD (AMD)", "value": "AMD"}
                ],
                value="AAPL",
                style={"color": "black"}
            )
        ], width=6),

        dbc.Col([
            html.Label("Select Year:"),
            dcc.Slider(
                id="year-slider",
                min=2018,
                max=2025,
                step=1,
                value=2018,
                marks={
    year: {
        "label": str(year),
        "style": {"color": "white"}
    } for year in range(2018, 2026)
}
            )
        ], width=6),
    ], style={"marginBottom": "30px"}),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Model Confidence (R²)", className="card-title"),
                    html.H3(id="confidence-score", style={"color": "#00ffcc"})
                ])
            ], color="dark", outline=True)
        ], width=4),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Current Price", className="card-title"),
                    html.H3(id="current-price", style={"color": "#ffcc00"})
                ])
            ], color="dark", outline=True)
        ], width=4),
    ], className="mb-4", justify="center"),

    dbc.Row([
        dbc.Col([
            dcc.Graph(id="stock-chart")
        ])
    ]),

    dbc.Row([
        html.Div(
            "Created by Spany | Stock Prediction Dashboard",
            style={
                "textAlign": "center",
                "marginTop": "20px",
                "marginBottom": "20px"
            }
        )
    ])
], fluid=True)


# Callback

@app.callback(
    [Output("stock-chart", "figure"),
     Output("confidence-score", "children"),
     Output("current-price", "children")],
    [Input("stock-picker", "value"),
     Input("year-slider", "value")]
)
def update_graph(stock, year):
    # Load stock data
    df = yf.download(stock, start=f"{year}-01-01", end=f"{year}-12-31")

    # Moving averages
    df["MA7"] = df["Close"].rolling(window=7).mean()
    df["MA21"] = df["Close"].rolling(window=21).mean()

    # Clean data
    df = df.dropna()

    # Prediction setup
    future_days = 30
    df["Prediction"] = df["Close"].shift(-future_days)

    # Features
    features = df[["MA7", "MA21", "Volume"]].dropna()
    target = df["Prediction"].dropna()

    # Match lengths
    min_len = min(len(features), len(target))

    X = features.iloc[:min_len].values
    y = target.iloc[:min_len].values

    # Train model
    model = LinearRegression()
    model.fit(X, y)

    # Accuracy
    predictions_train = model.predict(X)
    accuracy = r2_score(y, predictions_train) * 100

    # Future prediction
    last_features = features.iloc[-1].values.reshape(1, -1)
    future_predictions = np.repeat(
        model.predict(last_features),
        future_days
    )

    # Future dates
    future_dates = pd.date_range(
        start=df.index[-1] + pd.Timedelta(days=1),
        periods=future_days
    )

           # Current price (universal safe extraction)
    last_close = df["Close"].iloc[-1]

    if isinstance(last_close, pd.Series):
        current_price = float(last_close.iloc[0])
    else:
        current_price = float(last_close)

    # Create figure
    fig = go.Figure()

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        name="Candlestick"
    ))

    # MA7
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df["MA7"],
        mode="lines",
        name="7-Day MA"
    ))

    # MA21
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df["MA21"],
        mode="lines",
        name="21-Day MA"
    ))

    # Prediction
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=future_predictions,
        mode="lines",
        name="30-Day Prediction",
        line=dict(dash="dash")
    ))

    # Layout
    fig.update_layout(
        title=f"{stock} Stock Price ({year}) + Prediction | Model Confidence: {accuracy:.2f}%",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        xaxis_rangeslider_visible=False,
        height=700,
        hovermode="x unified"
    )

    return fig, f"Model Confidence: {accuracy:.2f}%", f"Current Price: ${current_price:.2f}"

# Run App
if __name__ == "__main__":
    app.run(debug=True)