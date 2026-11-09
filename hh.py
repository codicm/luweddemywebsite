import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

class TimeSeriesDemandForecaster:
    """
    Time-Series Predictive Maintenance & Demand Forecasting Pipeline
    Author: Sandra Luwedde
    """
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.08, random_state=42)

    def engineer_temporal_features(self) -> tuple:
        # Simulate time-series dataset with lag and rolling statistics
        np.random.seed(42)
        dates = pd.date_range(start="2025-01-01", periods=365, freq="D")
        base_demand = 500 + 50 * np.sin(np.linspace(0, 2 * np.pi, 365)) + np.random.normal(0, 20, 365)

        df = pd.DataFrame({"date": dates, "demand": base_demand})

        # Temporal lag & rolling window engineering
        df['lag_1'] = df['demand'].shift(1)
        df['lag_7'] = df['demand'].shift(7)
        df['rolling_mean_7'] = df['demand'].rolling(window=7).mean()
        df.dropna(inplace=True)

        X = df[['lag_1', 'lag_7', 'rolling_mean_7']]
        y = df['demand']

        # Temporal split (preventing future data leakage)
        split_idx = int(len(df) * 0.8)
        return X.iloc[:split_idx], X.iloc[split_idx:], y.iloc[:split_idx], y.iloc[split_idx:]

    def train_and_forecast(self, X_train, X_test, y_train, y_test) -> dict:
        self.model.fit(X_train, y_train)
        predictions = self.model.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))

        return {"MAE": round(mae, 2), "RMSE": round(rmse, 2)}

if __name__ == "__main__":
    forecaster = TimeSeriesDemandForecaster("data/time_series.csv")
    X_train, X_test, y_train, y_test = forecaster.engineer_temporal_features()
    metrics = forecaster.train_and_forecast(X_train, X_test, y_train, y_test)
    print(f"Time-Series Forecast Metrics: {metrics}")