import numpy as np
from sklearn.linear_model import LinearRegression

# обучаем простую модель один раз при импорте
np.random.seed(42)
days = np.arange(30).reshape(-1, 1)
values = 10 + 0.5 * days.flatten() + np.random.normal(scale=1, size=30)

model = LinearRegression()
model.fit(days, values)

def forecast_time_series(n_future: int = 5):
    start = len(days)
    future_days = np.arange(start, start + n_future).reshape(-1, 1)
    preds = model.predict(future_days)
    return future_days.flatten().tolist(), preds.tolist()
