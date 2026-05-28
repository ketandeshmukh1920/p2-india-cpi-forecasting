import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

class WalkForwardValidator:

    def __init__(self, model, X, y, test_start):
        self.model = model
        self.X = X
        self.y = y
        self.test_start = test_start
        self.predictions = None
        self.actuals = None

    def run(self):
        predictions = []
        actuals = []
        test_indices = self.X[self.test_start:].index

        for date in test_indices:
            train_X = self.X[:date].iloc[:-1]
            train_y = self.y[:date].iloc[:-1]
            test_X = self.X.loc[[date]]

            self.model.fit(train_X, train_y)
            pred = self.model.predict(test_X)[0]

            predictions.append(pred)
            actuals.append(self.y.loc[date])

        self.predictions = pd.Series(predictions, index=test_indices)
        self.actuals = pd.Series(actuals, index=test_indices)
        return self.predictions

    def get_metrics(self):
        rmse = np.sqrt(mean_squared_error(self.actuals, self.predictions))
        mae = mean_absolute_error(self.actuals, self.predictions)
        return {'rmse': round(rmse, 4), 'mae': round(mae, 4)}
    

#Another class for ARIMA model

from statsmodels.tsa.arima.model import ARIMA

class ARIMAValidator:

    def __init__(self, y, order, test_start, exog=None):
        self.y = y
        self.order = order
        self.test_start = test_start
        self.exog = exog
        self.predictions = None
        self.actuals = None

    def run(self):
        predictions = []
        actuals = []
        test_indices = self.y[self.test_start:].index

        for date in test_indices:
            train_y = self.y[:date].iloc[:-1]
            model_exog = None
            forecast_exog = None

            if self.exog is not None:
                model_exog = self.exog[:date].iloc[:-1]
                forecast_exog = self.exog.loc[[date]]

            model = ARIMA(train_y, order=self.order, exog=model_exog)
            fitted = model.fit()
            pred = fitted.forecast(steps=1, exog=forecast_exog)[0]
            predictions.append(pred)
            actuals.append(self.y.loc[date])

        self.predictions = pd.Series(predictions, index=test_indices)
        self.actuals = pd.Series(actuals, index=test_indices)
        return self.predictions

    def get_metrics(self):
        rmse = np.sqrt(mean_squared_error(self.actuals, self.predictions))
        mae = mean_absolute_error(self.actuals, self.predictions)
        return {'rmse': round(rmse, 4), 'mae': round(mae, 4)}