import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

class ModelEvaluator:

    def __init__(self):
        self.results = {}

    def add_model(self, model_name, actuals, predictions):
        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        mae = mean_absolute_error(actuals, predictions)
        self.results[model_name] = {
            'actuals': actuals,
            'predictions': predictions,
            'rmse': round(rmse, 4),
            'mae': round(mae, 4)
        }

    def get_comparison_table(self):
        rows = []
        for model_name, metrics in self.results.items():
            rows.append({
                'model': model_name,
                'rmse': metrics['rmse'],
                'mae': metrics['mae']
            })
        return pd.DataFrame(rows).sort_values('rmse')

    def plot_predictions(self, model_name):
        data = self.results[model_name]
        plt.figure(figsize=(12, 4))
        plt.plot(data['actuals'], label='Actual', color='black')
        plt.plot(data['predictions'], label=f'{model_name} Predicted', color='blue')
        plt.title(f'{model_name} — Actual vs Predicted')
        plt.legend()
        plt.tight_layout()
        plt.show()