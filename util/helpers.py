import json
import matplotlib.pyplot as plt
import os
import logging
import numpy as np

# Custom JSON encoder for NumPy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

def plot_predictions(symbol: str, model_name: str, results: dict):
    """Plot prediction results."""
    try:
        plt.figure(figsize=(15, 10))

        plt.subplot(2, 2, 1)
        plt.plot(results[model_name]['actual_train'], label='Actual', alpha=0.7)
        plt.plot(results[model_name]['train_predictions'], label='Predicted', alpha=0.7)
        plt.title(f'{symbol} - {model_name} Training Predictions')
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 2, 2)
        plt.plot(results[model_name]['actual_test'], label='Actual', alpha=0.7)
        plt.plot(results[model_name]['test_predictions'], label='Predicted', alpha=0.7)
        plt.title(f'{symbol} - {model_name} Test Predictions')
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 2, 3)
        plt.scatter(results[model_name]['actual_train'], results[model_name]['train_predictions'], alpha=0.5)
        plt.plot([results[model_name]['actual_train'].min(), results[model_name]['actual_train'].max()],
                 [results[model_name]['actual_train'].min(), results[model_name]['actual_train'].max()], 'r--')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title(f'{symbol} - {model_name} Training Scatter')
        plt.grid(True)

        plt.subplot(2, 2, 4)
        plt.scatter(results[model_name]['actual_test'], results[model_name]['test_predictions'], alpha=0.5)
        plt.plot([results[model_name]['actual_test'].min(), results[model_name]['actual_test'].max()],
                 [results[model_name]['actual_test'].min(), results[model_name]['actual_test'].max()], 'r--')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title(f'{symbol} - {model_name} Test Scatter')
        plt.grid(True)

        plt.tight_layout()
        os.makedirs('model_results/img', exist_ok=True)
        plt.savefig(f'model_results/img/{symbol}_{model_name}_predictions.png', dpi=300, bbox_inches='tight')
        plt.close()

    except Exception as e:
        logging.error(f"Error plotting predictions for {symbol} ({model_name}): {str(e)}")