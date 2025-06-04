import os
from typing import Dict, Tuple, List
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import torch

from models.FFNNPriceModel import FFNNPriceModel
from models.LSTMPriceModel import LSTMPriceModel
from models.TransformerEnsemblePriceModel import TransformerEnsembleModel
from models.TransformerPriceModel import TransformerPriceModel
from util.helpers import plot_predictions
from util.realfi_info_client import RealFiInfoClient

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ModelGenerator:
    def __init__(self):
        """Initialize ModelGenerator with model types and client."""
        self.client = RealFiInfoClient()
        self.model_types = {
            'ffnn': FFNNPriceModel,
            'lstm': LSTMPriceModel,
            'transformer': TransformerPriceModel,
            'ensemble': TransformerEnsembleModel
        }
        try:
            for model_name, model_class in self.model_types.items():
                test_model = model_class()
                logging.info(f"Successfully instantiated {model_name}: {str(test_model)}")
        except Exception as e:
            logging.error(f"Failed to instantiate models: {str(e)}")
            raise
        self.training_results = {}
        self.backtest_results = {}

    def get_seq_length(self, symbol: str, model_name: str, data_length: int) -> int:
        """Determine sequence length from model class and data availability."""
        min_seq_length = 5
        max_seq_length = 20

        # Fetch seq_length from model class
        model_class = self.model_types.get(model_name)
        if model_class is None:
            logging.warning(f"Unknown model: {model_name}. Defaulting to seq_length=15")
            default_seq_length = 15
        else:
            default_seq_length = model_class.get_model_info()['hyperparameters'].get('seq_length', 15)

        # Adjust based on data length
        if data_length < 100:
            seq_length = min_seq_length
        elif data_length < 200:
            seq_length = 7
        elif data_length < 300:
            seq_length = 10
        else:
            seq_length = default_seq_length

        seq_length = min(max(seq_length, min_seq_length), max_seq_length)
        logging.debug(f"Seq_length for {symbol} ({model_name}): {seq_length} (data_length={data_length}, default={default_seq_length})")
        return seq_length

    def fetch_historical_data(self, days: int = 365) -> Dict[str, pd.DataFrame]:
        """Fetch historical data for top tokens."""
        print("Fetching historical data...")
        tokens = self.client.get_top_tokens_by_volume_df()
        historical_data = {}
        for row in tokens.itertuples():
            symbol = row.symbol
            print(f"Fetching data for {symbol}...")
            try:
                df = self.client.get_historical_data_df(row.unit_id, days=days)
                if not df.empty and len(df) >= 100:
                    price_std = df['close'].pct_change().std()
                    price_max = df['close'].max()
                    price_min = df['close'].min()
                    logging.info(f"{symbol}: {len(df)} points, Price Range=[{price_min:.6f}, {price_max:.6f}], Volatility={price_std:.6f}")
                    if price_std > 0.5 or price_max / price_min > 100:
                        logging.warning(f"High volatility or price range for {symbol}. Skipping.")
                        continue
                    historical_data[symbol] = df
                    print(f"  ✓ Got {len(df)} data points for {symbol}")
                else:
                    print(f"  ✗ Insufficient data for {symbol}")
            except Exception as e:
                logging.error(f"Error fetching data for {symbol}: {str(e)}")
                print(f"  ✗ Error fetching data for {symbol}: {str(e)}")
        return historical_data

    def time_series_split(self, n_samples: int, n_splits: int = 5) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate indices for time series cross-validation."""
        test_size = max(n_samples // (n_splits + 1), 1)
        for i in range(n_splits):
            train_end = n_samples - (n_splits - i) * test_size
            test_start = train_end
            test_end = test_start + test_size
            yield np.arange(train_end), np.arange(test_start, min(test_end, n_samples))

    def train_all_models(self, historical_data: Dict[str, pd.DataFrame]) -> Dict:
        """Train all models for each token."""
        print("\nTraining models...")
        all_results = {}
        required_cols = ['open', 'high', 'low', 'close', 'volume']

        for symbol, df in historical_data.items():
            print(f"\nTraining models for {symbol}...")
            logging.info(f"Input data for {symbol}: columns={list(df.columns)}, shape={df.shape}")
            df = df.rename(columns={col: col.lower() for col in df.columns})
            if not all(col in df.columns for col in required_cols):
                print(f"  ✗ Invalid input data for {symbol}: missing columns {required_cols}")
                logging.error(f"Missing columns for {symbol}: {list(df.columns)}")
                continue

            results = {}
            for model_name, model_class in self.model_types.items():
                logging.info(f"Attempting to train model: {model_name} for {symbol}")
                try:
                    seq_length = self.get_seq_length(symbol, model_name, len(df))
                    model = model_class(seq_length=seq_length)
                    logging.debug(f"Instantiated model {model_name} for {symbol}: {type(model)}")

                    X, y = model.prepare_data(df)
                    if X.shape[0] == 0 or y.shape[0] == 0:
                        logging.error(f"Failed to prepare data for {symbol} ({model_name}): X shape={X.shape}, y shape={y.shape}")
                        print(f"  ✗ Failed to prepare data for {symbol} ({model_name})")
                        continue

                    logging.info(f"Prepared data for {symbol} ({model_name}): X shape={X.shape}, y shape={y.shape}")

                    train_mse, test_mse, train_mae, test_mae, train_r2, test_r2 = [], [], [], [], [], []
                    train_predictions, test_predictions, actual_train, actual_test = [], [], [], []

                    n_splits = min(5, len(X) // (model.seq_length + 1))
                    if n_splits < 1:
                        logging.error(f"Insufficient data for splits in {symbol} ({model_name}): {len(X)} rows")
                        print(f"  ✗ Insufficient data for {symbol} ({model_name})")
                        continue

                    for fold_idx, (train_idx, test_idx) in enumerate(self.time_series_split(len(X), n_splits=n_splits)):
                        X_train, y_train = X[train_idx], y[train_idx]
                        X_test, y_test = X[test_idx], y[test_idx]
                        if len(X_train) < model.seq_length or len(X_test) < 1:
                            logging.warning(f"Skipping fold {fold_idx} for {symbol} ({model_name}): train={len(X_train)}, test={len(X_test)}")
                            continue

                        logging.debug(f"Fold {fold_idx} for {symbol} ({model_name}): "
                                      f"X_train shape={X_train.shape}, y_train shape={y_train.shape}, "
                                      f"X_test shape={X_test.shape}, y_test shape={y_test.shape}")

                        history = model.train_model(X_train, y_train, X_test, y_test)
                        logging.debug(f"Training completed for fold {fold_idx} of {model_name} for {symbol}")

                        train_pred = model.predict(X_train)
                        test_pred = model.predict(X_test)

                        train_metrics = model.evaluate(X_train, y_train)
                        test_metrics = model.evaluate(X_test, y_test)

                        train_mse.append(train_metrics['mse'])
                        train_mae.append(train_metrics['mae'])
                        train_r2.append(train_metrics['r2'])
                        test_mse.append(test_metrics['mse'])
                        test_mae.append(test_metrics['mae'])
                        test_r2.append(test_metrics['r2'])

                        train_predictions.extend(train_pred.tolist())
                        test_predictions.extend(test_pred.tolist())
                        actual_train.extend(y_train.flatten().tolist())
                        actual_test.extend(y_test.flatten().tolist())

                        logging.debug(f"Fold {fold_idx} for {symbol} ({model_name}): Test R²={test_metrics.get('r2', 'N/A'):.4f}")

                    if not test_r2:
                        logging.error(f"No valid folds for {symbol} ({model_name})")
                        print(f"  ✗ No valid folds for {symbol} ({model_name})")
                        continue

                    results[model_name] = {
                        'train_r2': np.mean(train_r2),
                        'test_r2': np.mean(test_r2),
                        'train_mae': np.mean(train_mae),
                        'test_mae': np.mean(test_mae),
                        'train_mse': np.mean(train_mse),
                        'test_mse': np.mean(test_mse),
                        'train_predictions': np.array(train_predictions),
                        'test_predictions': np.array(test_predictions),
                        'actual_train': np.array(actual_train),
                        'actual_test': np.array(actual_test),
                        'model': model
                    }
                    logging.info(f"{symbol} - {model_name}: Test R²={np.mean(test_r2):.4f}")
                    print(f"  ✓ Trained {model_name} for {symbol}: Test R²={np.mean(test_r2):.4f}")

                except Exception as e:
                    logging.error(f"Error training {model_name} for {symbol}: {str(e)}")
                    print(f"  ✗ Error training {model_name} for {symbol}: {str(e)}")
                    continue

            if results:
                all_results[symbol] = results
                print(f"  ✓ Successfully trained {len(results)} models for {symbol}")
                for model in results.keys():
                    try:
                        plot_predictions(symbol, model, results)
                    except Exception as e:
                        logging.error(f"Error plotting predictions for {symbol} - {model}: {str(e)}")
            else:
                print(f"  ✗ Failed to train models for {symbol}")

        self.training_results = all_results
        return all_results

    def run_backtests(self, historical_data: Dict[str, pd.DataFrame], window_size: int = 15,
                      initial_capital: float = 10000, threshold: float = 0.01) -> Dict:
        """Run backtests for trained models."""
        print("\nRunning backtests...")
        all_results = {}
        for symbol, df in historical_data.items():
            print(f"\nBacktesting models for {symbol}...")
            logging.info(f"Backtesting data for {symbol}: shape={df.shape}")
            df = df.rename(columns={col: col.lower() for col in df.columns})
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_cols):
                print(f"  ✗ Invalid input data for {symbol}: missing columns {required_cols}")
                continue

            results = {}
            for model_name in self.training_results.get(symbol, {}):
                try:
                    model = self.training_results[symbol][model_name]['model']
                    X, y = model.prepare_data(df)
                    if X.shape[0] == 0 or y.shape[0] == 0:
                        logging.warning(f"Failed to prepare data for backtesting {symbol} ({model_name})")
                        continue

                    predictions, actual_values, positions = [], [], []
                    returns = pd.Series(y.flatten()).pct_change().fillna(0).clip(-0.1, 0.1)

                    window_size = model.seq_length
                    logging.debug(f"Using window_size={window_size} for {symbol} ({model_name})")

                    for i in range(window_size, len(X) + 1):
                        X_window = X[i - window_size:i][-1]
                        if X_window.shape[0] != window_size:
                            logging.warning(f"Invalid X_window shape for {symbol} ({model_name}): {X_window.shape}")
                            continue
                        if np.any(np.isnan(X_window)):
                            logging.warning(f"NaN values in X_window for {symbol}")
                            X_window = np.nan_to_num(X_window, nan=np.nanmean(X_window))

                        pred = model.predict(X_window[np.newaxis, :])[0]
                        current_price = y[i - 1][0]
                        pred = min(max(pred, current_price * 0.9), current_price * 1.1)
                        pred_return = (pred - current_price) / current_price
                        sma_50 = np.mean(y[i - 50:i].flatten()) if i >= 50 else current_price
                        position = (1 if pred_return > threshold and current_price > sma_50 else
                                   -1 if pred_return < -threshold and current_price < sma_50 else 0)

                        predictions.append(pred)
                        actual_values.append(y[i - 1][0])
                        positions.append(position)

                    if not predictions:
                        logging.error(f"No valid predictions for {symbol} ({model_name})")
                        continue

                    predictions = np.array(predictions)
                    actual_values = np.array(actual_values)
                    positions = np.array(positions)

                    pred_tensor = torch.tensor(predictions, dtype=torch.float32)
                    actual_tensor = torch.tensor(actual_values, dtype=torch.float32)
                    mse = torch.mean((pred_tensor - actual_tensor) ** 2).item()
                    mae = torch.mean(torch.abs(pred_tensor - actual_tensor)).item()
                    ss_tot = torch.sum((actual_tensor - actual_tensor.mean()) ** 2)
                    ss_res = torch.sum((actual_tensor - pred_tensor) ** 2)
                    r2 = max(-10, (1 - ss_res / (ss_tot + 1e-8)).item())
                    directional_accuracy = np.mean(np.sign(np.diff(actual_values)) == np.sign(np.diff(predictions)))

                    transaction_cost = 0.001
                    strategy_returns = returns[-len(positions):] * positions
                    trade_indices = np.where(np.abs(np.diff(positions)) > 0)[0]
                    for idx in trade_indices:
                        strategy_returns.iloc[idx + 1] -= transaction_cost
                    strategy_returns = strategy_returns.replace([np.inf, -np.inf], 0).fillna(0).clip(-0.1, 0.1)
                    cumulative_returns = (1 + strategy_returns).cumprod()
                    total_return = cumulative_returns.iloc[-1] - 1 if len(cumulative_returns) > 0 else np.nan
                    volatility = strategy_returns.std() * np.sqrt(252) if len(strategy_returns) > 0 else np.nan
                    sharpe_ratio = (strategy_returns.mean() * 252) / volatility if volatility != 0 else np.nan
                    max_drawdown = (cumulative_returns / cumulative_returns.cummax() - 1).min()
                    num_trades = np.sum(np.abs(np.diff(positions))) if len(positions) > 1 else 0
                    win_rate = (strategy_returns > 0).mean() if len(strategy_returns) > 0 else np.nan
                    buy_signals = np.sum(positions == 1)
                    sell_signals = np.sum(positions == -1)
                    hold_signals = np.sum(positions == 0)

                    results[model_name] = {
                        'mse': mse,
                        'mae': mae,
                        'r2': r2,
                        'directional_accuracy': directional_accuracy,
                        'total_return': total_return,
                        'volatility': volatility,
                        'sharpe_ratio': sharpe_ratio,
                        'max_drawdown': max_drawdown,
                        'num_trades': num_trades,
                        'win_rate': win_rate,
                        'buy_signals': buy_signals,
                        'sell_signals': sell_signals,
                        'hold_signals': hold_signals,
                        'predictions': predictions,
                        'actual': actual_values,
                        'positions': positions,
                        'strategy_returns': strategy_returns,
                        'cumulative_returns': cumulative_returns
                    }
                    print(f"  ✓ Backtest completed for {symbol} - {model_name}: Total Return={total_return:.4f}, Sharpe Ratio={sharpe_ratio:.4f}")
                    logging.info(f"Backtest completed for {symbol} - {model_name}: Total Return={total_return:.4f}, Sharpe Ratio={sharpe_ratio:.4f}")
                except Exception as e:
                    logging.error(f"Error backtesting {model_name} for {symbol}: {str(e)}")
                    print(f"  ✗ Error backtesting {model_name} for {symbol}: {str(e)}")
                    continue

            if results:
                all_results[symbol] = results
            else:
                print(f"  ✗ No backtest results for {symbol}")

        self.backtest_results = all_results
        return all_results

    def filter_and_rank_models(self, historical_data: Dict[str, pd.DataFrame]) -> Tuple[List[Dict], List[Dict]]:
        """Filter and rank models based on performance criteria."""
        sorted_models, excluded_models = [], []
        for symbol, models in self.training_results.items():
            if symbol not in historical_data:
                excluded_models.extend([{'symbol': symbol, 'model': model_name, 'reason': 'No historical data available'} for model_name in models.keys()])
                continue

            price_scale = historical_data[symbol]['close'].mean()
            mae_threshold = max(price_scale * 0.3, 0.001)

            for model_name, metrics in models.items():
                test_r2 = metrics.get('test_r2', -float('inf'))
                test_mae = metrics.get('test_mae', float('inf'))
                backtest_metrics = self.backtest_results.get(symbol, {}).get(model_name, {})
                backtest_r2 = backtest_metrics.get('r2', -float('inf'))
                backtest_mae = backtest_metrics.get('mae', float('inf'))
                total_return = backtest_metrics.get('total_return', -float('inf'))
                sharpe_ratio = backtest_metrics.get('sharpe_ratio', -float('inf'))
                num_trades = backtest_metrics.get('num_trades', 0)

                if total_return < -0.3 or backtest_r2 < -5.0 or num_trades < 5:
                    excluded_models.append({
                        'symbol': symbol,
                        'model': model_name,
                        'reason': f"Total Return={total_return:.6f}, Backtest R²={backtest_r2:.4f}, Num Trades={num_trades}"
                    })
                    continue

                if (backtest_r2 >= -1.0 or total_return >= 0.05 or sharpe_ratio >= 0.3) and num_trades >= 5:
                    performance = {
                        'symbol': symbol,
                        'model': model_name,
                        'test_r2': test_r2,
                        'test_mae': test_mae,
                        'backtest_r2': backtest_r2,
                        'backtest_mae': backtest_mae,
                        'total_return': total_return,
                        'sharpe_ratio': sharpe_ratio,
                        'num_trades': num_trades,
                        'both_criteria_passed': (test_r2 >= -2.0 and test_mae <= mae_threshold) and
                                                (backtest_r2 >= -1.0 and backtest_mae <= mae_threshold)
                    }
                    performance['status'] = ('✓ Cross-validation' if test_r2 >= -2.0 and test_mae <= mae_threshold else '') + \
                                            (' ✓ Backtest' if backtest_r2 >= -1.0 and backtest_mae <= mae_threshold else '')
                    sorted_models.append(performance)
                else:
                    excluded_models.append({
                        'symbol': symbol,
                        'model': model_name,
                        'reason': f"Failed criteria: Test R²={test_r2:.4f}, Test MAE={test_mae:.6f}, Backtest R²={backtest_r2:.4f}, Backtest MAE={backtest_mae:.6f}, Total Return={total_return:.6f}, Sharpe Ratio={sharpe_ratio:.4f}, Num Trades={num_trades}"
                    })

        sorted_models.sort(key=lambda x: (x['total_return'], x['sharpe_ratio'], x['backtest_r2'], x['test_r2'], -x['backtest_mae'], -x['test_mae']), reverse=True)
        return sorted_models, excluded_models

    def generate_performance_report(self, historical_data: Dict[str, pd.DataFrame]) -> str:
        """Generate a performance report for all models."""
        report = []
        report.append("=" * 80)
        report.append("CARDANO TOKEN PRICE PREDICTION MODEL PERFORMANCE REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        total_tokens = len(self.training_results)
        total_models = sum(len(models) for models in self.training_results.values())
        report.append(f"SUMMARY:\n  - Tokens analyzed: {total_tokens}\n  - Total models trained: {total_models}\n")
        report.append("DATA STATISTICS:\n" + "-" * 40)
        for symbol, df in historical_data.items():
            report.append(f"\n{symbol}:\n  - Data points: {len(df)}\n  - Date range: {df.index[0]} to {df.index[-1]}\n  - Mean close price: {df['close'].mean():.6f}\n  - Std close price: {df['close'].std():.6f}")
        report.append("\nTRAINING RESULTS (Cross-validation):\n" + "-" * 40)
        for symbol, models in self.training_results.items():
            if symbol not in historical_data:
                continue
            report.append(f"\n{symbol}:")
            best_model = max(models.keys(), key=lambda k: models[k]['test_r2']) if models else None
            price_scale = historical_data[symbol]['close'].mean()
            mae_threshold = max(price_scale * 0.3, 0.001)
            for model_name, metrics in models.items():
                test_r2 = metrics.get('test_r2', -float('inf'))
                test_mae = metrics.get('test_mae', float('inf'))
                status = "✓" if test_r2 >= -2.0 and test_mae <= mae_threshold else "✗"
                marker = " ★" if model_name == best_model else "  "
                report.append(f"{marker} {model_name} (Cross-validation: {status}):")
                report.append(f"    Train R²: {metrics['train_r2']:.4f}\n    Test R²:  {test_r2:.4f}\n    Test MAE: {test_mae:.6f} (Threshold: {mae_threshold:.6f})\n    Test MSE: {metrics['test_mse']:.6f}")

        if self.backtest_results:
            report.append("\n\nBACKTEST RESULTS:\n" + "-" * 40)
            for symbol, models in self.backtest_results.items():
                if symbol not in historical_data:
                    continue
                report.append(f"\n{symbol}:")
                best_backtest = max(models.keys(), key=lambda k: models[k]['total_return']) if models else None
                price_scale = historical_data[symbol]['close'].mean()
                mae_threshold = max(price_scale * 0.3, 0.001)
                for model_name, metrics in models.items():
                    backtest_r2 = metrics.get('r2', -float('inf'))
                    backtest_mae = metrics.get('mae', float('inf'))
                    status = "✓" if backtest_r2 >= -1.0 and backtest_mae <= mae_threshold else "✗"
                    marker = " ★" if model_name == best_backtest else "  "
                    report.append(f"{marker} {model_name} (Backtest: {status}):")
                    report.append(f"    Backtest R²: {backtest_r2:.4f}\n    Backtest MAE: {backtest_mae:.6f} (Threshold: {mae_threshold:.6f})\n    Total Return: {metrics['total_return']:.4f}\n    Volatility: {metrics['volatility']:.4f}\n    Sharpe Ratio: {metrics['sharpe_ratio']:.4f}\n    Max Drawdown: {metrics['max_drawdown']:.4f}\n    Number of Trades: {metrics['num_trades']}\n    Win Rate: {metrics['win_rate']:.4f}\n    Signal Distribution: BUY={metrics['buy_signals']}, SELL={metrics['sell_signals']}, HOLD={metrics['hold_signals']}")

        sorted_models, excluded_models = self.filter_and_rank_models(historical_data)
        report.append("\n\nMODEL RANKINGS:\n" + "-" * 40)
        report.append("\nTop Models by Total Return, Sharpe Ratio, Backtest R², Test R², MAE:")
        if sorted_models:
            for i, model in enumerate(sorted_models[:10], 1):
                report.append(f"{i:2d}. {model['symbol']} - {model['model']} ({model['status']}): Total Return = {model['total_return']:.6f}, Sharpe Ratio = {model['sharpe_ratio']:.4f}, Backtest R² = {model['backtest_r2']:.4f}, Test R² = {model['test_r2']:.4f}, Backtest MAE = {model['backtest_mae']:.6f}, Test MAE = {model['test_mae']:.6f}")
        else:
            report.append("  No models met performance criteria.")
        if excluded_models:
            report.append("\n\nEXCLUDED MODELS:\n" + "-" * 40)
            for model in excluded_models:
                report.append(f"  - {model['symbol']} - {model['model']}: {model['reason']}")
        report.append("\n\nRECOMMENDATIONS:\n" + "-" * 40)
        best_model = sorted_models[0] if sorted_models else None
        if best_model:
            report.append(f"Best overall model: {best_model['symbol']} - {best_model['model']}\n  Total Return: {best_model['total_return']:.6f}, Sharpe Ratio: {best_model['sharpe_ratio']:.4f}, Backtest R²: {best_model['backtest_r2']:.4f}, Test R²: {best_model['test_r2']:.4f}")
        report.append("\nBest model per token:")
        for symbol in sorted(self.training_results.keys()):
            valid_models = [p for p in sorted_models if p['symbol'] == symbol]
            if valid_models:
                best = max(valid_models, key=lambda x: (x['total_return'], x['sharpe_ratio'], x['backtest_r2'], x['test_r2'], -x['backtest_mae'], -x['test_mae']))
                report.append(f"  {symbol}: {best['model']} (Total Return = {best['total_return']:.6f}, Sharpe Ratio = {best['sharpe_ratio']:.4f}, Backtest R² = {best['backtest_r2']:.4f}, Test R² = {best['test_r2']:.4f}, Status: {best['status']})")
            else:
                report.append(f"  {symbol}: No models met performance criteria.")
        report.append("\n" + "=" * 80)
        return "\n".join(report)

    def save_results(self, historical_data: Dict[str, pd.DataFrame], output_dir: str = "model_results"):
        """Save model results and performance report."""
        os.makedirs(output_dir, exist_ok=True)
        sorted_models, _ = self.filter_and_rank_models(historical_data)
        model_properties = []
        model_path = os.path.join(output_dir, "cardano_models")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        model_properties.append("# Model manifest for recommended models")
        model_properties.append(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        model_properties.append("# Format: symbol.model_name.key=value")
        model_properties.append("# Performance metrics included as comments")
        model_properties.append("")

        for model in sorted_models:
            symbol = model['symbol']
            model_name = model['model']
            try:
                model_instance = self.training_results[symbol][model_name]['model']
                model_file = f"{model_path}_{symbol}_{model_name}.onnx"
                model_instance.save_onnx(model_file)
                model_properties.append(f"# {symbol} - {model_name}")
                model_properties.append(f"# Test R²: {model['test_r2']:.4f}, Test MAE: {model['test_mae']:.6f}, Backtest R²: {model['backtest_r2']:.4f}, Backtest MAE: {model['backtest_mae']:.6f}, Total Return: {model['total_return']:.6f}, Sharpe Ratio: {model['sharpe_ratio']:.4f}")
                model_properties.append(f"{symbol}.{model_name}.path={model_file}")
                model_properties.append("")
            except Exception as e:
                logging.error(f"Error saving model {symbol} - {model_name}: {str(e)}")

        properties_path = os.path.join(output_dir, "model.properties")
        with open(properties_path, 'w') as f:
            f.write("\n".join(model_properties))
        logging.info(f"Saved model properties to {properties_path}")

        report = self.generate_performance_report(historical_data)
        report_path = os.path.join(output_dir, "performance_report.txt")
        with open(report_path, 'w') as f:
            f.write(report)

        print(f"\nResults saved to {output_dir}/")
        print(f"  - Models: {model_path}*.onnx ({len(sorted_models)} models saved)")
        print(f"  - Model properties: {properties_path}")
        print(f"  - Performance report: {report_path}")

    def run_full_training_pipeline(self, days: int = 365, output_dir: str = "model_results"):
        """Run the full training pipeline."""
        print("Starting full training pipeline...")
        print(f"Historical data period: {days} days")
        historical_data = self.fetch_historical_data(days)
        if not historical_data:
            print("No historical data available. Exiting.")
            return
        print(f"Successfully fetched data for {len(historical_data)} tokens")
        training_results = self.train_all_models(historical_data)
        if not training_results:
            print("No models were successfully trained. Exiting.")
            return
        backtest_results = self.run_backtests(historical_data)
        report = self.generate_performance_report(historical_data)
        print("\n" + report)
        self.save_results(historical_data, output_dir)
        print(f"\nTraining pipeline completed successfully!")
        print(f"Trained models for {len(training_results)} tokens")
        print(f"Results saved to '{output_dir}' directory")
        return {
            'historical_data': historical_data,
            'training_results': training_results,
            'backtest_results': backtest_results,
            'report': report
        }

def main():
    """Main function to run the training pipeline."""
    trainer = ModelGenerator()
    results = trainer.run_full_training_pipeline(days=750)
    if results:
        print("\nTraining completed! Models are ready for live streaming.")
        print("Next steps:\n1. Review the performance report\n2. Run live_signalling_app.py to start live predictions")

if __name__ == "__main__":
    main()