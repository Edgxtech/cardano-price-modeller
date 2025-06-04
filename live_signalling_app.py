import os
import time
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
from dotenv import load_dotenv
from dataclasses import dataclass, asdict
from collections import deque
import signal
import sys
import json
import onnxruntime as ort
from util.realfi_info_client import RealFiInfoClient
from util.helpers import NumpyEncoder
from sklearn.preprocessing import MinMaxScaler

load_dotenv()

@dataclass
class PredictionRecord:
    timestamp: str
    symbol: str
    model_name: str
    current_price: float
    predicted_price: float
    signal: str
    confidence: float
    actual_price: Optional[float] = None
    prediction_accuracy: Optional[float] = None

@dataclass
class PerformanceMetrics:
    symbol: str
    model_name: str
    last_updated: str
    total_predictions: int = 0
    correct_predictions: int = 0
    accuracy: float = 0
    avg_error: float = 0
    signals_generated: Dict[str, int] = None

    def __post_init__(self):
        if self.signals_generated is None:
            self.signals_generated = {'BUY': 0, 'SELL': 0, 'NA': 0}

class LiveSignallingApp:
    def __init__(self, update_interval: float = 20, model_path: str = "model_results"):
        self.client = RealFiInfoClient()
        self.update_interval = update_interval
        self.model_path = model_path
        self.models = {}  # {symbol: {model_name: {'session': ort_session, 'scaler_X': MinMaxScaler, 'scaler_y': MinMaxScaler, 'seq_length': int}}}
        self.active_tokens = []
        self.setup_logging()
        self.load_models()
        if not self.active_tokens:
            self.logger.warning(f"No models loaded from {self.model_path}. Check model.properties file.")
        self.prediction_history = deque(maxlen=1000)
        self.price_history = {}
        self.performance_metrics = {}
        self.running = False
        self.last_signals = {}
        self.signal_price_threshold = 0.01
        for token in self.active_tokens:
            self.price_history[token] = deque(maxlen=100)
            self.performance_metrics[token] = {}
            for model_name in self.models[token].keys():
                self.performance_metrics[token][model_name] = PerformanceMetrics(
                    symbol=token,
                    model_name=model_name,
                    last_updated=datetime.now().isoformat()
                )
        print(f"Initialized live signalling app for {len(self.active_tokens)} tokens")
        print(f"Update interval: {self.update_interval:.2f} seconds")

    def setup_logging(self):
        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        root_logger.setLevel(logging.WARNING)

        self.logger = logging.getLogger('LiveSignallingApp')
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        self.logger.handlers = []

        file_handler = logging.FileHandler('logs/app.log')
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(file_handler)
        self.logger.addHandler(stream_handler)

        self.signal_logger = logging.getLogger('signals')
        self.signal_logger.setLevel(logging.INFO)
        self.signal_logger.propagate = False
        self.signal_logger.handlers = []
        signal_handler = logging.FileHandler('logs/trading_signals.log')
        signal_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        self.signal_logger.addHandler(signal_handler)

    def load_models(self):
        """Load ONNX models and scalers from model.properties, extracting seq_length from model inputs."""
        properties_file = os.path.join(self.model_path, "model.properties")
        if not os.path.exists(properties_file):
            self.logger.error(f"Model properties file not found: {properties_file}")
            return

        # Default sequence lengths for models (fallback)
        default_seq_lengths = {
            'lstm': 15,
            'transformer': 15,
            'ffnn': 10
        }

        try:
            with open(properties_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('#') or not line:
                        continue
                    if '=' in line:
                        key, value = line.split('=', 1)
                        try:
                            symbol, model_name, prop = key.split('.')
                        except ValueError:
                            self.logger.warning(f"Skipping invalid line in model.properties: {line}")
                            continue

                        if prop == 'path':
                            model_path = value
                            if not os.path.exists(model_path):
                                self.logger.warning(f"Model file not found for {symbol}.{model_name}: {model_path}")
                                continue
                            try:
                                session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
                                # Extract seq_length from model input shape
                                input_info = session.get_inputs()[0]
                                input_shape = input_info.shape
                                self.logger.debug(f"Input shape for {symbol}.{model_name}: {input_shape}")
                                if len(input_shape) == 3 and isinstance(input_shape[1], int):
                                    seq_length = input_shape[1]
                                else:
                                    seq_length = default_seq_lengths.get(model_name, 15)
                                    self.logger.warning(
                                        f"Could not determine seq_length for {symbol}.{model_name} (shape: {input_shape}). "
                                        f"Using default seq_length={seq_length}"
                                    )
                                if symbol not in self.models:
                                    self.models[symbol] = {}
                                self.models[symbol][model_name] = {
                                    'session': session,
                                    'scaler_X': MinMaxScaler(),
                                    'scaler_y': MinMaxScaler(),
                                    'seq_length': seq_length
                                }
                                if symbol not in self.active_tokens:
                                    self.active_tokens.append(symbol)
                                self.logger.info(f"Loaded ONNX model for {symbol} ({model_name}) from {model_path} with seq_length={seq_length}")
                            except Exception as e:
                                self.logger.error(f"Error loading ONNX model for {symbol}.{model_name}: {str(e)}")
                                continue
        except Exception as e:
            self.logger.error(f"Error loading models from {properties_file}: {str(e)}")

    def get_token_unit_ids(self) -> Dict[str, str]:
        """Get mapping of token symbols to unit IDs."""
        token_map = {}
        tokens = self.client.get_top_tokens_by_volume_df()
        for row in tokens.itertuples():
            symbol = row.symbol
            if symbol in self.active_tokens:
                token_map[symbol] = row.unit_id
        self.logger.debug(f"Token mapping: {token_map}")
        return token_map

    def fetch_current_prices(self) -> Dict[str, float]:
        """Fetch current prices for active tokens."""
        token_map = self.get_token_unit_ids()
        unit_ids = [token_map[symbol] for symbol in self.active_tokens if symbol in token_map]
        self.logger.debug(f"Fetching prices for unit IDs: {unit_ids}")
        if not unit_ids:
            self.logger.warning("No valid unit IDs found for active tokens")
            return {}
        try:
            price_data = self.client.get_latest_price(unit_ids)
            current_prices = {}
            if not price_data or 'assets' not in price_data:
                self.logger.warning("No assets data in price response")
                return {}
            latest_prices = {}
            for asset in price_data['assets']:
                unit_id = asset['asset']
                price = float(asset.get('last_price', 0))
                update_time = asset.get('last_update', 0)
                if unit_id not in latest_prices or update_time > latest_prices[unit_id]['last_update']:
                    latest_prices[unit_id] = {'price': price, 'last_update': update_time, 'provider': asset.get('provider', 'Unknown')}
            for symbol, unit_id in token_map.items():
                if unit_id in latest_prices:
                    current_prices[symbol] = latest_prices[unit_id]['price']
                    self.logger.info(f"Selected price for {symbol}: {current_prices[symbol]}")
                else:
                    current_prices[symbol] = 0.0
                    self.logger.warning(f"No price data found for {symbol} ({unit_id})")
            return current_prices
        except Exception as e:
            self.logger.error(f"Error fetching current prices: {e}")
            return {}

    def get_recent_features(self, symbol: str, unit_id: str) -> Optional[pd.DataFrame]:
        """Fetch recent data for prediction."""
        try:
            df = self.client.get_historical_data_df(unit_id, days=100, resolution="1D")
            if df.empty or len(df) < 30:
                self.logger.warning(f"Insufficient data for {symbol}: {len(df)} rows")
                return None
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_cols):
                self.logger.error(f"Missing required columns in data for {symbol}: {list(df.columns)}")
                return None
            result = df[required_cols].dropna()
            self.logger.debug(f"Data for {symbol}: type={type(result)}, columns={list(result.columns)}, rows={len(result)}")
            return result
        except Exception as e:
            self.logger.error(f"Error getting features for {symbol}: {e}")
            return None

    def make_predictions(self, current_prices: Dict[str, float]) -> List[PredictionRecord]:
        """Generate predictions and trading signals."""
        predictions = []
        token_map = self.get_token_unit_ids()
        for symbol in self.active_tokens:
            if symbol not in self.last_signals:
                self.last_signals[symbol] = {}
            if symbol not in current_prices or symbol not in token_map:
                self.logger.warning(f"Skipping {symbol}: No current price or unit ID")
                continue
            current_price = current_prices[symbol]
            unit_id = token_map[symbol]
            raw_df = self.get_recent_features(symbol, unit_id)
            if raw_df is None:
                self.logger.warning(f"Skipping {symbol}: No valid features")
                continue
            for model_name in self.models[symbol].keys():
                try:
                    model_info = self.models[symbol][model_name]
                    session = model_info['session']
                    scaler_X = model_info['scaler_X']
                    scaler_y = model_info['scaler_y']
                    seq_length = model_info['seq_length']
                    input_name = session.get_inputs()[0].name
                    if len(raw_df) < seq_length:
                        self.logger.warning(f"Insufficient data for {symbol} ({model_name}): {len(raw_df)} rows")
                        continue
                    # Prepare input data
                    X = raw_df[['open', 'high', 'low', 'close', 'volume']].values
                    X_seq = np.array([X[i:i + seq_length] for i in range(len(X) - seq_length + 1)])
                    if X_seq.shape[0] == 0:
                        self.logger.warning(f"No valid sequences for {symbol} ({model_name})")
                        continue
                    self.logger.debug(f"X_seq shape for {symbol} ({model_name}): {X_seq.shape}")
                    # Fit scalers on the fly
                    scaler_X.fit(X.reshape(-1, X.shape[-1]))
                    scaler_y.fit(raw_df['close'].values.reshape(-1, 1))
                    X_scaled = scaler_X.transform(X_seq.reshape(-1, X_seq.shape[-1])).reshape(X_seq.shape)
                    input_data = X_scaled[-1:].astype(np.float32)  # Use the latest sequence
                    self.logger.debug(f"Input data shape for {symbol} ({model_name}): {input_data.shape}")
                    self.logger.debug(f"Scaled input data for {symbol} ({model_name}): {input_data[-1]}")
                    # Run model
                    pred_scaled = session.run(None, {input_name: input_data})[0].flatten()[0]
                    self.logger.debug(f"Raw scaled prediction for {symbol} ({model_name}): {pred_scaled}")
                    predicted_price = scaler_y.inverse_transform(np.array([[pred_scaled]]))[0][0]
                    self.logger.debug(f"Unclamped predicted price for {symbol} ({model_name}): {predicted_price}")
                    # Clamping disabled
                    # predicted_price = min(max(predicted_price, current_price * 0.8), current_price * 1.2)
                    signal = self.generate_trading_signal(current_price, predicted_price)
                    last_signal_info = self.last_signals[symbol].get(model_name, {'signal': None, 'price': None, 'timestamp': None})
                    price_change = (abs(current_price - last_signal_info['price']) / last_signal_info['price'] if last_signal_info['price'] else float('inf'))
                    if signal != last_signal_info['signal'] or price_change >= self.signal_price_threshold:
                        confidence = self.calculate_confidence(symbol, model_name)
                        prediction = PredictionRecord(
                            timestamp=datetime.now().isoformat(),
                            symbol=symbol,
                            model_name=model_name,
                            current_price=current_price,
                            predicted_price=predicted_price,
                            signal=signal,
                            confidence=confidence
                        )
                        predictions.append(prediction)
                        self.last_signals[symbol][model_name] = {
                            'signal': signal,
                            'price': current_price,
                            'timestamp': prediction.timestamp
                        }
                        if signal != 'NA':
                            self.signal_logger.info(f"{signal} {symbol} - Current: {current_price:.8f}, Predicted: {predicted_price:.8f}, Model: {model_name}, Confidence: {confidence:.2f}")
                        self.performance_metrics[symbol][model_name].total_predictions += 1
                        self.performance_metrics[symbol][model_name].signals_generated[signal] += 1
                except Exception as e:
                    self.logger.error(f"Error making prediction for {symbol} with {model_name}: {str(e)}")
                    continue
        return predictions

    def generate_trading_signal(self, current_price: float, predicted_price: float, threshold: float = 0.005) -> str:
        """Generate trading signal based on price prediction."""
        if predicted_price is None or predicted_price <= 0:
            return 'NA'
        price_change = (predicted_price - current_price) / current_price
        if price_change > threshold:
            return 'BUY'
        elif price_change < -threshold:
            return 'SELL'
        return 'NA'

    def calculate_confidence(self, symbol: str, model_name: str) -> float:
        """Calculate prediction confidence."""
        if symbol not in self.performance_metrics:
            return 0.5
        metrics = self.performance_metrics[symbol][model_name]
        if metrics.total_predictions == 0:
            return 0.5
        base_confidence = metrics.accuracy
        sample_factor = min(metrics.total_predictions / 100, 1.0)
        confidence = base_confidence * sample_factor + 0.5 * (1 - sample_factor)
        return max(0.1, min(0.9, confidence))

    def update_prediction_accuracy(self, current_prices: Dict[str, float]):
        """Update prediction accuracy with actual prices."""
        updated_count = 0
        for prediction in self.prediction_history:
            if prediction.actual_price is None and prediction.symbol in current_prices and prediction.timestamp:
                pred_time = datetime.fromisoformat(prediction.timestamp)
                time_diff = datetime.now() - pred_time
                if time_diff.total_seconds() >= self.update_interval:
                    prediction.actual_price = current_prices[prediction.symbol]
                    if prediction.predicted_price > 0:
                        relative_error = abs(prediction.actual_price - prediction.predicted_price) / prediction.predicted_price
                        prediction.prediction_accuracy = max(0, 1 - relative_error)
                    else:
                        prediction.prediction_accuracy = 0
                    symbol = prediction.symbol
                    model_name = prediction.model_name
                    if symbol in self.performance_metrics and model_name in self.performance_metrics[symbol]:
                        metrics = self.performance_metrics[symbol][model_name]
                        total_with_accuracy = sum(1 for p in self.prediction_history
                                                  if p.symbol == symbol and p.model_name == model_name and p.prediction_accuracy is not None)
                        if total_with_accuracy > 0:
                            avg_accuracy = sum(p.prediction_accuracy for p in self.prediction_history
                                               if p.symbol == symbol and p.model_name == model_name and p.prediction_accuracy is not None) / total_with_accuracy
                            avg_error = sum(abs(p.actual_price - p.predicted_price) / p.predicted_price for p in self.prediction_history
                                            if p.symbol == symbol and p.model_name == model_name and p.actual_price is not None and p.predicted_price > 0) / total_with_accuracy
                            metrics.accuracy = avg_accuracy
                            metrics.avg_error = avg_error
                            metrics.last_updated = datetime.now().isoformat()
                    updated_count += 1
        if updated_count > 0:
            self.logger.info(f"Updated accuracy for {updated_count} predictions")

    def display_status(self, predictions: List[PredictionRecord], current_prices: Dict[str, float]):
        """Display current status and predictions."""
        print("=" * 100)
        print("CARDANO LIVE PRICE PREDICTION & TRADING SIGNALS")
        print("=" * 100)
        print(f"Last Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Active Tokens: {len(self.active_tokens)} | Update Interval: {self.update_interval}s")
        print(f"Active Tokens List: {self.active_tokens}")
        print("\nCURRENT PRICES:\n" + "-" * 50)
        for symbol, price in current_prices.items():
            print(f"{symbol:20s}: {price:.8f}")
        print("\nLATEST PREDICTIONS & SIGNALS:\n" + "-" * 90)
        print(f"{'Token':<15} {'Model':<15} {'Current':<12} {'Predicted':<12} {'Signal':<6} {'Conf':<6} {'Time':<20}")
        print("-" * 90)
        latest_predictions = {}
        for pred in sorted(self.prediction_history, key=lambda x: x.timestamp, reverse=True):
            key = (pred.symbol, pred.model_name)
            if key not in latest_predictions:
                latest_predictions[key] = pred
        for (symbol, model_name), pred in sorted(latest_predictions.items()):
            if symbol in self.active_tokens:
                time_str = datetime.fromisoformat(pred.timestamp).strftime('%Y-%m-%d %H:%M:%S')
                print(f"{pred.symbol:<15} {pred.model_name:<15} {pred.current_price:<12.8f} "
                      f"{pred.predicted_price:<12.8f} {pred.signal:<6} {pred.confidence:<6.2%} {time_str:<20}")
        print("\nPERFORMANCE SUMMARY:\n" + "-" * 70)
        print(f"{'Token':<15} {'Model':<15} {'Predictions':<12} {'Accuracy':<10} {'Signals':<10}")
        print("-" * 70)
        for symbol in self.active_tokens:
            if symbol in self.performance_metrics:
                for model_name, metrics in self.performance_metrics[symbol].items():
                    total_signals = sum(metrics.signals_generated.values())
                    print(f"{symbol:<15} {model_name:<15} {metrics.total_predictions:<12} "
                          f"{metrics.accuracy:<10.2%} {total_signals:<10}")
        print(f"\nTotal Predictions in History: {len(self.prediction_history)}")
        active_signals = [p for p in list(self.prediction_history)[-20:] if p.signal != 'NA']
        if active_signals:
            print(f"\nRECENT SIGNAL ACTIVITY (Last {len(active_signals)}):\n" + "-" * 60)
            for signal in active_signals[-10:]:
                time_str = datetime.fromisoformat(signal.timestamp).strftime('%H:%M:%S')
                print(f"{time_str} - {signal.signal} {signal.symbol} ({signal.model_name}) @ {signal.current_price:.8f}")

    def run_prediction_cycle(self):
        """Run a single prediction cycle."""
        try:
            current_prices = self.fetch_current_prices()
            if not current_prices:
                self.logger.warning("No current prices available")
                return
            self.update_prediction_accuracy(current_prices)
            predictions = self.make_predictions(current_prices)
            self.prediction_history.extend(predictions)
            self.display_status(predictions, current_prices)
            signal_count = sum(1 for p in predictions if p.signal != 'NA')
            self.logger.info(f"Cycle completed: {len(predictions)} new predictions, {signal_count} new signals")
        except Exception as e:
            self.logger.error(f"Error in prediction cycle: {e}")

    def save_session_data(self):
        """Save session data to JSON with atomic writing."""
        try:
            self.logger.info("Saving session data...")
            session_data = {
                'prediction_history': [asdict(pred) for pred in self.prediction_history],
                'performance_metrics': {
                    symbol: {model: asdict(metrics) for model, metrics in models.items()}
                    for symbol, models in self.performance_metrics.items()
                },
                'last_saved': datetime.now().isoformat()
            }
            os.makedirs('session', exist_ok=True)
            temp_file = 'session/live_session_data_temp.json'
            final_file = 'session/live_session_data.json'
            with open(temp_file, 'w') as f:
                json.dump(session_data, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
            os.replace(temp_file, final_file)
            self.logger.info("Session data saved successfully")
        except Exception as e:
            self.logger.error(f"Error saving session data: {str(e)}")
            if os.path.exists(temp_file):
                os.remove(temp_file)  # Clean up temporary file on error

    def load_session_data(self):
        """Load session data from JSON."""
        try:
            with open('session/live_session_data.json', 'r') as f:
                session_data = json.load(f)
            for pred_dict in session_data.get('prediction_history', []):
                if pred_dict.get('signal') == 'HOLD':
                    pred_dict['signal'] = 'NA'
                self.prediction_history.append(PredictionRecord(**pred_dict))
            for symbol, models in session_data.get('performance_metrics', {}).items():
                if symbol not in self.performance_metrics:
                    self.performance_metrics[symbol] = {}
                for model_name, metrics_dict in models.items():
                    signals = metrics_dict.get('signals_generated', {})
                    if 'HOLD' in signals:
                        signals['NA'] = signals.get('NA', 0) + signals.pop('HOLD')
                    self.performance_metrics[symbol][model_name] = PerformanceMetrics(**metrics_dict)
            self.logger.info(f"Loaded {len(self.prediction_history)} predictions from previous session")
        except FileNotFoundError:
            self.logger.info("No previous session data found")
        except Exception as e:
            self.logger.error(f"Error loading session data: {e}")

    def start(self):
        """Start the live signalling application."""
        self.logger.info("Starting live signalling application...")
        self.load_session_data()
        self.running = True
        try:
            while self.running:
                start_time = time.time()
                self.run_prediction_cycle()
                if len(self.prediction_history) % 10 == 0 and len(self.prediction_history) > 0:
                    self.save_session_data()
                elapsed = time.time() - start_time
                sleep_time = max(0, self.update_interval - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
        except KeyboardInterrupt:
            self.logger.info("Received interrupt signal, stopping...")
            self.stop()
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
            self.stop()
        finally:
            self.stop()

    def stop(self):
        """Stop the live signalling application."""
        self.logger.info("Stopping live signalling application...")
        self.running = False
        self.save_session_data()
        self.generate_session_report()
        self.logger.info("Live signalling application stopped")

    def generate_session_report(self):
        """Generate a session report."""
        try:
            total_predictions = len(self.prediction_history)
            signals_generated = sum(1 for p in self.prediction_history if p.signal != 'NA')
            report = []
            report.append("=" * 60)
            report.append("LIVE STREAMING SESSION REPORT")
            report.append("=" * 60)
            report.append(f"Session End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report.append(f"Total Predictions Made: {total_predictions}")
            report.append(f"Trading Signals Generated: {signals_generated}")
            report.append("")
            for symbol in self.active_tokens:
                if symbol in self.performance_metrics:
                    report.append(f"{symbol}:")
                    for model_name, metrics in self.performance_metrics[symbol].items():
                        report.append(f"  {model_name}:")
                        report.append(f"    Predictions: {metrics.total_predictions}")
                        report.append(f"    Accuracy: {metrics.accuracy:.2%}")
                        report.append(f"    Avg Error: {metrics.avg_error:.4f}")
                        na_count = metrics.signals_generated.get('NA', 0) + metrics.signals_generated.get('HOLD', 0)
                        report.append(f"    Signals: BUY={metrics.signals_generated.get('BUY', 0)}, "
                                      f"SELL={metrics.signals_generated.get('SELL', 0)}, NA={na_count}")
            report_text = "\n".join(report)
            os.makedirs('session', exist_ok=True)
            with open('session/session_report.txt', 'w') as f:
                f.write(report_text)
            print("\n" + report_text)
        except Exception as e:
            self.logger.error(f"Error generating session report: {str(e)}")

def signal_handler(signum, frame):
    """Handle interrupt signals."""
    print("\nReceived interrupt signal. Stopping gracefully...")
    sys.exit(0)

def main():
    update_interval = float(os.getenv('UPDATE_INTERVAL', 20))
    model_path = os.getenv('MODEL_PATH', 'model_results')
    print(f"Starting Cardano Live Signalling App")
    print(f"Update Interval: {update_interval} seconds")
    print(f"Model Path: {model_path}")
    print("Press Ctrl+C to stop")
    print()
    app = LiveSignallingApp(update_interval=update_interval, model_path=model_path)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    app.start()

if __name__ == "__main__":
    main()