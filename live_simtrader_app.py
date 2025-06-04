import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional
from dotenv import load_dotenv
from dataclasses import dataclass, asdict
from collections import deque
import signal
import sys

from live_signalling_app import LiveSignallingApp, PredictionRecord, PerformanceMetrics
from util.helpers import NumpyEncoder

load_dotenv()

@dataclass
class PortfolioHolding:
    symbol: str
    quantity: float
    average_buy_price: float
    total_cost: float

@dataclass
class TradeRecord:
    timestamp: str
    symbol: str
    model_name: str
    signal: str
    price: float
    quantity: float
    trade_value: float
    portfolio_value: float
    cash_balance: float

class LiveFakePortfolioApp(LiveSignallingApp):
    def __init__(self, initial_capital: float = 10000.0, update_interval: int = 60,
                 model_path: str = "model_results", trade_fee: float = 0.001):
        super().__init__(update_interval=update_interval, model_path=model_path)
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.trade_fee = trade_fee
        self.holdings = {}
        self.trade_history = deque(maxlen=1000)
        self.portfolio_value_history = deque(maxlen=1000)
        self.max_weight = 0.2
        self.trade_cooldown = 300
        self.last_trade_times = {}

        for token in self.active_tokens:
            self.holdings[token] = PortfolioHolding(symbol=token, quantity=0.0, average_buy_price=0.0, total_cost=0.0)
            self.last_trade_times[token] = {}

        self.setup_portfolio_logging()
        print(f"Initialized fake portfolio app with ${initial_capital:.2f} initial capital")
        print(f"Trade fee: {trade_fee * 100:.2f}%")
        self.logger.info(f"Portfolio initialized: Cash=${self.cash:.2f}, Tokens={self.active_tokens}")

    def setup_portfolio_logging(self):
        self.portfolio_logger = logging.getLogger('portfolio')
        self.portfolio_logger.setLevel(logging.INFO)
        self.portfolio_logger.propagate = False
        self.portfolio_logger.handlers.clear()

        file_handler = logging.FileHandler('logs/portfolio.log')
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.portfolio_logger.addHandler(file_handler)

    def execute_trade(self, symbol: str, signal: str, price: float, model_name: str, confidence: float) -> Optional[TradeRecord]:
        try:
            self.portfolio_logger.debug(f"Attempting {signal} trade for {symbol} with {model_name}, price={price:.8f}, confidence={confidence:.2f}")
            holding = self.holdings[symbol]
            portfolio_value = self.calculate_portfolio_value({symbol: price})
            current_weight = (holding.quantity * price) / portfolio_value if portfolio_value > 0 else 0.0
            self.portfolio_logger.debug(f"Portfolio value={portfolio_value:.2f}, current_weight={current_weight:.4f}, max_weight={self.max_weight:.4f}")

            # Check cooldown
            last_trade = self.last_trade_times.get(symbol, {}).get(model_name, {}).get(signal)
            if last_trade:
                last_time = datetime.fromisoformat(last_trade)
                time_diff = (datetime.now() - last_time).total_seconds()
                if time_diff < self.trade_cooldown:
                    self.portfolio_logger.debug(f"Skipping {signal} for {symbol} ({model_name}): Cooldown active ({time_diff:.1f}s < {self.trade_cooldown}s)")
                    return None
            else:
                self.portfolio_logger.debug(f"No previous {signal} trade for {symbol} ({model_name})")

            trade_quantity = 0.0
            trade_value = 0.0

            if signal == 'BUY' and self.cash > 0 and current_weight < self.max_weight:
                target_value = portfolio_value * self.max_weight
                available_value = target_value - (holding.quantity * price)
                allocation = min(self.cash * 0.1 * confidence, available_value, self.cash)
                trade_quantity = allocation / price * (1 - self.trade_fee)
                trade_value = trade_quantity * price
                if trade_value <= 0 or trade_quantity <= 0:
                    self.portfolio_logger.debug(f"Invalid BUY: trade_value={trade_value:.2f}, trade_quantity={trade_quantity:.8f}")
                    return None

                self.cash -= trade_value
                new_quantity = holding.quantity + trade_quantity
                new_total_cost = holding.total_cost + trade_value
                holding.quantity = new_quantity
                holding.total_cost = new_total_cost
                holding.average_buy_price = new_total_cost / new_quantity if new_quantity > 0 else 0.0

                self.portfolio_logger.info(f"BUY {symbol}: {trade_quantity:.8f} @ {price:.8f}, "
                                           f"Cost=${trade_value:.2f}, Cash=${self.cash:.2f}")

            elif signal == 'SELL' and holding.quantity > 0:
                trade_quantity = min(holding.quantity * 0.5 * confidence, holding.quantity)
                trade_value = trade_quantity * price * (1 - self.trade_fee)
                if trade_value <= 0 or trade_quantity <= 0:
                    self.portfolio_logger.debug(f"Invalid SELL: trade_value={trade_value:.2f}, trade_quantity={trade_quantity:.8f}")
                    return None

                self.cash += trade_value
                holding.quantity -= trade_quantity
                holding.total_cost = holding.quantity * holding.average_buy_price
                holding.average_buy_price = holding.average_buy_price if holding.quantity > 0 else 0.0

                self.portfolio_logger.info(f"SELL {symbol}: {trade_quantity:.8f} @ {price:.8f}, "
                                           f"Proceeds=${trade_value:.2f}, Cash=${self.cash:.2f}")

            else:
                self.portfolio_logger.debug(f"No action: signal={signal}, cash={self.cash:.2f}, holding.quantity={holding.quantity:.8f}")
                return None

            if symbol not in self.last_trade_times:
                self.last_trade_times[symbol] = {}
            if model_name not in self.last_trade_times[symbol]:
                self.last_trade_times[symbol][model_name] = {}
            self.last_trade_times[symbol][model_name][signal] = datetime.now().isoformat()

            portfolio_value = self.calculate_portfolio_value({symbol: price})
            trade = TradeRecord(
                timestamp=datetime.now().isoformat(),
                symbol=symbol,
                model_name=model_name,
                signal=signal,
                price=price,
                quantity=trade_quantity,
                trade_value=trade_value,
                portfolio_value=portfolio_value,
                cash_balance=self.cash
            )
            self.trade_history.append(trade)
            self.portfolio_value_history.append({
                'timestamp': trade.timestamp,
                'portfolio_value': portfolio_value
            })
            self.portfolio_logger.debug(f"Trade recorded: {signal} {trade_quantity:.8f} {symbol} @ {price:.8f}")
            return trade

        except Exception as e:
            self.logger.error(f"Error executing trade for {symbol} with {model_name}: {str(e)}", exc_info=True)
            return None

    def calculate_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        total_value = self.cash
        for symbol, holding in self.holdings.items():
            price = current_prices.get(symbol, 0.0)
            total_value += holding.quantity * price
        return max(total_value, 1.0)  # Avoid division by zero

    def make_predictions(self, current_prices: Dict[str, float]) -> List[PredictionRecord]:
        predictions = super().make_predictions(current_prices)
        trades_executed = []

        predictions_by_symbol = {}
        for pred in predictions:
            if pred.symbol not in predictions_by_symbol:
                predictions_by_symbol[pred.symbol] = []
            predictions_by_symbol[pred.symbol].append(pred)

        for symbol, symbol_preds in predictions_by_symbol.items():
            non_na_signals = [pred.signal for pred in symbol_preds if pred.signal != 'NA']
            if not non_na_signals:
                self.portfolio_logger.debug(f"No non-NA signals for {symbol}")
                continue

            if len(set(non_na_signals)) == 1:
                signal = non_na_signals[0]
                active_models = len(symbol_preds)
                agreeing_models = len([pred for pred in symbol_preds if pred.signal == signal])
                if agreeing_models == active_models or all(pred.signal in [signal, 'NA'] for pred in symbol_preds):
                    confidences = [pred.confidence for pred in symbol_preds if pred.signal == signal]
                    avg_confidence = sum(confidences) / len(confidences) if confidences else 1.0
                    price = current_prices.get(symbol, symbol_preds[0].current_price)
                    self.portfolio_logger.debug(f"Consensus for {symbol}: {signal} by {agreeing_models}/{active_models} models, confidence={avg_confidence:.2f}")
                    trade = self.execute_trade(
                        symbol=symbol,
                        signal=signal,
                        price=price,
                        model_name=symbol_preds[0].model_name,
                        confidence=avg_confidence
                    )
                    if trade:
                        trades_executed.append(trade)
                        self.portfolio_logger.info(
                            f"Consensus trade executed for {symbol}: {signal} with confidence {avg_confidence:.2f} "
                            f"across {agreeing_models}/{active_models} models"
                        )
                else:
                    self.portfolio_logger.debug(
                        f"No consensus for {symbol}: {agreeing_models}/{active_models} models agree on {signal}"
                    )
            else:
                self.portfolio_logger.debug(
                    f"Conflicting signals for {symbol}: {non_na_signals}"
                )

        return predictions

    def display_status(self, predictions: List[PredictionRecord], current_prices: Dict[str, float]):
        super().display_status(predictions, current_prices)

        print("\nPORTFOLIO STATUS:")
        print("-" * 70)
        print(f"{'Token':<15} {'Quantity':<12} {'Avg Buy Price':<15} {'Current Price':<15} {'Value':<12}")
        print("-" * 70)

        portfolio_value = self.calculate_portfolio_value(current_prices)
        for symbol, holding in self.holdings.items():
            price = current_prices.get(symbol, 0.0)
            value = holding.quantity * price
            print(f"{symbol:<15} {holding.quantity:<12.8f} {holding.average_buy_price:<15.8f} "
                  f"{price:<15.8f} {value:<12.2f}")

        print(f"\nCash: ${self.cash:.2f}")
        print(f"Total Portfolio Value: ${portfolio_value:.2f}")
        print(f"Returns: {(portfolio_value / self.initial_capital - 1) * 100:.2f}%")

        if self.trade_history:
            print(f"\nRECENT TRADES (Last {min(5, len(self.trade_history))}):")
            print("-" * 80)
            for trade in list(self.trade_history)[-5:]:
                time_str = datetime.fromisoformat(trade.timestamp).strftime('%H:%M:%S')
                print(f"{time_str} - {trade.signal} {trade.quantity:.8f} {trade.symbol} "
                      f"@ {trade.price:.8f} (${trade.trade_value:.2f})")

    def save_session_data(self):
        try:
            self.logger.info("Attempting to save session data...")
            session_data = {
                'prediction_history': [asdict(pred) for pred in self.prediction_history],
                'performance_metrics': {
                    symbol: {
                        model: asdict(metrics) for model, metrics in models.items()
                    } for symbol, models in self.performance_metrics.items()
                },
                'portfolio_state': {
                    'cash': self.cash,
                    'holdings': {symbol: asdict(holding) for symbol, holding in self.holdings.items()},
                    'trade_history': [asdict(trade) for trade in self.trade_history],
                    'portfolio_value_history': list(self.portfolio_value_history),
                    'initial_capital': self.initial_capital
                },
                'last_signals': self.last_signals,
                'signal_price_threshold': self.signal_price_threshold,
                'last_trade_times': self.last_trade_times,
                'max_weight': self.max_weight,
                'trade_cooldown': self.trade_cooldown,
                'last_saved': datetime.now().isoformat()
            }
            self.logger.info(f"Session data prepared: {len(session_data['prediction_history'])} predictions, "
                             f"{sum(len(models) for models in session_data['performance_metrics'].values())} models, "
                             f"{len(session_data['portfolio_state']['trade_history'])} trades")
            os.makedirs('session', exist_ok=True)
            temp_file = 'session/live_session_data_temp.json'
            final_file = 'session/live_session_data.json'
            with open(temp_file, 'w') as f:
                json.dump(session_data, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
            os.replace(temp_file, final_file)
            self.logger.info("Session data saved successfully to live_session_data.json")
        except Exception as e:
            self.logger.error(f"Error saving session data: {str(e)}", exc_info=True)
            if os.path.exists(temp_file):
                os.remove(temp_file)

    def load_session_data(self):
        try:
            with open('session/live_session_data.json', 'r') as f:
                session_data = json.load(f)

            for pred_dict in session_data.get('prediction_history', []):
                pred = PredictionRecord(**pred_dict)
                self.prediction_history.append(pred)

            for symbol, models in session_data.get('performance_metrics', {}).items():
                if symbol not in self.performance_metrics:
                    self.performance_metrics[symbol] = {}
                for model_name, metrics_dict in models.items():
                    self.performance_metrics[symbol][model_name] = PerformanceMetrics(**metrics_dict)

            portfolio_state = session_data.get('portfolio_state', {})
            self.cash = portfolio_state.get('cash', self.initial_capital)
            for symbol, holding_dict in portfolio_state.get('holdings', {}).items():
                if symbol in self.holdings:
                    self.holdings[symbol] = PortfolioHolding(**holding_dict)
            for trade_dict in portfolio_state.get('trade_history', []):
                self.trade_history.append(TradeRecord(**trade_dict))
            for value_entry in portfolio_state.get('portfolio_value_history', []):
                self.portfolio_value_history.append(value_entry)

            self.last_signals = session_data.get('last_signals', {})
            self.signal_price_threshold = session_data.get('signal_price_threshold', 0.01)
            self.last_trade_times = session_data.get('last_trade_times', {})
            self.max_weight = session_data.get('max_weight', 0.2)
            self.trade_cooldown = session_data.get('trade_cooldown', 300)

            self.logger.info(f"Loaded {len(self.prediction_history)} predictions, "
                             f"{len(self.trade_history)} trades from previous session")

        except FileNotFoundError:
            self.logger.info("No previous session data found")
        except Exception as e:
            self.logger.error(f"Error loading session data: {e}")

    def generate_session_report(self):
        try:
            total_predictions = len(self.prediction_history)
            signals_generated = sum(1 for p in self.prediction_history if p.signal != 'NA')
            total_trades = len(self.trade_history)
            portfolio_value = self.calculate_portfolio_value(self.fetch_current_prices())

            report = []
            report.append("=" * 60)
            report.append("LIVE PORTFOLIO SESSION REPORT")
            report.append("=" * 60)
            report.append(f"Session End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report.append(f"Initial Capital: ${self.initial_capital:.2f}")
            report.append(f"Final Portfolio Value: ${portfolio_value:.2f}")
            report.append(f"Returns: {(portfolio_value / self.initial_capital - 1) * 100:.2f}%")
            report.append(f"Total Predictions Made: {total_predictions}")
            report.append(f"Trading Signals Generated: {signals_generated}")
            report.append(f"Total Trades Executed: {total_trades}")
            report.append("")

            for symbol in self.active_tokens:
                if symbol in self.performance_metrics:
                    report.append(f"{symbol}:")
                    for model_name, metrics in self.performance_metrics[symbol].items():
                        report.append(f"  {model_name}:")
                        report.append(f"    Predictions: {metrics.total_predictions}")
                        report.append(f"    Accuracy: {metrics.accuracy:.2%}")
                        report.append(f"    Avg Error: {metrics.avg_error:.4f}")
                        report.append(f"    Signals: BUY={metrics.signals_generated['BUY']}, "
                                      f"SELL={metrics.signals_generated['SELL']}, "
                                      f"NA={metrics.signals_generated['NA']}")
                    holding = self.holdings.get(symbol, PortfolioHolding(symbol, 0.0, 0.0, 0.0))
                    report.append(f"  Holding: {holding.quantity:.8f} @ Avg ${holding.average_buy_price:.8f}")

            report_text = "\n".join(report)

            os.makedirs('session', exist_ok=True)
            with open('session/portfolio_report.txt', 'w') as f:
                f.write(report_text)

            print("\n" + report_text)

        except Exception as e:
            self.logger.error(f"Error generating session report: {e}")

def main():
    update_interval = int(os.getenv('UPDATE_INTERVAL', 60))
    model_path = os.getenv('MODEL_PATH', 'model_results')
    initial_capital = float(os.getenv('INITIAL_CAPITAL', 10000.0))

    print(f"Starting Cardano Live Portfolio App")
    print(f"Initial Capital: ${initial_capital:.2f}")
    print(f"Update Interval: {update_interval} seconds")
    print(f"Model Path: {model_path}")
    print("Press Ctrl+C to stop")
    print()

    app = LiveFakePortfolioApp(
        initial_capital=initial_capital,
        update_interval=update_interval,
        model_path=model_path
    )

    def signal_handler(signum, frame):
        print("\nReceived interrupt signal. Stopping gracefully...")
        app.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    app.start()

if __name__ == "__main__":
    main()