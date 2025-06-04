import os

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import onnxruntime as ort
import logging
from typing import Tuple, Dict, Any, Optional
import onnx

logger = logging.getLogger(__name__)

class LSTMPriceModel(nn.Module):
    def __init__(self, input_features: int = 5, num_tech_features: int = 7, hidden_size: int = 64,
                 num_layers: int = 2, dropout: float = 0.4, seq_length: int = 15):
        super(LSTMPriceModel, self).__init__()
        self.input_features = input_features
        self.num_tech_features = num_tech_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_length = seq_length

        # Normalization buffers
        self.register_buffer('input_center', torch.zeros(input_features))
        self.register_buffer('input_scale', torch.ones(input_features))
        self.register_buffer('tech_center', torch.zeros(num_tech_features))
        self.register_buffer('tech_scale', torch.ones(num_tech_features))
        self.register_buffer('price_center', torch.tensor(0.0))
        self.register_buffer('price_scale', torch.tensor(1.0))

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_features + num_tech_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # Fully connected layer
        self.fc = nn.Linear(hidden_size, 1)

        # Scalers
        self.scaler_X = RobustScaler()
        self.scaler_y = RobustScaler()

        # Training parameters
        self.learning_rate = 0.0001
        self.batch_size = 32
        self.epochs = 300
        self.patience = 20
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def compute_technical_features(self, x: torch.Tensor) -> torch.Tensor:
        """Compute technical indicators."""
        if x.dim() == 2:
            x = x.unsqueeze(0)
        elif x.dim() < 2 or x.dim() > 3:
            raise ValueError(f"Expected input with 2 or 3 dimensions, got shape {x.shape}")

        batch_size, seq_len, _ = x.shape
        close = x[:, :, 3]  # Close price
        close_norm = (close - self.price_center) / (self.price_scale + 1e-8)
        close_norm = torch.clamp(close_norm, 0.0, 1.0)

        # 1. Price Change
        price_change = close_norm[:, 1:] / (close_norm[:, :-1] + 1e-8) - 1
        price_change = torch.clamp(price_change, -2.0, 2.0)
        price_change = torch.cat([torch.zeros_like(close_norm[:, :1]), price_change], dim=1)

        # 2. EMA-10
        ema_10 = torch.zeros_like(close_norm)
        alpha = 2 / (10 + 1)
        for t in range(seq_len):
            ema_10[:, t] = close_norm[:, t] if t == 0 else alpha * close_norm[:, t] + (1 - alpha) * ema_10[:, t - 1]

        # 3. RSI-7
        delta = close_norm[:, 1:] - close_norm[:, :-1]
        delta = torch.cat([torch.zeros_like(close_norm[:, :1]), delta], dim=1)
        gain = torch.clamp(delta, min=0.0)
        loss = -torch.clamp(delta, max=0.0)
        avg_gain = torch.zeros_like(gain)
        avg_loss = torch.zeros_like(loss)
        for t in range(7, seq_len):
            avg_gain[:, t] = gain[:, t - 7:t].mean(dim=1)
            avg_loss[:, t] = loss[:, t - 7:t].mean(dim=1)
        rs = avg_gain / (avg_loss + 1e-8)
        rs = torch.clamp(rs, 0.0, 100.0)
        rsi_7 = 100 - 100 / (1 + rs)
        rsi_7 = torch.where(torch.isnan(rsi_7) | torch.isinf(rsi_7), torch.tensor(0.5, device=x.device), rsi_7)

        # 4. SMA-20
        sma_20 = torch.zeros_like(close_norm)
        for t in range(20, seq_len):
            sma_20[:, t] = close_norm[:, t - 20:t].mean(dim=1)

        # 5. MACD (12, 26, 9)
        ema_12 = torch.zeros_like(close_norm)
        ema_26 = torch.zeros_like(close_norm)
        alpha_12 = 2 / (12 + 1)
        alpha_26 = 2 / (26 + 1)
        for t in range(seq_len):
            ema_12[:, t] = close_norm[:, t] if t == 0 else alpha_12 * close_norm[:, t] + (1 - alpha_12) * ema_12[:, t - 1]
            ema_26[:, t] = close_norm[:, t] if t == 0 else alpha_26 * close_norm[:, t] + (1 - alpha_26) * ema_26[:, t - 1]
        macd = ema_12 - ema_26
        signal_line = torch.zeros_like(macd)
        alpha_9 = 2 / (9 + 1)
        for t in range(seq_len):
            signal_line[:, t] = macd[:, t] if t == 0 else alpha_9 * macd[:, t] + (1 - alpha_9) * signal_line[:, t - 1]
        macd_hist = macd - signal_line

        # 6. Volatility (20-period std)
        vol_20 = torch.zeros_like(close_norm)
        for t in range(20, seq_len):
            vol_20[:, t] = close_norm[:, t - 20:t].std(dim=1)

        # 7. Bollinger Bands Width
        bb_upper = sma_20 + 2 * vol_20
        bb_lower = sma_20 - 2 * vol_20
        bb_width = (bb_upper - bb_lower) / (sma_20 + 1e-8)
        bb_width = torch.where(torch.isnan(bb_width) | torch.isinf(bb_width), torch.tensor(0.0, device=x.device), bb_width)

        # Stack features
        features = torch.stack([price_change, ema_10, rsi_7, sma_20, macd_hist, vol_20, bb_width], dim=2)
        features = torch.where(torch.isnan(features) | torch.isinf(features), torch.tensor(0.0, device=x.device), features)
        return features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        if x.dim() == 2:
            x = x.unsqueeze(0)
        elif x.dim() < 2 or x.dim() > 3:
            raise ValueError(f"Expected input with 2 or 3 dimensions, got shape {x.shape}")

        x_norm = (x - self.input_center.to(self.device)) / (self.input_scale.to(self.device) + 1e-8)
        x_norm = torch.clamp(x_norm, 0.0, 1.0)

        tech_features = self.compute_technical_features(x)
        tech_norm = (tech_features - self.tech_center.to(self.device)) / (self.tech_scale.to(self.device) + 1e-8)
        tech_norm = torch.clamp(tech_norm, 0.0, 1.0)

        combined_features = torch.cat([x_norm, tech_norm], dim=2)
        lstm_out, _ = self.lstm(combined_features)
        lstm_out = lstm_out[:, -1, :]
        output = self.fc(lstm_out)
        output = torch.sigmoid(output)
        output = output * self.price_scale.to(self.device) + self.price_center.to(self.device)
        return output.squeeze(-1)

    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare time series data for training."""
        logger.debug(f"Preparing data with shape: {df.shape}")
        if df.empty or len(df) < self.seq_length + 1:
            logger.error(f"Input DataFrame is empty or too short: {len(df)} rows, required: {self.seq_length + 1}")
            return np.array([]), np.array([])

        df = df.copy()
        df.columns = [col.lower() for col in df.columns]
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"Missing columns: {missing_cols}")
            return np.array([]), np.array([])

        df[required_cols] = df[required_cols].interpolate(method='linear').ffill().bfill()
        remaining_nans = df[required_cols].isna().sum()
        if remaining_nans.any():
            logger.error(f"Could not fill NaNs: {remaining_nans.to_dict()}")
            return np.array([]), np.array([])

        target = df['close'].shift(-1)
        valid_idx = target.notna()
        df = df.loc[valid_idx]
        target = target.loc[valid_idx]

        if len(df) < self.seq_length + 1:
            logger.error(f"Insufficient data after target creation: {len(df)} rows")
            return np.array([]), np.array([])

        X = df[required_cols].values
        y = target.values.reshape(-1, 1)

        X_seq, y_seq = [], []
        for i in range(len(X) - self.seq_length):
            X_seq.append(X[i:i + self.seq_length])
            y_seq.append(y[i + self.seq_length])

        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)

        if X_seq.shape[0] == 0:
            logger.error(f"No valid sequences created: len(X)={len(X)}, seq_length={self.seq_length}")
            return np.array([]), np.array([])

        logger.debug(f"Prepared data: X shape={X_seq.shape}, y shape={y_seq.shape}")
        return X_seq, y_seq

    def train_model(self, X: np.ndarray, y: np.ndarray, X_val: Optional[np.ndarray] = None,
                    y_val: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Train the model with time series cross-validation, keeping best model in memory."""
        logger.debug(f"Entering train_model with X shape={X.shape}, y shape={y.shape}")
        if X.shape[0] == 0 or y.shape[0] == 0:
            logger.error("Empty input or target data")
            return {}

        self.scaler_X.fit(X.reshape(-1, X.shape[-1]))
        self.scaler_y.fit(y)
        X_scaled = self.scaler_X.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
        y_scaled = self.scaler_y.transform(y)

        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler_X.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
            y_val_scaled = self.scaler_y.transform(y_val)
        else:
            X_val_scaled, y_val_scaled = None, None

        self.input_center.data = torch.tensor(self.scaler_X.center_, dtype=torch.float32)
        self.input_scale.data = torch.tensor(self.scaler_X.scale_, dtype=torch.float32)
        self.price_center.data = torch.tensor(self.scaler_y.center_, dtype=torch.float32)
        self.price_scale.data = torch.tensor(self.scaler_y.scale_, dtype=torch.float32)

        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            tech_features = self.compute_technical_features(X_tensor)
            self.tech_center.data = tech_features.median(dim=1)[0].median(dim=0)[0]
            self.tech_scale.data = (tech_features.quantile(0.75, dim=1) - tech_features.quantile(0.25, dim=1)).median(dim=0)[0] + 1e-8

        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
        criterion = nn.HuberLoss()
        train_losses, val_losses = [], []

        best_loss = float('inf')
        best_state_dict = None
        patience_counter = 0

        for epoch in range(self.epochs):
            self.train()
            epoch_loss = 0
            indices = np.random.permutation(len(X_scaled))
            for i in range(0, len(X_scaled), self.batch_size):
                batch_indices = indices[i:i + self.batch_size]
                batch_X = torch.tensor(X_scaled[batch_indices], dtype=torch.float32).to(self.device)
                batch_y = torch.tensor(y_scaled[batch_indices], dtype=torch.float32).to(self.device)
                optimizer.zero_grad()
                output = self(batch_X)
                loss = criterion(output, batch_y.squeeze(-1))
                if not torch.isnan(loss):
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                    optimizer.step()
                    epoch_loss += loss.item()
            avg_train_loss = epoch_loss / (len(X_scaled) // self.batch_size + 1)
            train_losses.append(avg_train_loss)
            scheduler.step()

            if X_val_scaled is not None:
                self.eval()
                with torch.no_grad():
                    val_output = self(torch.tensor(X_val_scaled, dtype=torch.float32).to(self.device))
                    val_loss = criterion(val_output, torch.tensor(y_val_scaled, dtype=torch.float32).to(self.device).squeeze(-1))
                    val_losses.append(val_loss.item())
                    if val_loss < best_loss:
                        best_loss = val_loss
                        best_state_dict = {k: v.cpu() for k, v in self.state_dict().items()}
                        patience_counter = 0
                    else:
                        patience_counter += 1
                    if patience_counter >= self.patience:
                        logger.info(f"Early stopping at epoch {epoch}")
                        break
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Train Loss: {avg_train_loss:.6f}, Val Loss: {val_losses[-1] if val_losses else 'N/A'}")

        if best_state_dict is not None:
            self.load_state_dict(best_state_dict)
        logger.debug(f"Training completed: {len(train_losses)} epochs")
        return {'train_losses': train_losses, 'val_losses': val_losses}

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on input data."""
        logger.debug(f"Predicting with X shape={X.shape}")
        if X.shape[0] == 0:
            logger.error("Empty input data for prediction")
            return np.array([])

        X_scaled = self.scaler_X.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
        self.eval()
        with torch.no_grad():
            pred_scaled = self(X_tensor).cpu().numpy()
        pred = self.scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
        return pred

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance."""
        logger.debug(f"Evaluating with X shape={X.shape}, y shape={y.shape}")
        if X.shape[0] == 0 or y.shape[0] == 0:
            logger.error("Empty input or target data for evaluation")
            return {'mse': np.nan, 'mae': np.nan, 'r2': np.nan}

        y_pred = self.predict(X)
        mse = float(mean_squared_error(y, y_pred))
        mae = float(mean_absolute_error(y, y_pred))
        r2 = float(r2_score(y, y_pred))
        logger.debug(f"Evaluation metrics: MSE={mse:.6f}, MAE={mae:.6f}, R²={r2:.6f}")
        return {'mse': mse, 'mae': mae, 'r2': r2}

    def save_onnx(self, save_path: str, input_size: Tuple[int, int, int] = None):
        """Save model in ONNX format with fixed sequence length."""
        logger.debug(f"Saving ONNX model to {save_path}")
        self.eval()
        if input_size is None:
            input_size = (1, self.seq_length, self.input_features)
        dummy_input = torch.randn(input_size, device=self.device)
        try:
            torch.onnx.export(
                self,
                dummy_input,
                save_path,
                export_params=True,
                opset_version=12,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )
            onnx_model = onnx.load(save_path)
            onnx.checker.check_model(onnx_model)
            logger.info(f"ONNX model saved and verified at {save_path}")
        except Exception as e:
            logger.error(f"Error saving ONNX model: {str(e)}")

    def predict_onnx(self, X: np.ndarray, model_path: str) -> np.ndarray:
        """Make predictions using ONNX model."""
        logger.debug(f"Predicting with ONNX model at {model_path}, X shape={X.shape}")
        if not os.path.exists(model_path):
            logger.error(f"ONNX model not found at {model_path}")
            return np.array([])

        X_scaled = self.scaler_X.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
        session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        input_name = session.get_inputs()[0].name
        pred_scaled = session.run(None, {input_name: X_scaled.astype(np.float32)})[0]
        pred = self.scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
        logger.debug(f"ONNX prediction completed: {len(pred)} values")
        return pred

    @staticmethod
    def get_model_info() -> Dict[str, Any]:
        """Return model information."""
        return {
            'name': 'lstm',
            'type': 'recurrent',
            'description': 'LSTM recurrent neural network with technical indicators for price prediction',
            'features': ['open', 'high', 'low', 'close', 'volume', 'price_change', 'ema_10', 'rsi_7', 'sma_20', 'macd_hist', 'vol_20', 'bb_width'],
            'hyperparameters': {
                'input_features': 5,
                'num_tech_features': 7,
                'hidden_size': 64,
                'num_layers': 2,
                'dropout': 0.4,
                'seq_length': 15,
                'learning_rate': 0.0001,
                'batch_size': 32,
                'epochs': 300,
                'patience': 20
            }
        }