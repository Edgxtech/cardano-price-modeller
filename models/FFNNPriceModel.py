import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, r2_score
import onnxruntime as ort
import logging
from typing import Tuple, Dict, Any, Optional

logger = logging.getLogger(__name__)

class FFNNPriceModel(nn.Module):
    def __init__(self, input_features: int = 5, num_tech_features: int = 4, seq_length: int = 10):
        super(FFNNPriceModel, self).__init__()
        self.input_features = input_features
        self.num_tech_features = num_tech_features
        self.seq_length = seq_length

        # Statistics for normalization
        self.register_buffer('input_center', torch.zeros(input_features))
        self.register_buffer('input_scale', torch.ones(input_features))
        self.register_buffer('tech_center', torch.zeros(num_tech_features))
        self.register_buffer('tech_scale', torch.ones(num_tech_features))
        self.register_buffer('price_center', torch.tensor(0.0))
        self.register_buffer('price_scale', torch.tensor(1.0))

        # Feedforward neural network
        self.network = nn.Sequential(
            nn.Linear(input_features + num_tech_features, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        # Scalers for external data
        self.scaler_X = RobustScaler()
        self.scaler_y = RobustScaler()

        # Training parameters
        self.learning_rate = 0.0001
        self.batch_size = 32
        self.epochs = 50
        self.patience = 10
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def compute_technical_features(self, x: torch.Tensor) -> torch.Tensor:
        """Compute technical indicators (price change, EMA-10, RSI-7, SMA-20)."""
        if x.dim() == 2:
            x = x.unsqueeze(0)
        elif x.dim() < 2 or x.dim() > 3:
            raise ValueError(f"Expected input with 2 or 3 dimensions, got shape {x.shape}")

        close = x[:, :, 3]  # Close price
        close_norm = (close - self.price_center) / (self.price_scale + 1e-8)
        close_norm = torch.clamp(close_norm, 0.0, 1.0)

        # 1. Price Change
        price_change = close_norm[:, 1:] / (close_norm[:, :-1] + 1e-8) - 1
        price_change = torch.clamp(price_change, min=-2.0, max=2.0)
        price_change = torch.cat([torch.zeros_like(close_norm[:, :1]), price_change], dim=1)

        # 2. EMA-10
        ema_10 = torch.zeros_like(close_norm)
        alpha = 2 / (10 + 1)
        for t in range(ema_10.shape[1]):
            if t == 0:
                ema_10[:, t] = close_norm[:, t]
            else:
                ema_10[:, t] = alpha * close_norm[:, t] + (1 - alpha) * ema_10[:, t - 1]

        # 3. RSI-7
        delta = close_norm[:, 1:] - close_norm[:, :-1]
        delta = torch.cat([torch.zeros_like(close_norm[:, :1]), delta], dim=1)
        gain = torch.clamp(delta, min=0.0)
        loss = -torch.clamp(delta, max=0.0)
        avg_gain = torch.zeros_like(gain)
        avg_loss = torch.zeros_like(loss)
        for t in range(7, gain.shape[1]):
            avg_gain[:, t] = gain[:, t - 7:t].mean(dim=1)
            avg_loss[:, t] = loss[:, t - 7:t].mean(dim=1)
        rs = avg_gain / (avg_loss + 1e-8)
        rs = torch.clamp(rs, min=0.0, max=100.0)
        rsi_7 = 100 - 100 / (1 + rs)
        rsi_7 = torch.where(torch.isnan(rsi_7) | torch.isinf(rsi_7), torch.tensor(50.0, device=x.device), rsi_7) / 100

        # 4. SMA-20
        sma_20 = torch.zeros_like(close_norm)
        for t in range(20, close_norm.shape[1]):
            sma_20[:, t] = close_norm[:, t - 20:t].mean(dim=1)

        features = torch.stack([price_change, ema_10, rsi_7, sma_20], dim=2)
        features = torch.where(torch.isnan(features) | torch.isinf(features), torch.tensor(0.0, device=x.device), features)
        return features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        if x.dim() == 2:
            x = x.unsqueeze(0)
        elif x.dim() < 2 or x.dim() > 3:
            raise ValueError(f"Expected input with 2 or 3 dimensions, got shape {x.shape}")

        # Normalize raw inputs
        x_norm = (x - self.input_center.to(x.device)) / (self.input_scale.to(x.device) + 1e-8)
        x_norm = torch.clamp(x_norm, 0.0, 1.0)

        # Compute and normalize technical features
        tech_features = self.compute_technical_features(x)
        tech_norm = (tech_features - self.tech_center.to(x.device)) / (self.tech_scale.to(x.device) + 1e-8)
        tech_norm = torch.clamp(tech_norm, 0.0, 1.0)

        # Combine features
        combined_features = torch.cat([x_norm, tech_norm], dim=2)

        # Predict normalized price for the last time step
        output = self.network(combined_features[:, -1, :])
        output = torch.sigmoid(output)

        # Denormalize to price range
        output = output * self.price_scale.to(x.device) + self.price_center.to(x.device)
        return output.squeeze(-1)

    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare time series data for training."""
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

        # Interpolate NaN values
        df[required_cols] = df[required_cols].interpolate(method='linear').ffill().bfill()
        remaining_nans = df[required_cols].isna().sum()
        if remaining_nans.any():
            logger.error(f"Could not fill NaNs: {remaining_nans.to_dict()}")
            return np.array([]), np.array([])

        # Create target
        target = df['close'].shift(-1)
        valid_idx = target.notna()
        df = df.loc[valid_idx]
        target = target.loc[valid_idx]

        if len(df) < self.seq_length + 1:
            logger.error(f"Insufficient data after target creation: {len(df)} rows")
            return np.array([]), np.array([])

        X = df[required_cols].values
        y = target.values.reshape(-1, 1)

        # Create sequences
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

    def train_model(self, X: np.ndarray, y: np.ndarray, X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Train the model with time series cross-validation, keeping best model in memory."""
        logger.debug(f"Entering train_model with X shape={X.shape}, y shape={y.shape}")
        if X.shape[0] == 0 or y.shape[0] == 0:
            logger.error("Empty input or target data")
            return {}

        # Fit scalers
        self.scaler_X.fit(X.reshape(-1, X.shape[-1]))
        self.scaler_y.fit(y)
        X_scaled = self.scaler_X.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
        y_scaled = self.scaler_y.transform(y)

        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler_X.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
            y_val_scaled = self.scaler_y.transform(y_val)
        else:
            X_val_scaled, y_val_scaled = None, None

        # Set normalization parameters
        self.input_center.data = torch.tensor(self.scaler_X.center_, dtype=torch.float32)
        self.input_scale.data = torch.tensor(self.scaler_X.scale_, dtype=torch.float32)
        self.price_center.data = torch.tensor(self.scaler_y.center_, dtype=torch.float32)
        self.price_scale.data = torch.tensor(self.scaler_y.scale_, dtype=torch.float32)

        # Compute technical feature normalization parameters
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            tech_features = self.compute_technical_features(X_tensor)
            self.tech_center.data = tech_features.median(dim=1)[0].median(dim=0)[0]
            self.tech_scale.data = (tech_features.quantile(0.75, dim=1) - tech_features.quantile(0.25, dim=1)).median(dim=0)[0] + 1e-8

        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
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
        if X.shape[0] == 0:
            logger.warning("Empty input for prediction")
            return np.array([])

        X_scaled = self.scaler_X.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
        self.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
            predictions = self(X_tensor).cpu().numpy()
        return self.scaler_y.inverse_transform(predictions.reshape(-1, 1)).flatten()

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance."""
        if X.shape[0] == 0 or y.shape[0] == 0:
            logger.warning("Empty input or target for evaluation")
            return {'mae': float('inf'), 'mse': float('inf'), 'r2': -float('inf')}

        predictions = self.predict(X)
        mse = np.mean((predictions - y.flatten()) ** 2)
        mae = mean_absolute_error(y, predictions)
        r2 = r2_score(y, predictions)
        return {'mse': mse, 'mae': mae, 'r2': r2}

    def save_onnx(self, filepath: str):
        """Save model in ONNX format."""
        self.eval()
        dummy_input = torch.randn(1, self.seq_length, self.input_features).to(self.device)
        torch.onnx.export(
            self,
            dummy_input,
            filepath,
            opset_version=12,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}}
        )
        logger.info(f"Model saved to {filepath}")

    def load_onnx(self, filepath: str):
        """Load model from ONNX file."""
        session = ort.InferenceSession(filepath, providers=['CPUExecutionProvider'])
        self.session = session
        logger.info(f"Loaded ONNX model from {filepath}")

    def predict_onnx(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using ONNX model."""
        if not hasattr(self, 'session'):
            raise ValueError("ONNX model must be loaded before prediction")
        X_scaled = self.scaler_X.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
        input_name = self.session.get_inputs()[0].name
        predictions = []
        for i in range(0, len(X_scaled), self.batch_size):
            batch = X_scaled[i:i + self.batch_size].astype(np.float32)
            pred = self.session.run(None, {input_name: batch})[0]
            predictions.extend(pred)
        return self.scaler_y.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()

    @staticmethod
    def get_model_info() -> Dict[str, Any]:
        """Return model information."""
        return {
            'name': 'ffnn',
            'type': 'feedforward',
            'description': 'Feedforward neural network with technical indicators for price prediction',
            'features': ['open', 'high', 'low', 'close', 'volume', 'price_change', 'ema_10', 'rsi_7', 'sma_20'],
            'hyperparameters': {
                'input_features': 5,
                'num_tech_features': 4,
                'seq_length': 10,
                'learning_rate': 0.0001,
                'batch_size': 32,
                'epochs': 50,
                'patience': 10
            }
        }

def create_model(**kwargs) -> 'FFNNPriceModel':
    """Factory function to create model instance."""
    return FFNNPriceModel(**kwargs)