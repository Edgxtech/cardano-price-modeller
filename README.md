# Cardano Price Modeller

## Introduction

**This is a Work-In-Progress (WIP)**

Train price predictor models in the popular and portable ONNX format using integrated historical price and volume data.
Back test models to glean performance.
Conduct live signalling and simulated portfolio trading using generated models.
No claim is made about performance of any models generated or the examples provided.

## Setup

### Installation

```bash
git clone https://github.com/Edgxtech/cardano-price-modeller.git
cd cardano-price-modeller
```

### Configure Python environment
1. Install Conda [https://www.anaconda.com/docs/getting-started/miniconda/install](https://www.anaconda.com/docs/getting-started/miniconda/install)
2. `conda create -n cntmodeller`
3. `conda activate cntmodeller`
3. `pip install -r requirements.txt`

### Configure access to price data
1. Delegate a Cardano wallet to [AUSST (https://ausstaker.com.au)](https://ausstaker.com.au) Cardano Stakepool to get access to a price data API Token
2. Go to https://realfi.info, login as a web3 user with the delegating wallet to access the token
3. copy `.env.example` to `.env`
4. Add access token to `.env`,`REALFI_API_TOKEN=` e.g. `ey........`

## Running

### Running the model generator

```bash
python model_generator.py
```

This will produce `.onnx` files in `model_results/` and a `model.properties` file which can be curated to choose models 
to retain for live use. E.g.

```properties
# Model manifest for recommended models
# Generated: 2025-06-04 14:40:54
# Format: symbol.model_name.key=value
# Performance metrics included as comments

# SNEK - ffnn
# Test RÂ²: -4.1718, Test MAE: 0.001128, Backtest RÂ²: 0.9580, Backtest MAE: 0.000281, Total Return: 0.324266, Sharpe Ratio: 0.4936
SNEK.ffnn.path=model_results/cardano_models_SNEK_ffnn.onnx

# SNEK - lstm
# Test RÂ²: -2.2189, Test MAE: 0.000927, Backtest RÂ²: 0.9594, Backtest MAE: 0.000267, Total Return: 0.142957, Sharpe Ratio: 0.4217
SNEK.lstm.path=model_results/cardano_models_SNEK_lstm.onnx
```

### Running the live signaller app

```bash
python live_signalling_app.py
```

This will use `model.properties` from the model generation step to load models and produce trading signals 
from monitoring live data. It will also track the accuracy of signals over time.

To reset the session, delete all contents in `session/`

## Running the live simulated portfolio trader

```bash
python live_simtrader_app.py
```

Similar to running the live signaller however will conduct trades and manage a portfolio from a simulated initial capital amount.

To reset the session, delete all contents in `session/`