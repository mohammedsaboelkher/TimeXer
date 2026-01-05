# Overview

This repository is an implementation of **TimeXer**, a transformer-based architecture for multivariate time series forecasting that explicitly incorporates exogenous variables into the prediction process.

Access the paper here: https://arxiv.org/abs/2402.19072
Access to the original github code: https://github.com/thuml/TimeXer

## Why I wrote this implementation

The original TimeXer implementation, while pioneering, was **a bit complicated to follow** and **difficult to customize** for specific use cases. Given that TimeXer highlights a **very important idea**—forecasting with exogenous variables—I decided to create a cleaner, more accessible version of the codebase.

This implementation prioritizes:
- **Simplicity**: Clean, well-organized code that's easy to understand and follow
- **Customizability**: Modular design that allows you to easily adapt components for your needs
- **Configurability**: Flexible configuration system powered by Hydra for rapid experimentation
- **Research Evolution**: Enable researchers to quickly iterate on ideas and adapt the model without wrestling with complex implementations

The goal is to lower the barrier to entry for researchers interested in exploring time series forecasting with exogenous variables, allowing the field to evolve more rapidly.

---

## Implementation Note

I've made my best effort to faithfully implement the TimeXer architecture as described in the paper. However, if you encounter any issues, discrepancies, or have suggestions for improvements, please don't hesitate to submit an issue. Your feedback is valuable and helps improve this implementation for the entire community.

---

## Why Exogenous Variables Matter

Forecasting with exogenous variables is a crucial and practical problem in real-world applications. As highlighted in the TimeXer research, exogenous variables—such as weather conditions, market indicators, promotional campaigns, or macroeconomic factors—are often **readily available** and can **significantly influence** the time series we aim to predict.

For example:
- **Energy demand forecasting** benefits from weather data (temperature, humidity, wind speed)
- **Stock price prediction** can leverage economic indicators and market sentiment
- **Sales forecasting** improves with promotional calendars and seasonal factors
- **Traffic prediction** uses weather, events, and historical patterns

By explicitly modeling these exogenous variables within the forecasting architecture, TimeXer captures not just temporal patterns but also the causal relationships between external factors and the target series, leading to more accurate and interpretable predictions.

---

## Project Structure

```
TimeXer/
├── train.py                 # Main training script
├── requirements.txt         # Project dependencies
├── README.md                # This file
│
├── conf/                  # Hydra configuration files
│   ├── train.yaml         # Training configuration
│   ├── callbacks/         # Callback configurations (early stopping, checkpoints, etc.)
│   ├── data/              # Data loading configuration
│   ├── experiment/        # Experiment-specific settings
│   └── trainer/           # PyTorch Lightning trainer configuration
│
├── model/                 # Model architecture
│   ├── architecture.py    # Core TimeXer model
│   └── modules/           # Individual components
│       ├── attention.py
│       ├── encoder.py
│       ├── encoder_block.py
│       ├── instance_norm.py
│       ├── patch_embedding.py
│       ├── positional_encoding.py
│       └── variate_embedding.py
│
├── src/                   # Source code
│   ├── experiment.py      # PyTorch Lightning experiment wrapper
│   └── data/
│       ├── dataset.py     # Dataset class
│       ├── factory.py     # Data factory for loading datasets
│       └── module.py      # PyTorch Lightning data module
│
└── util/                  # Utility functions
    └── time.py            # Time-related utilities
```

---

## Getting Started

### Installation

1. Clone this repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Training

Run the training script with:
```bash
python train.py
```

Configuration is managed through **Hydra**, a powerful framework for configuring Python applications. The base configuration files are located in `conf/` and are organized by component:
- `data/default.yaml` - Data loading settings
- `experiment/default.yaml` - Model and training hyperparameters
- `callbacks/default.yaml` - Callback configurations
- `trainer/default.yaml` - PyTorch Lightning trainer settings

If you're not familiar with hydra, It's easy to learn! checkout hydra.cc

#### Customizing Configuration

The recommended approach for customization is:

1. **Create a new configuration file** in the desired directory (e.g., `conf/experiment/`):
   ```bash
   cp conf/experiment/default.yaml conf/experiment/custom_exp.yaml
   ```

2. **Edit your custom configuration** with your desired settings

3. **Override in training** either by:
   - **Option A: Modifying `train.yaml`**
     ```yaml
     defaults:
       - data: default
       - experiment: custom_exp      # Changed from 'default'
       - callbacks: default
       - trainer: default
     ```
   
   - **Option B: Command line override**
     ```bash
     python train.py experiment=custom_exp
     ```


**Override specific parameters directly from command line:**
```bash
python train.py experiment.optimizer.lr=0.001 trainer.max_epochs=100
python train.py experiment.scheduler=null # to not use learning rate scheduling
python train.py callbacks.learning_rate_monitor=null # to not use a particular callback
```

## Example: Training on the NP benchmark

### Step 1: Prepare the Data

Download `NP.csv` from the [original TimeXer repository](https://github.com/thuml/TimeXer) and save it to:
```
datasets/NP.csv
```

### Step 2: Create Data Configuration

Create `conf/data/NP.yaml`:

```yaml
_target_: src.data.factory.load

data_path: datasets/NP.csv
timestamp: date
target: OT
seq: 168
overlap: 0
horizon: 24
freq: 5min
train_ratio: 0.8
test_ratio: 0.1
batch_size: 4
```

### Step 3: Create Experiment Configuration

Create `conf/experiment/NP.yaml`:

```yaml
_target_: src.Experiment

model:
  _target_: model.TimeXer
  n_encoder_blocks: 3
  patch_len: 24
  patch_overlap: 0
  d_model: 512
  n_heads: 8
  d_ff: 512
  pred_len: 24
  use_instance_norm: true
  dropout: 0.1
  bias: true

optimizer:
  _target_: torch.optim.Adam
  _partial_: true
  lr: 0.0001
  weight_decay: 0.0

scheduler:
  _target_: torch.optim.lr_scheduler.ReduceLROnPlateau
  _partial_: true
  mode: min
  factor: 0.5
  patience: 10
```

### Step 4: Train the Model

Run the training script with the NP configuration:

```bash
python train.py data=NP experiment=NP
```

You can further customize callbacks and trainer settings by creating additional configuration files in `conf/callbacks/` and `conf/trainer/` and overriding them similarly.

---

## Implementation Scope

This implementation focuses on **univariate time series forecasting**, where we predict a single target variable using its historical values and exogenous features. While the original TimeXer supports multivariate forecasting, this streamlined version prioritizes clarity and ease of experimentation for univariate scenarios.

Additionally, I've introduced **patch overlap** as a configurable parameter to enable researchers to easily experiment with different temporal receptive field configurations and explore novel architectural variations without modifying the core codebase.