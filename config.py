from dataclasses import dataclass

@dataclass
class DataConfig:
    data_path: str
    freq: str
    target_column: str
    timestamp_column: str
    scale: bool
    time_features: bool
    train_ratio: float
    test_ratio: float
    batch_size: int

@dataclass
class ModelConfig:
    n_encoder_blocks: int
    patch_len: int
    patch_overlap: int
    d_model: int
    n_heads: int
    d_ff: int
    glb_token: bool = True
    use_instance_norm: bool = True
    dropout: float = 0.1
    bias: bool = True

@dataclass
class TrainingConfig:
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 50
    patience: int = 5
    gradient_clip_val: float = 1.0
    accelerator: str = "auto"
    devices: int = 1

@dataclass
class Config:
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    seq_len: int
    label_len: int
    pred_len: int