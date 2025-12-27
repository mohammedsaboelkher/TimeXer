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
class Config:
    data: DataConfig
    model: ModelConfig
    seq_len: int
    label_len: int
    pred_len: int

