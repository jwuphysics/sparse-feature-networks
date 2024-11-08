from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class TrainingConfig:
    learning_rate: float = 0.1
    batch_size: int = 64
    epochs: int = 20
    device: str = "cuda"


@dataclass
class DataConfig:
    root: Path = Path("..").resolve()
    image_size: int = 160
    crop_size: int = 144
    train_split: float = 0.8


@dataclass
class ExperimentConfig:
    name: str
    target: str
    k: int
    model_dir: Path
    results_dir: Path
    data_config: DataConfig
    training_config: TrainingConfig
    seed: int = 256
