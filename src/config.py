from dataclasses import dataclass
from pathlib import Path

@dataclass
class DataConfig:
    base_path: Path = Path("data/cards")


    test_dir : str = "test"
    train_dir: str = "train"
    valid_dir: str = "valid"

    img_size: int = 224
    batch_size: int = 32
    seed : int = 42

    def train_path(self) -> Path:
        return self.base_path / self.train_dir
    def valid_path(self) -> Path:
        return self.base_path / self.valid_dir
    def test_path(self) -> Path:
        return self.base_path / self.test_dir


@dataclass
class TrainConfig:
    backbone: str = "resnet18"
    epochs : int = 12
    lr: float = 3e-3
    freeze_epochs: int = 1
    model_name: str = "card_classifier"


