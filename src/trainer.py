from fastai.vision.all import Learner
from .config import TrainConfig

class Trainer:
    def __init__(self, learner: Learner, config: TrainConfig):
        self.learner = learner
        self.config = config

    def train(self):
        self.learner.fine_tune(self.config.epochs)

    def save_model(self):
        self.learner.export(f"{self.config.model_name}.pkl")
        print(f"Model saved as {self.config.model_name}.pkl")
    def plot_loss(self):
        self.learner.recorder.plot_loss()