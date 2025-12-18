from fastai.vision.all import *
from pathlib import Path

class Evaluator:
    def __init__(self, learn: Learner):
        self.learner = learn

    def plot_confusion(self,figsize=(10,10),dpi=100):
        interp = ClassificationInterpretation.from_learner(self.learner)
        interp.plot_confusion_matrix(figsize=figsize,dpi = dpi)

    def plot_top_losses(self, n=9,nrows=3,figsize=(9,9)):
        interp = ClassificationInterpretation.from_learner(self.learner)
        interp.plot_top_losses(n, nrows=nrows, figsize=figsize)

    def test_accuracy(self, test_path: Path)-> Tensor:
        test_files = get_image_files(test_path)
        test_dl = self.learner.dls.test_dl(test_files, with_labels=True)
        preds, targs = self.learner.get_preds(dl=test_dl)
        acc_tensor = accuracy(preds, targs)
        acc = acc_tensor.item()
        print(f"Test Accuracy: {acc:.4f}")
        return acc


