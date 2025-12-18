import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from fastai.vision.all import *
from .config import TrainConfig

def _get_backbone(backbone_name: str):
    backbone_dict = {
        "resnet18": resnet18,
        "resnet34": resnet34,
        "resnet50": resnet50,
        "resnet101": resnet101,

    }
    if backbone_name not in backbone_dict:
        raise ValueError(f"Backbone '{backbone_name}' is not supported.")
    return backbone_dict[backbone_name]


def create_learner(dls, config: TrainConfig) -> Learner:
    arch = _get_backbone(config.backbone)
    learn = vision_learner(dls, arch, metrics=accuracy, pretrained=True)
    return learn

