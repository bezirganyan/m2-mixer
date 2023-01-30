from typing import List

import torch
from omegaconf import DictConfig
from torch.nn import CrossEntropyLoss
from torchmetrics import Accuracy

from models.train_test_module import AbstractTrainTestModule
from modules.mixer import MLPool


class ImagenetPooler(AbstractTrainTestModule):
    def __init__(self, model_cfg: DictConfig, optimizer_cfg: DictConfig, **kwargs):
        super(ImagenetPooler, self).__init__(optimizer_cfg, **kwargs)
        self.optimizer_cfg = optimizer_cfg
        self.model = MLPool(**model_cfg.modalities.image)
        self.classifier = torch.nn.Linear(model_cfg.modalities.image.hidden_dims[-1],
                                          model_cfg.modalities.classification.num_classes)

    def shared_step(self, batch):
        image, labels = batch
        logits = self.model(image)

        logits = self.classifier(logits.mean(dim=1))
        loss = self.criterion(logits, labels)
        preds = torch.softmax(logits, dim=1).argmax(dim=1)

        return {
            'preds': preds,
            'labels': labels,
            'loss': loss
        }

    def setup_criterion(self) -> torch.nn.Module:
        return CrossEntropyLoss()

    def setup_scores(self) -> List[torch.nn.Module]:
        train_scores = dict(acc=Accuracy(task="multiclass", num_classes=1000))
        val_scores = dict(acc=Accuracy(task="multiclass", num_classes=1000))
        test_scores = dict(acc=Accuracy(task="multiclass", num_classes=1000))

        return [train_scores, val_scores, test_scores]
