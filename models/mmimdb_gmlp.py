from typing import List

import torch
from omegaconf import DictConfig
from torch.nn import BCEWithLogitsLoss
from torchmetrics import F1Score

from models.train_test_module import AbstractTrainTestModule
from modules.gmpl import VisiongMLP


class MMIDB_GMLP(AbstractTrainTestModule):
    def __init__(self, model_cfg: DictConfig, optimizer_cfg: DictConfig, **kwargs):
        super(MMIDB_GMLP, self).__init__(optimizer_cfg, **kwargs)
        self.model = VisiongMLP(dropout=model_cfg.dropout, **model_cfg.modalities.image)

    def setup_criterion(self) -> torch.nn.Module:
        return BCEWithLogitsLoss()

    def setup_scores(self) -> List[torch.nn.Module]:
        train_score = F1Score(task="multilabel", num_labels=23, average='weighted')
        val_score = F1Score(task="multilabel", num_labels=23, average='weighted')
        test_score = F1Score(task="multilabel", num_labels=23, average='weighted')

        return [train_score, val_score, test_score]

    def shared_step(self, batch):
        image = batch['image']
        text = batch['text']
        labels = batch['label']
        logits = self.model(image.float())
        loss = self.criterion(logits, labels.float())
        preds = torch.sigmoid(logits) > 0.5

        return {
            'preds': preds,
            'labels': labels,
            'loss': loss
        }