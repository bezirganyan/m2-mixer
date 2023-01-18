from typing import List, Dict, Any

import numpy as np
import torch
import wandb
from omegaconf import DictConfig
# from sklearn.metrics import f1_score
from torch.nn import BCEWithLogitsLoss
import pytorch_lightning as pl
from torchmetrics import F1Score

from models.mmixer import MultimodalMixer
from models.train_test_module import AbstractTrainTestModule


class MMIDB_Mixer(AbstractTrainTestModule):
    def __init__(self, model_cfg: DictConfig, optimizer_cfg: DictConfig, **kwargs):
        super(MMIDB_Mixer, self).__init__(optimizer_cfg, **kwargs)
        self.optimizer_cfg = optimizer_cfg
        self.model = MultimodalMixer(
            model_cfg.modalities.image,
            model_cfg.modalities.text,
            model_cfg.modalities.multimodal,
            model_cfg.modalities.bottleneck,
            model_cfg.modalities.classification,
            dropout=model_cfg.dropout
        )
        self.criterion = BCEWithLogitsLoss()
        self.train_score = F1Score(task="multilabel", num_labels=23, average='weighted')
        self.val_score = F1Score(task="multilabel", num_labels=23, average='weighted')
        self.test_score = F1Score(task="multilabel", num_labels=23, average='weighted')

    def shared_step(self, batch):
        image = batch['image']
        text = batch['text']
        labels = batch['label']
        logits = self.model(image, text)
        loss = self.criterion(logits, labels.float())
        preds = torch.sigmoid(logits) > 0.5

        return {
            'preds': preds,
            'labels': labels,
            'loss': loss
        }

    def setup_criterion(self) -> torch.nn.Module:
        return BCEWithLogitsLoss()

    def setup_scores(self) -> List[torch.nn.Module]:
        train_score = F1Score(task="multilabel", num_labels=23, average='weighted')
        val_score = F1Score(task="multilabel", num_labels=23, average='weighted')
        test_score = F1Score(task="multilabel", num_labels=23, average='weighted')

        return [train_score, val_score, test_score]