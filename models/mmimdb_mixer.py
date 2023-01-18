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
        pw = torch.tensor([4.57642832, 7.38544978, 10.79846869, 13.23391421,
                           15.59020924, 18.62735849, 22.48861048, 25.21711367,
                           74.50943396, 31.31641554, 31.79549114, 32.90833333,
                           39.64859438, 56.90201729, 40.46106557, 58.24483776,
                           67.3890785, 84.92473118, 58.33087149, 62.68253968,
                           114.13294798, 141.54121864, 116.83431953])
        return BCEWithLogitsLoss(pos_weight=pw)

    def setup_scores(self) -> List[torch.nn.Module]:
        train_score = F1Score(task="multilabel", num_labels=23, average='weighted')
        val_score = F1Score(task="multilabel", num_labels=23, average='weighted')
        test_score = F1Score(task="multilabel", num_labels=23, average='weighted')

        return [train_score, val_score, test_score]
