from typing import List

import torch
from omegaconf import DictConfig
from torch.nn import BCEWithLogitsLoss
from torchmetrics import F1Score

from models.mmixer import MultimodalMixer
from models.train_test_module import AbstractTrainTestModule
from modules.mixer import MLPool


class MMIDBMixer(AbstractTrainTestModule):
    def __init__(self, model_cfg: DictConfig, optimizer_cfg: DictConfig, **kwargs):
        super(MMIDBMixer, self).__init__(optimizer_cfg, **kwargs)
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
        logits = self.model(image.float(), text)
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
        train_scores = dict(f1w=F1Score(task="multilabel", num_labels=23, average='weighted'),
                            f1m=F1Score(task="multilabel", num_labels=23, average='macro'))
        val_scores = dict(f1w=F1Score(task="multilabel", num_labels=23, average='weighted'),
                          f1m=F1Score(task="multilabel", num_labels=23, average='macro'))
        test_scores = dict(f1w=F1Score(task="multilabel", num_labels=23, average='weighted'),
                           f1m=F1Score(task="multilabel", num_labels=23, average='macro'))

        return [train_scores, val_scores, test_scores]


class MMIDBPooler(AbstractTrainTestModule):
    def __init__(self, model_cfg: DictConfig, optimizer_cfg: DictConfig, **kwargs):
        super(MMIDBPooler, self).__init__(optimizer_cfg, **kwargs)
        self.optimizer_cfg = optimizer_cfg
        self.model = MLPool(**model_cfg.modalities.image)
        self.classifier = torch.nn.Linear(model_cfg.modalities.image.hidden_dims[-1],
                                          model_cfg.modalities.classification.num_classes)

    def shared_step(self, batch):
        image = batch['image']
        text = batch['text']
        labels = batch['label']
        logits = self.model(image)
        logits = self.classifier(logits.mean(dim=1))
        loss = self.criterion(logits, labels.float())
        preds = torch.sigmoid(logits)

        return {
            'preds': preds,
            'labels': labels,
            'loss': loss
        }

    def setup_criterion(self) -> torch.nn.Module:
        return BCEWithLogitsLoss(pos_weight=self.loss_pos_weight)

    def setup_scores(self) -> List[torch.nn.Module]:
        train_scores = dict(f1w=F1Score(task="multilabel", num_labels=23, average='weighted'),
                            f1m=F1Score(task="multilabel", num_labels=23, average='macro'))
        val_scores = dict(f1w=F1Score(task="multilabel", num_labels=23, average='weighted'),
                          f1m=F1Score(task="multilabel", num_labels=23, average='macro'))
        test_scores = dict(f1w=F1Score(task="multilabel", num_labels=23, average='weighted'),
                           f1m=F1Score(task="multilabel", num_labels=23, average='macro'))

        return [train_scores, val_scores, test_scores]
