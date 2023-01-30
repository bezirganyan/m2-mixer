from typing import List

import timm
import torch
from omegaconf import DictConfig
from torch.nn import BCEWithLogitsLoss
from torchmetrics import F1Score

from models.train_test_module import AbstractTrainTestModule
from modules.gmpl import VisiongMLP

from torchvision.models import vgg19_bn


class MMIDB_GMLP(AbstractTrainTestModule):
    def __init__(self, model_cfg: DictConfig, optimizer_cfg: DictConfig, **kwargs):
        super(MMIDB_GMLP, self).__init__(optimizer_cfg, **kwargs)
        # self.model = VisiongMLP(dropout=model_cfg.dropout, **model_cfg.modalities.image)
        # self.model = timm.create_model('mixer_b16_224_miil_in21k', pretrained=True, num_classes=23)
        self.model = vgg19_bn(pretrained=True)
        self.model.classifier = torch.nn.Sequential(
            torch.nn.Linear(25088, 4096),
            torch.nn.ReLU(True),
            torch.nn.Dropout(),
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(True),
            torch.nn.Dropout(),
            torch.nn.Linear(4096, 23),
        )

    def setup_criterion(self) -> torch.nn.Module:
        # pw = torch.tensor([4.57642832, 7.38544978, 10.79846869, 13.23391421,
        #                    15.59020924, 18.62735849, 22.48861048, 25.21711367,
        #                    74.50943396, 31.31641554, 31.79549114, 32.90833333,
        #                    39.64859438, 56.90201729, 40.46106557, 58.24483776,
        #                    67.3890785, 84.92473118, 58.33087149, 62.68253968,
        #                    114.13294798, 141.54121864, 116.83431953])
        pw = torch.tensor([4.69368723, 7.20594714, 11.74685817, 12.27579737,
                           16.86340206, 17.9260274, 24.32342007, 25.96428571,
                           31.45673077, 32.55223881, 34.80319149, 31.60869565,
                           37.17613636, 44.81506849, 57.90265487, 56.89565217,
                           61.72641509, 60.02752294, 82.82278481, 94.82608696,
                           96.22058824, 110.89830508, 198.27272727])
        return BCEWithLogitsLoss(pos_weight=pw)

    def setup_scores(self) -> List[torch.nn.Module]:
        train_scores = dict(f1w=F1Score(task="multilabel", num_labels=23, average='weighted'),
                            f1m=F1Score(task="multilabel", num_labels=23, average='macro'))
        val_scores = dict(f1w=F1Score(task="multilabel", num_labels=23, average='weighted'),
                          f1m=F1Score(task="multilabel", num_labels=23, average='macro'))
        test_scores = dict(f1w=F1Score(task="multilabel", num_labels=23, average='weighted'),
                           f1m=F1Score(task="multilabel", num_labels=23, average='macro'))

        return [train_scores, val_scores, test_scores]

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


class MMIDB_GMLP_ext(AbstractTrainTestModule):
    def __init__(self, model_cfg: DictConfig, optimizer_cfg: DictConfig, **kwargs):
        super(MMIDB_GMLP_ext, self).__init__(optimizer_cfg, **kwargs)
        self.model = VisiongMLP(dropout=model_cfg.dropout, **model_cfg.modalities.image)

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
