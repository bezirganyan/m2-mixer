from typing import List

from omegaconf import DictConfig
from torch.nn import BCEWithLogitsLoss
from torchmetrics import F1Score
from torchvision.models import resnet50
from torch import nn
import torch

from modules.train_test_module import AbstractTrainTestModule


class ConvNet(AbstractTrainTestModule):
    def __init__(self, model_cfg: DictConfig, optimizer_cfg: DictConfig, **kwargs):
        super(ConvNet, self).__init__(optimizer_cfg, **kwargs)
        self.backbone = resnet50(weights="IMAGENET1K_V2")
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.fc = nn.Sequential(nn.Linear(2048, 512),
                                          nn.Linear(512, 128),
                                          nn.Linear(128, model_cfg.modalities.classification.num_classes))
        # self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    def shared_step(self, batch):
        image = batch['image']
        text = batch['text']
        labels = batch['label']
        logits = self.backbone(image.float())
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
