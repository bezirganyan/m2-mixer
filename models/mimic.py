from os import path
from typing import List, Optional, Any

import numpy as np
import wandb
import torch
from einops.layers.torch import Rearrange
from omegaconf import DictConfig
from softadapt import LossWeightedSoftAdapt
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import F1Score, Accuracy, Precision, Recall

import modules
from modules.train_test_module import AbstractTrainTestModule


class MultiOFFMixerMultiLoss(AbstractTrainTestModule):
    def __init__(self, model_cfg: DictConfig, optimizer_cfg: DictConfig, **kwargs):
        self.num_classes = model_cfg.modalities.classification.get('num_classes', 3)
        super(MultiOFFMixerMultiLoss, self).__init__(optimizer_cfg, log_confusion_matrix=True, **kwargs)
        self.modalities_freezed = False
        self.optimizer_cfg = optimizer_cfg
        self.checkpoint_path = None
        self.mute = model_cfg.get('mute', None)
        self.freeze_modalities_on_epoch = model_cfg.get('freeze_modalities_on_epoch', None)
        self.random_modality_muting_on_freeze = model_cfg.get('random_modality_muting_on_freeze', False)
        self.muting_probs = model_cfg.get('muting_probs', None)
        static_config = model_cfg.modalities.static
        time_config = model_cfg.modalities.time
        multimodal_config = model_cfg.modalities.multimodal
        dropout = model_cfg.get('dropout', 0.0)
        self.time_mixer = modules.get_block_by_name(**time_config, dropout=dropout)
        self.static_extractor = modules.get_block_by_name(**static_config, dropout=dropout)
        self.fusion_function = modules.get_fusion_by_name(**model_cfg.modalities.multimodal)
        num_patches = self.fusion_function.get_output_shape(1, self.time_mixer.num_patch,
                                                            dim=1)
        self.fusion_mixer = modules.get_block_by_name(**multimodal_config, num_patches=num_patches, dropout=dropout)
        self.classifier_static = torch.nn.Linear(model_cfg.modalities.static.output_dim,
                                                 model_cfg.modalities.classification.num_classes)
        self.classifier_time = torch.nn.Linear(model_cfg.modalities.time.hidden_dim,
                                               model_cfg.modalities.classification.num_classes)
        self.classifier_fusion = modules.get_classifier_by_name(**model_cfg.modalities.classification)
        # weight = torch.tensor([1.32156605, 286.73626374, 74.12784091, 99.59160305,
        #                        9.00068989, 9.50564663])
        weight = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        self.static_criterion = CrossEntropyLoss(weight=weight)
        self.time_criterion = CrossEntropyLoss(weight=weight)
        self.fusion_criterion = CrossEntropyLoss(weight=weight)
        self.fusion_loss_weight = model_cfg.get('fusion_loss_weight', 1.0 / 3)
        self.fusion_loss_change = model_cfg.get('fusion_loss_change', 0)
        self.use_softadapt = model_cfg.get('use_softadapt', False)
        if self.use_softadapt:
            self.static_criterion_history = list()
            self.time_criterion_history = list()
            self.fusion_criterion_history = list()
            self.update_loss_weights_per_epoch = model_cfg.get('update_loss_weights_per_epoch', 6)
            self.softadapt = LossWeightedSoftAdapt(beta=-0.1, accuracy_order=self.update_loss_weights_per_epoch - 1)
        # self.init_weights()

    def shared_step(self, batch, **kwargs):
        # Load data
        static, time, labels = batch

        # get modality encodings from feature extractors
        static_logits = self.static_extractor(static)
        time = self.time_mixer(time)

        # fuse modalities
        fused_moalities = self.fusion_function(static_logits.unsqueeze(1), time)
        logits = self.fusion_mixer(fused_moalities)

        # get classification logits
        static_logits = self.classifier_static(static_logits)
        time_logits = self.classifier_time(time.mean(1))
        logits = self.classifier_fusion(logits)

        # compute losses
        loss_fusion = self.fusion_criterion(logits, labels)
        loss_static = self.static_criterion(static_logits, labels)
        loss_time = self.time_criterion(time_logits, labels)

        ow = (1 - self.fusion_loss_weight) / 2
        loss = self.fusion_loss_weight * loss_fusion + ow * loss_static + ow * loss_time

        # get predictions
        preds = torch.softmax(logits, dim=1).argmax(dim=1)
        preds_static = torch.softmax(static_logits, dim=1).argmax(dim=1)
        preds_time = torch.softmax(time_logits, dim=1).argmax(dim=1)

        return {
            'preds': preds,
            'preds_static': preds_static,
            'preds_time': preds_time,
            'labels': labels.long(),
            'loss': loss,
            'loss_fusion': loss_fusion,
            'loss_static': loss_static,
            'loss_time': loss_time,
            'logits': logits,
            'logits_static': static_logits,
            'logits_time': time_logits,
        }

    def training_epoch_end(self, outputs) -> None:
        super().training_epoch_end(outputs)
        self.fusion_loss_weight = min(1, self.fusion_loss_weight + self.fusion_loss_change)
        wandb.log({'train_loss_static': torch.stack([x['loss_static'] for x in outputs]).mean().item()})
        wandb.log({'train_loss_time': torch.stack([x['loss_time'] for x in outputs]).mean().item()})
        wandb.log({'train_loss_fusion': torch.stack([x['loss_fusion'] for x in outputs]).mean().item()})
        self.log('train_loss_fusion', torch.stack([x['loss_fusion'] for x in outputs]).mean().item())

    def setup_criterion(self) -> torch.nn.Module:
        return None

    def setup_scores(self) -> List[torch.nn.Module]:
        train_scores = dict(f1_micro=F1Score(task="multiclass", num_classes=self.num_classes, average="micro"),
                            accuracy_micro=Accuracy(task="multiclass", num_classes=self.num_classes, average="micro"),
                            precision_micro=Precision(task="multiclass", num_classes=self.num_classes, average="micro"),
                            recall_micro=Recall(task="multiclass", num_classes=self.num_classes, average="micro"))

        val_scores = dict(f1_micro=F1Score(task="multiclass", num_classes=self.num_classes, average="micro"),
                          accuracy_micro=Accuracy(task="multiclass", num_classes=self.num_classes, average="micro"),
                          precision_micro=Precision(task="multiclass", num_classes=self.num_classes, average="micro"),
                          recall_micro=Recall(task="multiclass", num_classes=self.num_classes, average="micro"))

        test_scores = dict(f1_micro=F1Score(task="multiclass", num_classes=self.num_classes, average="micro"),
                           accuracy_micro=Accuracy(task="multiclass", num_classes=self.num_classes, average="micro"),
                           precision_micro=Precision(task="multiclass", num_classes=self.num_classes, average="micro"),
                           recall_micro=Recall(task="multiclass", num_classes=self.num_classes, average="micro"))

        return [train_scores, val_scores, test_scores]

    @classmethod
    def load_from_checkpoint(
            cls,
            checkpoint_path,
            map_location=None,
            hparams_file: Optional = None,
            strict: bool = True,
            **kwargs: Any,
    ):
        model = super().load_from_checkpoint(checkpoint_path, map_location, hparams_file, strict, **kwargs)
        model.checkpoint_path = checkpoint_path
        return model

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
