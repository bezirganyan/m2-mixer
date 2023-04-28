from copy import deepcopy
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
from torch.utils.data import Subset, DataLoader
from torchmetrics import F1Score, Accuracy, Precision, Recall

import modules
from modules.gradblend import GradBlend
from modules.train_test_module import AbstractTrainTestModule


class MimicMixerMultiLoss(AbstractTrainTestModule):
    def __init__(self, model_cfg: DictConfig, optimizer_cfg: DictConfig, **kwargs):
        self.num_classes = model_cfg.modalities.classification.get('num_classes', 3)
        super(MimicMixerMultiLoss, self).__init__(optimizer_cfg, log_confusion_matrix=True, **kwargs)
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
        self.time_encoder = modules.get_block_by_name(**time_config, dropout=dropout)
        self.static_extractor = modules.get_block_by_name(**static_config, dropout=dropout)
        self.fusion_function = modules.get_fusion_by_name(**model_cfg.modalities.multimodal)
        num_patches = self.fusion_function.get_output_shape(1, self.time_encoder.num_patch,
                                                            dim=1)
        self.fusion_mixer = modules.get_block_by_name(**multimodal_config, num_patches=num_patches, dropout=dropout)
        self.classifier_static = torch.nn.Linear(model_cfg.modalities.static.output_dim,
                                                 model_cfg.modalities.classification.num_classes)
        self.classifier_time = torch.nn.Linear(model_cfg.modalities.time.hidden_dim,
                                               model_cfg.modalities.classification.num_classes)
        self.classifier_fusion = modules.get_classifier_by_name(**model_cfg.modalities.classification)
        self.static_criterion = CrossEntropyLoss()
        self.time_criterion = CrossEntropyLoss()
        self.fusion_criterion = CrossEntropyLoss()
        self.fusion_loss_weight = model_cfg.get('fusion_loss_weight', 1.0 / 3)
        self.fusion_loss_change = model_cfg.get('fusion_loss_change', 0)
        self.loss_change_epoch = model_cfg.get('loss_change_epoch', 0)
        self.use_softadapt = model_cfg.get('use_softadapt', False)
        if self.use_softadapt:
            self.static_criterion_history = []
            self.time_criterion_history = []
            self.fusion_criterion_history = []
            self.update_loss_weights_per_epoch = model_cfg.get('update_loss_weights_per_epoch', 6)
            self.softadapt = LossWeightedSoftAdapt(beta=-0.1, accuracy_order=self.update_loss_weights_per_epoch - 1)
        # self.init_weights()
        self.use_gradblend = model_cfg.get('gradblend', False)
        if self.use_gradblend:
            self.gb_update_freq = model_cfg.get('gb_update_freq', 20)
            self.gb_weights = None
            self.gradblend = None
            self.gb_train_loader = None
            self.gb_val_loader = None

    def on_train_epoch_start(self) -> None:
        if self.use_gradblend and self.current_epoch % self.gb_update_freq == 0:
            encoders = [deepcopy(self.static_extractor), deepcopy(self.time_encoder)]
            heads = [deepcopy(self.classifier_static), deepcopy(self.classifier_time)]
            if (self.gb_val_loader is None) or (self.gb_train_loader is None):
                ds = self.trainer.train_dataloader.dataset.datasets
                ds_train = Subset(ds, range(int(len(ds) * 0.1), len(ds)))
                ds_val = Subset(ds, range(int(len(ds) * 0.1)))
                bs = self.trainer.train_dataloader.loaders.batch_size
                self.gb_train_loader = DataLoader(ds_train, batch_size=bs, shuffle=True)
                self.gb_val_loader = DataLoader(ds_val, batch_size=bs, shuffle=True)
            self.gradblend = GradBlend(self, encoders, heads, deepcopy(self.fusion_mixer),
                                       deepcopy(self.classifier_fusion),
                                       nn.CrossEntropyLoss, self.gb_train_loader, self.gb_val_loader)
            self.gb_weights = self.gradblend.get_weights()
            print("GradBlend weights:", self.gb_weights)

    def shared_step(self, batch, mode='train', **kwargs):
        # Load data
        static, time, labels = batch

        # get modality encodings from feature extractors
        static_logits = self.static_extractor(static)
        time = self.time_encoder(time)

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
        if (
            self.use_gradblend
            and self.gb_weights is None
            or not self.use_gradblend
        ):
            loss = self.fusion_loss_weight * loss_fusion + ow * loss_static + ow * loss_time
        else:
            loss = self.gb_weights[2] * loss_fusion + self.gb_weights[0] * loss_static + self.gb_weights[
                1] * loss_time
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

    def validation_epoch_end(self, outputs):
        super().validation_epoch_end(outputs)
        val_loss_fusion = torch.stack([x['loss_fusion'] for x in outputs]).mean().item()
        self.log('val_loss_fusion', val_loss_fusion)
        wandb.log({'val_loss_fusion': val_loss_fusion})
        if self.current_epoch >= self.loss_change_epoch:
            self.fusion_loss_weight = min(1, self.fusion_loss_weight + self.fusion_loss_change)
    def training_epoch_end(self, outputs) -> None:
        super().training_epoch_end(outputs)
        wandb.log({'train_loss_static': torch.stack([x['loss_static'] for x in outputs]).mean().item()})
        wandb.log({'train_loss_time': torch.stack([x['loss_time'] for x in outputs]).mean().item()})
        wandb.log({'train_loss_fusion': torch.stack([x['loss_fusion'] for x in outputs]).mean().item()})
        self.log('train_loss_fusion', torch.stack([x['loss_fusion'] for x in outputs]).mean().item())

    def setup_criterion(self) -> torch.nn.Module:
        return None

    def setup_scores(self) -> List[torch.nn.Module]:
        train_scores = dict(f1_micro=F1Score(task="multiclass", num_classes=self.num_classes, average="micro"),
                            acc=Accuracy(task="multiclass", num_classes=self.num_classes, average="micro"),
                            precision_micro=Precision(task="multiclass", num_classes=self.num_classes, average="micro"),
                            recall_micro=Recall(task="multiclass", num_classes=self.num_classes, average="micro"))

        val_scores = dict(f1_micro=F1Score(task="multiclass", num_classes=self.num_classes, average="micro"),
                          acc=Accuracy(task="multiclass", num_classes=self.num_classes, average="micro"),
                          precision_micro=Precision(task="multiclass", num_classes=self.num_classes, average="micro"),
                          recall_micro=Recall(task="multiclass", num_classes=self.num_classes, average="micro"))

        test_scores = dict(f1_micro=F1Score(task="multiclass", num_classes=self.num_classes, average="micro"),
                           acc=Accuracy(task="multiclass", num_classes=self.num_classes, average="micro"),
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

    def configure_optimizers(self):
        optimizer_cfg = self.optimizer_cfg
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), **optimizer_cfg)
        scheduler = ReduceLROnPlateau(optimizer, patience=5, verbose=True)

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss_fusion",
        }


class MimicRecurrent(MimicMixerMultiLoss):
    def __init__(self, model_cfg: DictConfig, optimizer_cfg: DictConfig, **kwargs):
        self.num_classes = model_cfg.modalities.classification.get('num_classes', 6)
        super(MimicMixerMultiLoss, self).__init__(optimizer_cfg, log_confusion_matrix=True, **kwargs)
        self.optimizer_cfg = optimizer_cfg
        self.checkpoint_path = None
        static_config = model_cfg.modalities.static
        time_config = model_cfg.modalities.time
        multimodal_config = model_cfg.modalities.multimodal
        dropout = model_cfg.get('dropout', 0.0)
        self.time_encoder = modules.get_block_by_name(**time_config, dropout=dropout)
        self.static_extractor = modules.get_block_by_name(**static_config, dropout=dropout)
        self.fusion_function = modules.get_fusion_by_name(**model_cfg.modalities.multimodal)
        self.fusion_encoder = modules.get_block_by_name(**multimodal_config, dropout=dropout)
        self.classifier_static = torch.nn.Linear(model_cfg.modalities.static.output_dim,
                                                 model_cfg.modalities.classification.num_classes)
        time_out = model_cfg.modalities.time.input_dim * model_cfg.modalities.time.hidden_dim * 2
        self.classifier_time = torch.nn.Linear(time_out,
                                               model_cfg.modalities.classification.num_classes)
        self.classifier_fusion = modules.get_classifier_by_name(**model_cfg.modalities.classification)
        self.static_criterion = CrossEntropyLoss()
        self.time_criterion = CrossEntropyLoss()
        self.fusion_criterion = CrossEntropyLoss()
        self.fusion_loss_weight = model_cfg.get('fusion_loss_weight', 1.0 / 3)
        self.fusion_loss_change = model_cfg.get('fusion_loss_change', 0)
        self.loss_change_epoch = model_cfg.get('loss_change_epoch', 0)
        self.use_softadapt = model_cfg.get('use_softadapt', False)
        if self.use_softadapt:
            self.static_criterion_history = []
            self.time_criterion_history = []
            self.fusion_criterion_history = []
            self.update_loss_weights_per_epoch = model_cfg.get('update_loss_weights_per_epoch', 6)
            self.softadapt = LossWeightedSoftAdapt(beta=-0.1, accuracy_order=self.update_loss_weights_per_epoch - 1)
        # self.init_weights()
        self.use_gradblend = model_cfg.get('gradblend', False)
        if self.use_gradblend:
            self.gb_update_freq = model_cfg.get('gb_update_freq', 20)
            self.gb_weights = None
            self.gb_train_loader = None
            self.gradblend = None
            self.gb_val_loader = None

    def shared_step(self, batch, mode='train', **kwargs):
        # Load data
        static, time, labels = batch

        # get modality encodings from feature extractors
        static_logits = self.static_extractor(static)
        time = self.time_encoder(time)

        # fuse modalities
        fused_moalities = self.fusion_function(static_logits, time)
        logits = self.fusion_encoder(fused_moalities)

        # get classification logits
        static_logits = self.classifier_static(static_logits)
        time_logits = self.classifier_time(time)
        logits = self.classifier_fusion(logits)

        # compute losses
        loss_fusion = self.fusion_criterion(logits, labels)
        loss_static = self.static_criterion(static_logits, labels)
        loss_time = self.time_criterion(time_logits, labels)

        ow = (1 - self.fusion_loss_weight) / 2
        if (
            self.use_gradblend
            and self.gb_weights is None
            or not self.use_gradblend
        ):
            loss = self.fusion_loss_weight * loss_fusion + ow * loss_static + ow * loss_time
        else:
            loss = self.gb_weights[2] * loss_fusion + self.gb_weights[0] * loss_static + self.gb_weights[
                1] * loss_time
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

    def on_train_epoch_start(self) -> None:
        if self.use_gradblend and self.current_epoch % self.gb_update_freq == 0:
            encoders = [deepcopy(self.static_extractor), deepcopy(self.time_encoder)]
            heads = [deepcopy(self.classifier_static), deepcopy(self.classifier_time)]
            if (self.gb_val_loader is None) or (self.gb_train_loader is None):
                ds = self.trainer.train_dataloader.dataset.datasets
                ds_train = Subset(ds, range(int(len(ds) * 0.1), len(ds)))
                ds_val = Subset(ds, range(int(len(ds) * 0.1)))
                bs = self.trainer.train_dataloader.loaders.batch_size
                self.gb_train_loader = DataLoader(ds_train, batch_size=bs, shuffle=True)
                self.gb_val_loader = DataLoader(ds_val, batch_size=bs, shuffle=True)
            self.gradblend = GradBlend(self, encoders, heads, deepcopy(self.fusion_encoder),
                                       deepcopy(self.classifier_fusion),
                                       nn.CrossEntropyLoss, self.gb_train_loader, self.gb_val_loader)
            self.gb_weights = self.gradblend.get_weights()
            print("GradBlend weights:", self.gb_weights)

    def configure_optimizers(self):
        optimizer_cfg = self.optimizer_cfg
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), **optimizer_cfg)
        scheduler = ReduceLROnPlateau(optimizer, patience=5, verbose=True, mode='min')

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }

    def validation_epoch_end(self, outputs):
        super().validation_epoch_end(outputs)
        val_loss_fusion = torch.stack([x['loss_fusion'] for x in outputs]).mean().item()
        self.log('val_loss_fusion', val_loss_fusion)
        wandb.log({'val_loss_fusion': val_loss_fusion})
