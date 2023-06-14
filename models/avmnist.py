from abc import ABC, abstractmethod
from copy import deepcopy
from os import path

import numpy as np
import wandb
from omegaconf import DictConfig
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset

from modules.gradblend import GradBlend
from modules.losses import EDLMSELoss
from modules.train_test_module import AbstractTrainTestModule
from modules.mixer import MLPMixer

import torch

from typing import Dict, List, Any, Optional
from torch.nn import CrossEntropyLoss
from torchmetrics import Accuracy, F1Score, Metric, Precision, Recall
import modules

try:
    from softadapt import LossWeightedSoftAdapt
except ModuleNotFoundError:
    print('Warning: Could not import softadapt. LossWeightedSoftAdapt will not be available.')
    LossWeightedSoftAdapt = None


def setup_scores():
    train_scores = dict(acc=Accuracy(task="multiclass", num_classes=10),
                        f1m=F1Score(task="multiclass", num_classes=10, average='macro'),
                        prec_m=Precision(task="multiclass", num_classes=10, average='macro'),
                        rec_m=Recall(task="multiclass", num_classes=10, average='macro'))
    val_scores = dict(acc=Accuracy(task="multiclass", num_classes=10),
                      f1m=F1Score(task="multiclass", num_classes=10, average='macro'),
                      prec_m=Precision(task="multiclass", num_classes=10, average='macro'),
                      rec_m=Recall(task="multiclass", num_classes=10, average='macro'))
    test_scores = dict(acc=Accuracy(task="multiclass", num_classes=10),
                       f1m=F1Score(task="multiclass", num_classes=10, average='macro'),
                       prec_m=Precision(task="multiclass", num_classes=10, average='macro'),
                       rec_m=Recall(task="multiclass", num_classes=10, average='macro'))
    return test_scores, train_scores, val_scores


class AbstractAVMnistMixer(AbstractTrainTestModule, ABC):
    def __init__(self, model_cfg: DictConfig, optimizer_cfg: DictConfig, **kwargs):
        super(AbstractAVMnistMixer, self).__init__(optimizer_cfg, **kwargs)
        self.optimizer_cfg = optimizer_cfg
        self.model = None
        self.classifier = None

    def shared_step(self, batch, **kwargs):
        logits = self.get_logits(batch)
        labels = batch['label']
        loss = self.criterion(logits, labels)
        preds = torch.softmax(logits, dim=1).argmax(dim=1)

        return {
            'preds': preds,
            'labels': labels,
            'loss': loss
        }

    @abstractmethod
    def get_logits(self, batch):
        raise NotImplementedError

    def setup_criterion(self) -> torch.nn.Module:
        return CrossEntropyLoss()

    def setup_scores(self) -> List[torch.nn.Module]:
        return setup_scores()

    def configure_optimizers(self):
        optimizer_cfg = self.optimizer_cfg
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), **optimizer_cfg)
        scheduler = ReduceLROnPlateau(optimizer, patience=5, verbose=True)

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }


class AVMnistImageMixer(AbstractAVMnistMixer):
    def __init__(self, model_cfg: DictConfig, optimizer_cfg: DictConfig, **kwargs):
        super(AVMnistImageMixer, self).__init__(model_cfg, optimizer_cfg, **kwargs)
        self.model = MLPMixer(**model_cfg.modalities.image, dropout=model_cfg.dropout)
        self.classifier = torch.nn.Linear(model_cfg.modalities.image.hidden_dim,
                                          model_cfg.modalities.classification.num_classes)

    def get_logits(self, batch):
        image = batch['image']
        labels = batch['label']
        logits = self.model(image)
        logits = self.classifier(logits.mean(dim=1))
        return logits


class AVMnistAudioMixer(AbstractAVMnistMixer):
    def __init__(self, model_cfg: DictConfig, optimizer_cfg: DictConfig, **kwargs):
        super(AVMnistAudioMixer, self).__init__(model_cfg, optimizer_cfg, **kwargs)
        self.model = MLPMixer(**model_cfg.modalities.audio, dropout=model_cfg.dropout)
        self.classifier = torch.nn.Linear(model_cfg.modalities.audio.hidden_dim,
                                          model_cfg.modalities.classification.num_classes)

    def get_logits(self, batch):
        audio = batch['audio']
        labels = batch['label']
        logits = self.model(audio)
        logits = self.classifier(logits.mean(dim=1))
        return logits


class AVMnistMixer(AbstractAVMnistMixer):
    def __init__(self, model_cfg: DictConfig, optimizer_cfg: DictConfig, **kwargs):
        super(AVMnistMixer, self).__init__(model_cfg, optimizer_cfg, **kwargs)
        self.optimizer_cfg = optimizer_cfg
        self.mute = model_cfg.get('mute', None)
        self.train_branch = model_cfg.get('train_branch', None)
        if self.train_branch is not None:
            self.branch_classifier = torch.nn.Linear(
                model_cfg.modalities.image.hidden_dim if self.train_branch == 'image' else model_cfg.modalities.audio.hidden_dim,
                model_cfg.modalities.classification.num_classes)
        image_config = model_cfg.modalities.image
        audio_config = model_cfg.modalities.audio
        multimodal_config = model_cfg.modalities.multimodal
        dropout = model_cfg.get('dropout', 0.0)
        self.image_mixer = modules.get_block_by_name(**image_config, dropout=dropout)
        self.audio_mixer = modules.get_block_by_name(**audio_config, dropout=dropout)
        self.fusion_function = modules.get_fusion_by_name(**model_cfg.modalities.multimodal)
        num_patches = self.fusion_function.get_output_shape(self.image_mixer.num_patch, self.audio_mixer.num_patch,
                                                            dim=1)
        self.fusion_mixer = modules.get_block_by_name(**multimodal_config, num_patches=num_patches, dropout=dropout)
        self.classifier = modules.get_classifier_by_name(**model_cfg.modalities.classification)

    def get_logits(self, batch):
        image = batch['image']
        audio = batch['audio']

        if self.mute == 'image':
            image = torch.zeros_like(image)
        elif self.mute == 'audio':
            audio = torch.zeros_like(audio)

        if self.train_branch == 'image':
            # freeze image branch
            for param in self.image_mixer.parameters():
                param.requires_grad = False

            image_logits = self.image_mixer(image)
            return self.branch_classifier(image_logits.mean(dim=1))

        elif self.train_branch == 'audio':
            # freeze audio branch
            for param in self.audio_mixer.parameters():
                param.requires_grad = False

            audio_logits = self.audio_mixer(audio)
            return self.branch_classifier(audio_logits.mean(dim=1))

        image_logits = self.image_mixer(image)
        audio_logits = self.audio_mixer(audio)

        # fuse modalities
        fused_moalities = self.fusion_function(image_logits, audio_logits)
        logits = self.fusion_mixer(fused_moalities)

        # logits = logits.reshape(logits.shape[0], -1, logits.shape[-1])
        audio_logits = audio_logits.reshape(audio_logits.shape[0], -1, audio_logits.shape[-1])
        image_logits = image_logits.reshape(image_logits.shape[0], -1, image_logits.shape[-1])

        # get logits for each modality
        logits = self.classifier(logits)

        return logits

    @classmethod
    def load_from_checkpoint(
            cls,
            checkpoint_path,
            map_location=None,
            hparams_file: Optional = None,
            strict: bool = True,
            **kwargs: Any,
    ):
        return super().load_from_checkpoint(
            checkpoint_path, map_location, hparams_file, False, **kwargs
        )

    def configure_optimizers(self):
        optimizer_cfg = self.optimizer_cfg
        scheduler_patience = optimizer_cfg.pop('scheduler_patience', 5)
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), **optimizer_cfg)
        # scheduler = ReduceLROnPlateau(optimizer, patience=scheduler_patience, verbose=True)

        return {
            "optimizer": optimizer,
            # "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }


class AVMnistMixerMultiLoss(AbstractTrainTestModule):
    def __init__(self, model_cfg: DictConfig, optimizer_cfg: DictConfig, **kwargs):
        super(AVMnistMixerMultiLoss, self).__init__(optimizer_cfg, log_confusion_matrix=True, **kwargs)
        self.modalities_freezed = False
        self.optimizer_cfg = optimizer_cfg
        self.scheduler_patience = optimizer_cfg.pop('scheduler_patience', 5)
        self.checkpoint_path = None
        self.mute = model_cfg.get('mute', None)
        self.freeze_modalities_on_epoch = model_cfg.get('freeze_modalities_on_epoch', None)
        self.random_modality_muting_on_freeze = model_cfg.get('random_modality_muting_on_freeze', False)
        self.muting_probs = model_cfg.get('muting_probs', None)
        image_config = model_cfg.modalities.image
        audio_config = model_cfg.modalities.audio
        multimodal_config = model_cfg.modalities.multimodal
        dropout = model_cfg.get('dropout', 0.0)
        self.image_mixer = modules.get_block_by_name(**image_config, dropout=dropout)
        self.audio_mixer = modules.get_block_by_name(**audio_config, dropout=dropout)
        self.fusion_function = modules.get_fusion_by_name(**model_cfg.modalities.multimodal)
        num_patches = self.fusion_function.get_output_shape(self.image_mixer.num_patch, self.audio_mixer.num_patch,
                                                            dim=1)
        self.fusion_mixer = modules.get_block_by_name(**multimodal_config, num_patches=num_patches, dropout=dropout)
        self.classifier_image = torch.nn.Linear(model_cfg.modalities.image.hidden_dim,
                                                model_cfg.modalities.classification.num_classes)
        self.classifier_audio = torch.nn.Linear(model_cfg.modalities.audio.hidden_dim,
                                                model_cfg.modalities.classification.num_classes)
        self.classifier_fusion = modules.get_classifier_by_name(**model_cfg.modalities.classification)

        self.image_criterion = CrossEntropyLoss()
        self.audio_criterion = CrossEntropyLoss()
        self.fusion_criterion = CrossEntropyLoss()
        self.fusion_loss_weight = model_cfg.get('fusion_loss_weight', 1.0 / 3)
        self.fusion_loss_change = model_cfg.get('fusion_loss_change', 0)
        self.loss_change_epoch = model_cfg.get('loss_change_epoch', 0)
        self.use_softadapt = model_cfg.get('use_softadapt', False)
        if self.use_softadapt:
            if LossWeightedSoftAdapt is None:
                print('SoftAdapt is not installed! Hence, will not be used!')
                self.use_softadapt = False
            else:
                self.image_criterion_history = []
                self.audio_criterion_history = []
                self.fusion_criterion_history = []
                self.loss_weights = torch.tensor([1.0 / 3, 1.0 / 3, 1.0 / 3], device=self.device)
                self.update_loss_weights_per_epoch = model_cfg.get('update_loss_weights_per_epoch', 6)
                self.softadapt = LossWeightedSoftAdapt(beta=-0.1, accuracy_order=self.update_loss_weights_per_epoch - 1)
        self.use_gradblend = model_cfg.get('gradblend', False)
        if self.use_gradblend:
            self.gb_update_freq = model_cfg.get('gb_update_freq', 20)
            self.gb_weights = None
            self.gradblend = None
            self.gb_train_loader = None
            self.gb_val_loader = None

    def on_train_epoch_start(self) -> None:
        if self.use_gradblend and self.current_epoch % self.gb_update_freq == 0:
            encoders = [deepcopy(self.audio_mixer), deepcopy(self.image_mixer)]
            heads = [deepcopy(self.classifier_audio), deepcopy(self.classifier_image)]
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

    def shared_step(self, batch, **kwargs):
        # Load data

        image = batch['image']
        audio = batch['audio']
        labels = batch['label']

        if kwargs.get('mode', None) == 'train':
            if self.freeze_modalities_on_epoch is not None and (self.current_epoch == self.freeze_modalities_on_epoch) \
                    and not self.modalities_freezed:
                self._freeze_modalities()
            if self.random_modality_muting_on_freeze and (self.current_epoch >= self.freeze_modalities_on_epoch):
                self.mute = np.random.choice(['image', 'audio', 'multimodal'], p=[self.muting_probs['image'],
                                                                                  self.muting_probs['audio'],
                                                                                  self.muting_probs['multimodal']])

            if self.mute != 'multimodal':
                if self.mute == 'image':
                    image = torch.zeros_like(image)
                elif self.mute == 'audio':
                    audio = torch.zeros_like(audio)

        # get modality encodings from feature extractors
        image_logits = self.image_mixer(image)
        audio_logits = self.audio_mixer(audio)

        # fuse modalities
        fused_moalities = self.fusion_function(image_logits, audio_logits)
        logits = self.fusion_mixer(fused_moalities)

        # logits = logits.reshape(logits.shape[0], -1, logits.shape[-1])
        audio_logits = audio_logits.reshape(audio_logits.shape[0], -1, audio_logits.shape[-1])
        image_logits = image_logits.reshape(image_logits.shape[0], -1, image_logits.shape[-1])

        # get logits for each modality
        image_logits = self.classifier_image(image_logits.mean(dim=1))
        audio_logits = self.classifier_audio(audio_logits.mean(dim=1))
        logits = self.classifier_fusion(logits)

        # compute losses
        loss_image = self.image_criterion(image_logits, labels)
        loss_audio = self.audio_criterion(audio_logits, labels)
        loss_fusion = self.fusion_criterion(logits, labels)

        if self.use_softadapt:
            loss = self.loss_weights[0] * loss_image + self.loss_weights[1] * loss_audio + self.loss_weights[
                2] * loss_fusion
        elif (
                self.use_gradblend
                and self.gb_weights is not None
        ):
            loss = self.gb_weights[2] * loss_fusion + self.gb_weights[1] * loss_image + self.gb_weights[
                0] * loss_audio
        else:
            ow = (1 - self.fusion_loss_weight) / 2
            loss = (self.fusion_loss_weight * loss_fusion + ow * loss_image + ow * loss_audio) * 3
        if self.modalities_freezed and kwargs.get('mode', None) == 'train':
            loss = loss_fusion

        # get predictions
        preds = torch.softmax(logits, dim=1).argmax(dim=1)
        preds_image = torch.softmax(image_logits, dim=1).argmax(dim=1)
        preds_audio = torch.softmax(audio_logits, dim=1).argmax(dim=1)

        return {
            'preds': preds,
            'preds_image': preds_image,
            'preds_audio': preds_audio,
            'labels': labels,
            'loss': loss,
            'loss_image': loss_image,
            'loss_audio': loss_audio,
            'loss_fusion': loss_fusion,
            'image_logits': image_logits,
            'audio_logits': audio_logits,
            'logits': logits
        }

    def _freeze_modalities(self):
        print('Freezing modalities')
        for param in self.image_mixer.parameters():
            param.requires_grad = False
        for param in self.audio_mixer.parameters():
            param.requires_grad = False
        for param in self.classifier_image.parameters():
            param.requires_grad = False
        for param in self.classifier_audio.parameters():
            param.requires_grad = False
        self.modalities_freezed = True

    def training_epoch_end(self, outputs) -> None:
        super().training_epoch_end(outputs)
        wandb.log({'train_loss_image': torch.stack([x['loss_image'] for x in outputs]).mean().item()})
        wandb.log({'train_loss_audio': torch.stack([x['loss_audio'] for x in outputs]).mean().item()})
        wandb.log({'train_loss_fusion': torch.stack([x['loss_fusion'] for x in outputs]).mean().item()})
        self.log('train_loss_fusion', torch.stack([x['loss_fusion'] for x in outputs]).mean().item())

    def validation_epoch_end(self, outputs) -> None:
        super().validation_epoch_end(outputs)
        val_loss_fusion = torch.stack([x['loss_fusion'] for x in outputs]).mean().item()
        self.log('val_loss_fusion', val_loss_fusion)
        wandb.log({'val_loss_fusion': val_loss_fusion})
        if self.current_epoch >= self.loss_change_epoch:
            self.fusion_loss_weight = min(1, self.fusion_loss_weight + self.fusion_loss_change)
        if self.use_softadapt:
            self.image_criterion_history.append(torch.stack([x['loss_image'] for x in outputs]).mean().item())
            self.audio_criterion_history.append(torch.stack([x['loss_audio'] for x in outputs]).mean().item())
            self.fusion_criterion_history.append(torch.stack([x['loss_fusion'] for x in outputs]).mean().item())
            wandb.log({'loss_weight_image': self.loss_weights[0].item()})
            wandb.log({'loss_weight_audio': self.loss_weights[1].item()})
            wandb.log({'loss_weight_fusion': self.loss_weights[2].item()})
            wandb.log({'val_loss_image': self.image_criterion_history[-1]})
            wandb.log({'val_loss_audio': self.audio_criterion_history[-1]})
            wandb.log({'val_loss_fusion': self.fusion_criterion_history[-1]})
            self.log('val_loss_fusion', self.fusion_criterion_history[-1])

            if self.current_epoch != 0 and (self.current_epoch % self.update_loss_weights_per_epoch == 0):
                print('[!] Updating loss weights')
                self.loss_weights = self.softadapt.get_component_weights(torch.tensor(self.image_criterion_history),
                                                                         torch.tensor(self.audio_criterion_history),
                                                                         torch.tensor(self.fusion_criterion_history),
                                                                         verbose=True)
                print(f'[!] loss weights: {self.loss_weights}')
                self.image_criterion_history = list()
                self.audio_criterion_history = list()
                self.fusion_criterion_history = list()

    def setup_criterion(self) -> torch.nn.Module:
        return None

    def setup_scores(self) -> List[torch.nn.Module]:
        return setup_scores()

    def test_epoch_end(self, outputs, save_preds=False):
        super().test_epoch_end(outputs, save_preds)
        preds = torch.cat([x['preds'] for x in outputs])
        preds_image = torch.cat([x['preds_image'] for x in outputs])
        preds_audio = torch.cat([x['preds_audio'] for x in outputs])
        labels = torch.cat([x['labels'] for x in outputs])
        image_logits = torch.cat([x['image_logits'] for x in outputs])
        audio_logits = torch.cat([x['audio_logits'] for x in outputs])
        logits = torch.cat([x['logits'] for x in outputs])

        if self.checkpoint_path is None:
            self.checkpoint_path = f'{self.logger.save_dir}/{self.logger.name}/version_{self.logger.version}/checkpoints/'
        save_path = path.dirname(self.checkpoint_path)
        torch.save(dict(preds=preds, preds_image=preds_image, preds_audio=preds_audio, labels=labels,
                        image_logits=image_logits, audio_logits=audio_logits, logits=logits),
                   save_path + '/test_preds.pt')
        print(f'[!] Saved test predictions to {save_path}/test_preds.pt')

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

    def configure_optimizers(self):
        optimizer_cfg = self.optimizer_cfg
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), **optimizer_cfg)
        scheduler = ReduceLROnPlateau(optimizer, patience=self.scheduler_patience, verbose=True)

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }

    def intermediate_step(self, batch: Any) -> Any:
        image = batch['image']
        audio = batch['audio']
        labels = batch['label']

        # get modality encodings from feature extractors
        image_logits = self.image_mixer(image)
        audio_logits = self.audio_mixer(audio)

        fused_moalities = self.fusion_function(image_logits, audio_logits)
        logits = self.fusion_mixer(fused_moalities)

        # fuse modalities
        results = self.shared_step(batch)

        fusion_correct = results['preds'] == labels
        image_correct = results['preds_image'] == labels
        audio_correct = results['preds_audio'] == labels

        return dict(image_logits=image_logits, audio_logits=audio_logits, logits=logits,
                    fusion_correct=fusion_correct, image_correct=image_correct, audio_correct=audio_correct)


class AVMnistMixerMultiLossEvidentialUQ(AVMnistMixerMultiLoss):
    def __init__(self, model_cfg, *args, **kwargs):
        super().__init__(model_cfg, *args, **kwargs)
        self.num_classes = model_cfg.modalities.classification.num_classes
        self.image_criterion = EDLMSELoss(self.num_classes, 10)
        self.audio_criterion = EDLMSELoss(self.num_classes, 10)
        self.fusion_criterion = EDLMSELoss(self.num_classes, 10)

    def shared_step(self, batch, **kwargs):
        image = batch['image']
        audio = batch['audio']
        labels = batch['label']

        if kwargs.get('mode', None) == 'train':
            if self.freeze_modalities_on_epoch is not None and (self.current_epoch == self.freeze_modalities_on_epoch) \
                    and not self.modalities_freezed:
                print('Freezing modalities')
                for param in self.image_mixer.parameters():
                    param.requires_grad = False
                for param in self.audio_mixer.parameters():
                    param.requires_grad = False
                for param in self.classifier_image.parameters():
                    param.requires_grad = False
                for param in self.classifier_audio.parameters():
                    param.requires_grad = False
                self.modalities_freezed = True

            if self.random_modality_muting_on_freeze and (self.current_epoch >= self.freeze_modalities_on_epoch):
                self.mute = np.random.choice(['image', 'audio', 'multimodal'], p=[self.muting_probs['image'],
                                                                                  self.muting_probs['audio'],
                                                                                  self.muting_probs['multimodal']])

            if self.mute != 'multimodal':
                if self.mute == 'image':
                    image = torch.zeros_like(image)
                elif self.mute == 'audio':
                    audio = torch.zeros_like(audio)

        # get modality encodings from feature extractors
        image_logits = self.image_mixer(image)
        audio_logits = self.audio_mixer(audio)

        # fuse modalities
        fused_moalities = self.fusion_function(image_logits, audio_logits)
        logits = self.fusion_mixer(fused_moalities)

        # logits = logits.reshape(logits.shape[0], -1, logits.shape[-1])
        audio_logits = audio_logits.reshape(audio_logits.shape[0], -1, audio_logits.shape[-1])
        image_logits = image_logits.reshape(image_logits.shape[0], -1, image_logits.shape[-1])

        # get logits for each modality
        image_logits = self.classifier_image(image_logits.mean(dim=1))
        audio_logits = self.classifier_audio(audio_logits.mean(dim=1))
        logits = self.classifier_fusion(logits)

        # compute losses
        loss_image = self.image_criterion(image_logits, labels, self.current_epoch)
        loss_audio = self.audio_criterion(audio_logits, labels, self.current_epoch)
        loss_fusion = self.fusion_criterion(logits, labels, self.current_epoch)

        if self.use_softadapt:
            loss = self.loss_weights[0] * loss_image + self.loss_weights[1] * loss_audio + self.loss_weights[
                2] * loss_fusion
        else:
            loss = loss_image + loss_audio + loss_fusion
        if self.modalities_freezed and kwargs.get('mode', None) == 'train':
            loss = loss_fusion

        # get predictions

        evidence = nn.functional.relu(logits)
        evidence_image = nn.functional.relu(image_logits)
        evidence_audio = nn.functional.relu(audio_logits)

        preds = evidence.argmax(dim=1)
        preds_image = evidence_image.argmax(dim=1)
        preds_audio = evidence_audio.argmax(dim=1)

        alpha = evidence + 1
        alpha_image = evidence_image + 1
        alpha_audio = evidence_audio + 1

        uncertainty = self.num_classes / torch.sum(alpha, dim=1, keepdim=True).squeeze(1)
        uncertainty_image = self.num_classes / torch.sum(alpha_image, dim=1, keepdim=True).squeeze(1)
        uncertainty_audio = self.num_classes / torch.sum(alpha_audio, dim=1, keepdim=True).squeeze(1)

        pred_combined = preds * (((uncertainty < uncertainty_image) & (uncertainty < uncertainty_audio))).long() \
                        + preds_image * (
                            ((uncertainty_image < uncertainty) & (uncertainty_image < uncertainty_audio))).long() \
                        + preds_audio * (
                            ((uncertainty_audio < uncertainty) & (uncertainty_audio < uncertainty_image))).long()

        return {
            'preds': pred_combined,
            'preds_image': preds_image,
            'preds_audio': preds_audio,
            'labels': labels,
            'loss': loss,
            'loss_image': loss_image,
            'loss_audio': loss_audio,
            'loss_fusion': loss_fusion,
            'image_logits': image_logits,
            'audio_logits': audio_logits,
            'logits': logits,
            'uncertainty': uncertainty.mean(),
            'uncertainty_image': uncertainty_image.mean(),
            'uncertainty_audio': uncertainty_audio.mean(),
        }

    def training_epoch_end(self, outputs) -> None:
        super().training_epoch_end(outputs)
        wandb.log({'train_uncertainty': torch.stack([x['uncertainty'] for x in outputs]).mean()})
        wandb.log({'train_uncertainty_image': torch.stack([x['uncertainty_image'] for x in outputs]).mean()})
        wandb.log({'train_uncertainty_audio': torch.stack([x['uncertainty_audio'] for x in outputs]).mean()})

    def validation_epoch_end(self, outputs) -> None:
        super().validation_epoch_end(outputs)
        wandb.log({'val_uncertainty': torch.stack([x['uncertainty'] for x in outputs]).mean()})
        wandb.log({'val_uncertainty_image': torch.stack([x['uncertainty_image'] for x in outputs]).mean()})
        wandb.log({'val_uncertainty_audio': torch.stack([x['uncertainty_audio'] for x in outputs]).mean()})

    def test_epoch_end(self, outputs) -> None:
        super().test_epoch_end(outputs)
        wandb.log({'test_uncertainty': torch.stack([x['uncertainty'] for x in outputs]).mean()})
        wandb.log({'test_uncertainty_image': torch.stack([x['uncertainty_image'] for x in outputs]).mean()})
        wandb.log({'test_uncertainty_audio': torch.stack([x['uncertainty_audio'] for x in outputs]).mean()})

    # def configure_optimizers(self):
    #     optimizer_cfg = self.optimizer_cfg
    #     optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), **optimizer_cfg)
    #     scheduler = StepLR(optimizer, step_size=1, gamma=0.1)
    #
    #     return [optimizer], [scheduler]


class AVMnistLateMixer(AbstractTrainTestModule):
    def __init__(self, model_cfg: DictConfig, optimizer_cfg: DictConfig, **kwargs):
        super(AVMnistLateMixer, self).__init__(optimizer_cfg, log_confusion_matrix=True, **kwargs)
        self.optimizer_cfg = optimizer_cfg
        self.mute = model_cfg.get('mute', None)
        image_config = model_cfg.modalities.image
        audio_config = model_cfg.modalities.audio
        dropout = model_cfg.get('dropout', 0.0)
        self.image_mixer = modules.get_block_by_name(**image_config, dropout=dropout)
        self.audio_mixer = modules.get_block_by_name(**audio_config, dropout=dropout)
        self.image_classifier = modules.get_classifier_by_name(**model_cfg.modalities.classification)
        self.audio_classifier = modules.get_classifier_by_name(**model_cfg.modalities.classification)
        self.fusion_function = modules.get_fusion_by_name(**model_cfg.modalities.multimodal)
        self.classifier = modules.get_classifier_by_name(**model_cfg.modalities.multimodal.classification)

        self.criterion = nn.CrossEntropyLoss()
        self.audio_criterion = nn.CrossEntropyLoss()
        self.image_criterion = nn.CrossEntropyLoss()

    def shared_step(self, batch, **kwargs):
        image = batch['image']
        audio = batch['audio']
        labels = batch['label']

        if self.mute == 'image':
            image = torch.zeros_like(image)
        elif self.mute == 'audio':
            audio = torch.zeros_like(audio)

        image_logits = self.image_mixer(image)
        audio_logits = self.audio_mixer(audio)

        image_logits = self.image_classifier(image_logits)
        audio_logits = self.audio_classifier(audio_logits)

        fused_decisions = self.fusion_function(image_logits, audio_logits)
        logits = self.classifier(fused_decisions)
        loss_fusion = self.criterion(logits, labels)
        loss_image = self.image_criterion(image_logits, labels)
        loss_audio = self.audio_criterion(audio_logits, labels)

        loss = loss_fusion + loss_image + loss_audio
        preds = torch.softmax(logits, dim=1).argmax(dim=1)

        return {
            'preds': preds,
            'labels': labels,
            'loss': loss
        }

    def configure_optimizers(self):
        optimizer_cfg = self.optimizer_cfg
        scheduler_patience = optimizer_cfg.pop('scheduler_patience', 2)
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), **optimizer_cfg)
        scheduler = ReduceLROnPlateau(optimizer, patience=scheduler_patience, verbose=True)

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }

    def setup_criterion(self) -> torch.nn.Module:
        return None

    def setup_scores(self) -> List[torch.nn.Module]:
        return setup_scores()


class AVMnistMixerMultiLossHomoscedasticUQ(AbstractTrainTestModule):
    def __init__(self, model_cfg: DictConfig, optimizer_cfg: DictConfig, **kwargs):
        super(AVMnistMixerMultiLossHomoscedasticUQ, self).__init__(optimizer_cfg, log_confusion_matrix=True, **kwargs)
        self.optimizer_cfg = optimizer_cfg
        self.scheduler_patience = optimizer_cfg.pop('scheduler_patience', 5)
        self.checkpoint_path = None
        self.mute = model_cfg.get('mute', None)
        image_config = model_cfg.modalities.image
        audio_config = model_cfg.modalities.audio
        multimodal_config = model_cfg.modalities.multimodal
        dropout = model_cfg.get('dropout', 0.0)
        self.image_mixer = modules.get_block_by_name(**image_config, dropout=dropout)
        self.audio_mixer = modules.get_block_by_name(**audio_config, dropout=dropout)
        self.fusion_function = modules.get_fusion_by_name(**model_cfg.modalities.multimodal)
        num_patches = self.fusion_function.get_output_shape(self.image_mixer.num_patch, self.audio_mixer.num_patch,
                                                            dim=1)
        self.fusion_mixer = modules.get_block_by_name(**multimodal_config, num_patches=num_patches, dropout=dropout)
        self.classifier_image = torch.nn.Linear(model_cfg.modalities.image.hidden_dim,
                                                model_cfg.modalities.classification.num_classes)
        self.classifier_audio = torch.nn.Linear(model_cfg.modalities.audio.hidden_dim,
                                                model_cfg.modalities.classification.num_classes)
        self.classifier_fusion = modules.get_classifier_by_name(**model_cfg.modalities.classification)

        self.image_criterion = CrossEntropyLoss()
        self.audio_criterion = CrossEntropyLoss()
        self.fusion_criterion = CrossEntropyLoss()

        self.log_var_image = nn.Parameter(torch.zeros((1,), requires_grad=True))
        self.log_var_audio = nn.Parameter(torch.zeros((1,), requires_grad=True))
        self.log_var_fusion = nn.Parameter(torch.zeros((1,), requires_grad=True))

    def shared_step(self, batch, **kwargs):
        # Load data

        image = batch['image']
        audio = batch['audio']
        labels = batch['label']

        if kwargs.get('mode', None) == 'train' and self.mute != 'multimodal':
            if self.mute == 'image':
                image = torch.zeros_like(image)
            elif self.mute == 'audio':
                audio = torch.zeros_like(audio)

        # get modality encodings from feature extractors
        image_logits = self.image_mixer(image)
        audio_logits = self.audio_mixer(audio)

        # fuse modalities
        fused_moalities = self.fusion_function(image_logits, audio_logits)
        logits = self.fusion_mixer(fused_moalities)

        # logits = logits.reshape(logits.shape[0], -1, logits.shape[-1])
        audio_logits = audio_logits.reshape(audio_logits.shape[0], -1, audio_logits.shape[-1])
        image_logits = image_logits.reshape(image_logits.shape[0], -1, image_logits.shape[-1])

        # get logits for each modality
        image_logits = self.classifier_image(image_logits.mean(dim=1))
        audio_logits = self.classifier_audio(audio_logits.mean(dim=1))
        logits = self.classifier_fusion(logits)

        # compute losses
        loss_image = torch.exp(-self.log_var_image) * self.image_criterion(image_logits, labels)
        loss_audio = torch.exp(-self.log_var_audio) * self.audio_criterion(audio_logits, labels)
        loss_fusion = torch.exp(-self.log_var_fusion) * self.fusion_criterion(logits, labels)
        loss = loss_image + loss_audio + loss_fusion + self.log_var_image + self.log_var_audio + self.log_var_fusion

        # print('=' * 20)
        # print(f'loss_image: {loss_image.item()}, loss_audio: {loss_audio.item()}, loss_fusion: {loss_fusion.item()}')
        # print(f'log_var_image: {self.log_var_image.item()}, log_var_audio: {self.log_var_audio.item()}, '
        #         f'log_var_fusion: {self.log_var_fusion.item()}')
        # print(f'exp_log_var_image: {torch.exp(-self.log_var_image).item()}, exp_log_var_audio: {torch.exp(-self.log_var_audio).item()} '
        #         f'exp_log_var_fusion: {torch.exp(-self.log_var_fusion).item()}')
        # print(f'loss: {loss.item()}')
        # print('=' * 20)

        # get predictions
        preds = torch.softmax(logits, dim=1).argmax(dim=1)

        return {
            'preds': preds,
            'labels': labels,
            'loss': loss,
            'logits': logits,
        }

    def training_epoch_end(self, outputs) -> None:
        super().training_epoch_end(outputs)
        wandb.log({'std_image': (torch.exp(self.log_var_image) ** 0.5).item(),
                   'std_audio': (torch.exp(self.log_var_audio) ** 0.5).item(),
                   'std_fusion': (torch.exp(self.log_var_fusion) ** 0.5).item()})

    def setup_criterion(self) -> torch.nn.Module:
        return None

    def setup_scores(self) -> List[torch.nn.Module]:
        test_scores, train_scores, val_scores = setup_scores()

        return [train_scores, val_scores, test_scores]

    def test_epoch_end(self, outputs, save_preds=False):
        super().test_epoch_end(outputs, save_preds)
        preds = torch.cat([x['preds'] for x in outputs])
        labels = torch.cat([x['labels'] for x in outputs])
        logits = torch.cat([x['logits'] for x in outputs])
        wandb.summary['test_std_image'] = (torch.exp(self.log_var_image) ** 0.5).item()
        wandb.summary['test_std_audio'] = (torch.exp(self.log_var_audio) ** 0.5).item()
        wandb.summary['test_std_fusion'] = (torch.exp(self.log_var_fusion) ** 0.5).item()

        if self.checkpoint_path is None:
            self.checkpoint_path = f'{self.logger.save_dir}/{self.logger.name}/version_{self.logger.version}/checkpoints/'
        save_path = path.dirname(self.checkpoint_path)
        torch.save(
            dict(
                preds=preds,
                labels=labels,
                logits=logits,
            ),
            f'{save_path}/test_preds.pt',
        )
        print(f'[!] Saved test predictions to {save_path}/test_preds.pt')

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

    def configure_optimizers(self):
        optimizer_cfg = self.optimizer_cfg
        optimizer = torch.optim.Adam(self.parameters(), **optimizer_cfg)
        scheduler = ReduceLROnPlateau(optimizer, patience=self.scheduler_patience, verbose=True)

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }

    def intermediate_step(self, batch: Any) -> Any:
        image = batch['image']
        audio = batch['audio']
        labels = batch['label']

        # get modality encodings from feature extractors
        image_logits = self.image_mixer(image)
        audio_logits = self.audio_mixer(audio)

        fused_moalities = self.fusion_function(image_logits, audio_logits)
        logits = self.fusion_mixer(fused_moalities)

        # fuse modalities
        results = self.shared_step(batch)

        fusion_correct = results['preds'] == labels
        image_correct = results['preds_image'] == labels
        audio_correct = results['preds_audio'] == labels

        return dict(image_logits=image_logits, audio_logits=audio_logits, logits=logits,
                    fusion_correct=fusion_correct, image_correct=image_correct, audio_correct=audio_correct)

