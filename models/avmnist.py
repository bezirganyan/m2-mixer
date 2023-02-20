from abc import ABC, abstractmethod
from os import path

import wandb
from omegaconf import DictConfig
from softadapt import LossWeightedSoftAdapt, NormalizedSoftAdapt

from modules.train_test_module import AbstractTrainTestModule
from modules.fusion import BiModalGatedUnit
from modules.gmpl import VisiongMLP, FusiongMLP
from modules.mixer import MLPool, MLPMixer, FusionMixer

import torch

from typing import List, Any, Optional
from torch.nn import CrossEntropyLoss
from torchmetrics import Accuracy
import modules


class AVMnistImagePooler(AbstractTrainTestModule):
    def __init__(self, model_cfg: DictConfig, optimizer_cfg: DictConfig, **kwargs):
        super(AVMnistImagePooler, self).__init__(optimizer_cfg, **kwargs)
        self.optimizer_cfg = optimizer_cfg
        self.model = MLPool(**model_cfg.modalities.image)
        self.classifier = torch.nn.Linear(model_cfg.modalities.image.hidden_dims[-1],
                                          model_cfg.modalities.classification.num_classes)

    def shared_step(self, batch):
        image = batch['image']
        labels = batch['label']
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
        train_scores = dict(acc=Accuracy(task="multiclass", num_classes=10))
        val_scores = dict(acc=Accuracy(task="multiclass", num_classes=10))
        test_scores = dict(acc=Accuracy(task="multiclass", num_classes=10))

        return [train_scores, val_scores, test_scores]


class AbstractAVMnistMixer(AbstractTrainTestModule, ABC):
    def __init__(self, model_cfg: DictConfig, optimizer_cfg: DictConfig, **kwargs):
        super(AbstractAVMnistMixer, self).__init__(optimizer_cfg, **kwargs)
        self.optimizer_cfg = optimizer_cfg
        self.model = None
        self.classifier = None

    def shared_step(self, batch):
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
        train_scores = dict(acc=Accuracy(task="multiclass", num_classes=10))
        val_scores = dict(acc=Accuracy(task="multiclass", num_classes=10))
        test_scores = dict(acc=Accuracy(task="multiclass", num_classes=10))

        return [train_scores, val_scores, test_scores]


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
        self.model = MLPMixer(**model_cfg.modalities.image, dropout=model_cfg.dropout)
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
        image_config = model_cfg.modalities.image
        audio_config = model_cfg.modalities.audio
        multimodal_config = model_cfg.modalities.multimodal
        dropout = model_cfg.get('dropout', 0.0)
        self.image_mixer = MLPMixer(**image_config, dropout=dropout)
        self.audio_mixer = MLPMixer(**audio_config, dropout=dropout)
        # num_patches = self.image_mixer.num_patch + self.audio_mixer.num_patch
        num_patches = self.image_mixer.num_patch
        self.fusion_mixer = FusionMixer(**multimodal_config, num_patches=num_patches, dropout=dropout)
        self.classifier = torch.nn.Linear(model_cfg.modalities.image.hidden_dim,
                                          model_cfg.modalities.classification.num_classes)

    def get_logits(self, batch):
        image = batch['image']
        audio = batch['audio']

        if self.mute == 'image':
            image = torch.zeros_like(image)
        elif self.mute == 'audio':
            audio = torch.zeros_like(audio)

        image_logits = self.image_mixer(image)
        audio_logits = self.audio_mixer(audio)
        # logits = self.fusion_mixer(torch.cat([image_logits, audio_logits], dim=1))
        logits = self.fusion_mixer(torch.maximum(image_logits, audio_logits))
        logits = self.classifier(logits.mean(dim=1))
        return logits


class AVMnistMixerLF(AbstractAVMnistMixer):
    def __init__(self, model_cfg: DictConfig, optimizer_cfg: DictConfig, **kwargs):
        super(AVMnistMixerLF, self).__init__(model_cfg, optimizer_cfg, **kwargs)
        self.optimizer_cfg = optimizer_cfg
        self.mute = model_cfg.get('mute', None)
        image_config = model_cfg.modalities.image
        audio_config = model_cfg.modalities.audio
        multimodal_config = model_cfg.modalities.multimodal
        dropout = model_cfg.get('dropout', 0.0)
        self.image_mixer = MLPMixer(**image_config, dropout=dropout)
        self.audio_mixer = MLPMixer(**audio_config, dropout=dropout)
        # num_patches = self.image_mixer.num_patch + self.audio_mixer.num_patch
        num_patches = self.image_mixer.num_patch
        # self.fusion_mixer = FusionMixer(**multimodal_config, num_patches=num_patches, dropout=dropout)
        self.classifier_image = torch.nn.Linear(model_cfg.modalities.image.hidden_dim,
                                                model_cfg.modalities.classification.num_classes)
        self.classifier_audio = torch.nn.Linear(model_cfg.modalities.audio.hidden_dim,
                                                model_cfg.modalities.classification.num_classes)
        self.classifier_fusion = torch.nn.Linear(model_cfg.modalities.classification.num_classes * 2,
                                                 model_cfg.modalities.classification.num_classes)

    def get_logits(self, batch):
        image = batch['image']
        audio = batch['audio']

        if self.mute == 'image':
            image = torch.zeros_like(image)
        elif self.mute == 'audio':
            audio = torch.zeros_like(audio)

        image_logits = self.image_mixer(image)
        audio_logits = self.audio_mixer(audio)
        # logits = self.fusion_mixer(torch.cat([image_logits, audio_logits], dim=1))
        # logits = self.fusion_mixer(torch.maximum(image_logits, audio_logits))
        image_logits = torch.softmax(self.classifier_image(image_logits.mean(dim=1)), dim=1)
        audio_logits = torch.softmax(self.classifier_audio(audio_logits.mean(dim=1)), dim=1)
        logits = self.classifier_fusion(torch.cat([image_logits, audio_logits], dim=1))
        # logits = self.classifier(logits.mean(dim=1))
        return logits


class AVMnistMixerMultiLoss(AbstractTrainTestModule):
    def __init__(self, model_cfg: DictConfig, optimizer_cfg: DictConfig, **kwargs):
        super(AVMnistMixerMultiLoss, self).__init__(optimizer_cfg, log_confusion_matrix=True, **kwargs)
        self.optimizer_cfg = optimizer_cfg
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
        self.classifier_fusion = torch.nn.Linear(model_cfg.modalities.image.hidden_dim,
                                                 model_cfg.modalities.classification.num_classes)

        self.image_criterion = CrossEntropyLoss()
        self.audio_criterion = CrossEntropyLoss()
        self.fusion_criterion = CrossEntropyLoss()

        self.use_softadapt = model_cfg.get('use_softadapt', False)
        if self.use_softadapt:
            self.image_criterion_history = list()
            self.audio_criterion_history = list()
            self.fusion_criterion_history = list()
            self.loss_weights = torch.tensor([1.0 / 3, 1.0 / 3, 1.0 / 3], device=self.device)
            self.softadapt = NormalizedSoftAdapt(beta=0.1, accuracy_order=5)
            self.update_loss_weights_per_epoch = model_cfg.get('update_loss_weights_per_epoch', 10)

    def shared_step(self, batch):
        # Load data

        image = batch['image']
        audio = batch['audio']
        labels = batch['label']

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

        logits = logits.reshape(logits.shape[0], -1, logits.shape[-1])
        audio_logits = audio_logits.reshape(audio_logits.shape[0], -1, audio_logits.shape[-1])
        image_logits = image_logits.reshape(image_logits.shape[0], -1, image_logits.shape[-1])

        # get logits for each modality
        image_logits = self.classifier_image(image_logits.mean(dim=1))
        audio_logits = self.classifier_audio(audio_logits.mean(dim=1))
        logits = self.classifier_fusion(logits.mean(dim=1))

        # compute losses
        loss_image = self.image_criterion(image_logits, labels)
        loss_audio = self.audio_criterion(audio_logits, labels)
        loss_fusion = self.fusion_criterion(logits, labels)

        if self.use_softadapt:
            loss = self.loss_weights[0] * loss_image + self.loss_weights[1] * loss_audio + self.loss_weights[
                2] * loss_fusion


        else:
            loss = loss_image + loss_audio + loss_fusion

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
            'loss_fusion': loss_fusion
        }

    def training_epoch_end(self, outputs) -> None:
        super().training_epoch_end(outputs)
        if self.use_softadapt:
            self.image_criterion_history.append(torch.stack([x['loss_image'] for x in outputs]).mean().item())
            self.audio_criterion_history.append(torch.stack([x['loss_audio'] for x in outputs]).mean().item())
            self.fusion_criterion_history.append(torch.stack([x['loss_fusion'] for x in outputs]).mean().item())
            wandb.log({'loss_weight_image': self.loss_weights[0].item()})
            wandb.log({'loss_weight_audio': self.loss_weights[1].item()})
            wandb.log({'loss_weight_fusion': self.loss_weights[2].item()})
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
        train_scores = dict(acc=Accuracy(task="multiclass", num_classes=10))
        val_scores = dict(acc=Accuracy(task="multiclass", num_classes=10))
        test_scores = dict(acc=Accuracy(task="multiclass", num_classes=10))

        return [train_scores, val_scores, test_scores]

    def test_epoch_end(self, outputs, save_preds=False):
        super().test_epoch_end(outputs, save_preds)
        preds = torch.cat([x['preds'] for x in outputs])
        preds_image = torch.cat([x['preds_image'] for x in outputs])
        preds_audio = torch.cat([x['preds_audio'] for x in outputs])
        labels = torch.cat([x['labels'] for x in outputs])

        save_path = path.dirname(self.checkpoint_path)
        torch.save(dict(preds=preds, preds_image=preds_image, preds_audio=preds_audio, labels=labels),
                   save_path + '/test_preds.pt')

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
        cls.checkpoint_path = checkpoint_path
        return model


class AVMnistMixerMultiLossGated(AbstractTrainTestModule):
    def __init__(self, model_cfg: DictConfig, optimizer_cfg: DictConfig, **kwargs):
        super(AVMnistMixerMultiLossGated, self).__init__(optimizer_cfg, log_confusion_matrix=True, **kwargs)
        self.optimizer_cfg = optimizer_cfg
        self.mute = model_cfg.get('mute', None)
        image_config = model_cfg.modalities.image
        audio_config = model_cfg.modalities.audio
        multimodal_config = model_cfg.modalities.multimodal
        dropout = model_cfg.get('dropout', 0.0)
        self.image_mixer = MLPMixer(**image_config, dropout=dropout)
        self.audio_mixer = MLPMixer(**audio_config, dropout=dropout)
        # num_patches = self.image_mixer.num_patch + self.audio_mixer.num_patch
        num_patches = self.image_mixer.num_patch
        self.fusion_mixer = FusionMixer(**multimodal_config, num_patches=num_patches, dropout=dropout)
        self.classifier_image = torch.nn.Linear(model_cfg.modalities.image.hidden_dim,
                                                model_cfg.modalities.classification.num_classes)
        self.classifier_audio = torch.nn.Linear(model_cfg.modalities.audio.hidden_dim,
                                                model_cfg.modalities.classification.num_classes)
        self.classifier_fusion = torch.nn.Linear(model_cfg.modalities.image.hidden_dim,
                                                 model_cfg.modalities.classification.num_classes)
        self.fusion_function = BiModalGatedUnit(model_cfg.modalities.image.hidden_dim,
                                                model_cfg.modalities.audio.hidden_dim,
                                                model_cfg.modalities.multimodal.hidden_dim)

        self.image_criterion = CrossEntropyLoss()
        self.audio_criterion = CrossEntropyLoss()
        self.fusion_criterion = CrossEntropyLoss()

    def shared_step(self, batch):
        image = batch['image']
        audio = batch['audio']
        labels = batch['label']

        if self.mute == 'image':
            image = torch.zeros_like(image)
        elif self.mute == 'audio':
            audio = torch.zeros_like(audio)

        image_logits = self.image_mixer(image)
        audio_logits = self.audio_mixer(audio)
        concat_logits = self.fusion_function(image_logits, audio_logits)
        logits = self.fusion_mixer(concat_logits)
        image_logits = self.classifier_image(image_logits.mean(dim=1))
        audio_logits = self.classifier_audio(audio_logits.mean(dim=1))
        logits = self.classifier_fusion(logits.mean(dim=1))

        # concat_logits = torch.cat([image_logits, audio_logits, logits], dim=1)

        # Losses
        loss_image = self.image_criterion(image_logits, labels)
        loss_audio = self.audio_criterion(audio_logits, labels)
        loss_fusion = self.fusion_criterion(logits, labels)

        loss = loss_image + loss_audio + loss_fusion
        preds = torch.softmax(logits, dim=1).argmax(dim=1)

        return {
            'preds': preds,
            'labels': labels,
            'loss': loss
        }

    def setup_criterion(self) -> torch.nn.Module:
        return None

    def setup_scores(self) -> List[torch.nn.Module]:
        train_scores = dict(acc=Accuracy(task="multiclass", num_classes=10))
        val_scores = dict(acc=Accuracy(task="multiclass", num_classes=10))
        test_scores = dict(acc=Accuracy(task="multiclass", num_classes=10))

        return [train_scores, val_scores, test_scores]


class AVMnistgMLP(AbstractAVMnistMixer):
    def __init__(self, model_cfg: DictConfig, optimizer_cfg: DictConfig, **kwargs):
        super(AVMnistgMLP, self).__init__(model_cfg, optimizer_cfg, **kwargs)
        image_config = model_cfg.modalities.image
        audio_config = model_cfg.modalities.audio
        self.mute = model_cfg.get('mute', None)
        multimodal_config = model_cfg.modalities.multimodal
        dropout = model_cfg.get('dropout', 0.0)
        self.image_mixer = VisiongMLP(**image_config, dropout=dropout)
        self.audio_mixer = VisiongMLP(**audio_config, dropout=dropout)
        num_patches = self.image_mixer.num_patch
        self.fusion_mixer = FusiongMLP(**multimodal_config, num_patches=num_patches, dropout=dropout)
        self.classifier = torch.nn.Linear(model_cfg.modalities.image.d_model,
                                          model_cfg.modalities.classification.num_classes)

    def get_logits(self, batch):
        image = batch['image']
        audio = batch['audio']

        if self.mute == 'image':
            image = torch.zeros_like(image)
        elif self.mute == 'audio':
            audio = torch.zeros_like(audio)

        image_logits = self.image_mixer(image)
        audio_logits = self.audio_mixer(audio)
        # logits = self.fusion_mixer(torch.cat([image_logits, audio_logits], dim=1))
        logits = self.fusion_mixer(torch.maximum(image_logits, audio_logits))
        logits = self.classifier(logits[:, 0])
        return logits
