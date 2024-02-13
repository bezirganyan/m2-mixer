from os import path
from typing import Any, List, Optional

import numpy as np
import torch
import wandb
from omegaconf import DictConfig
from torch import nn
from torch.nn import BCEWithLogitsLoss
from torchmetrics import F1Score

import modules
from modules.train_test_module import AbstractTrainTestModule

try:
    from softadapt import LossWeightedSoftAdapt
except ModuleNotFoundError:
    print('Warning: Could not import softadapt. LossWeightedSoftAdapt will not be available.')
    LossWeightedSoftAdapt = None


class MMIMDBMixerMultiLoss(AbstractTrainTestModule):
    def __init__(self, model_cfg: DictConfig, optimizer_cfg: DictConfig, **kwargs):
        super(MMIMDBMixerMultiLoss, self).__init__(optimizer_cfg, log_confusion_matrix=False, **kwargs)
        self.modalities_freezed = False
        self.optimizer_cfg = optimizer_cfg
        self.checkpoint_path = None
        self.mute = model_cfg.get('mute', None)
        self.freeze_modalities_on_epoch = model_cfg.get('freeze_modalities_on_epoch', None)
        self.random_modality_muting_on_freeze = model_cfg.get('random_modality_muting_on_freeze', False)
        self.muting_probs = model_cfg.get('muting_probs', None)
        image_config = model_cfg.modalities.image
        text_config = model_cfg.modalities.text
        multimodal_config = model_cfg.modalities.multimodal
        dropout = model_cfg.get('dropout', 0.0)
        self.image_mixer = modules.get_block_by_name(**image_config, dropout=dropout)
        self.text_mixer = modules.get_block_by_name(**text_config, dropout=dropout)
        self.fusion_function = modules.get_fusion_by_name(**model_cfg.modalities.multimodal)
        num_patches = self.fusion_function.get_output_shape(self.image_mixer.num_patch, self.text_mixer.num_patch,
                                                            dim=1)
        self.fusion_mixer = modules.get_block_by_name(**multimodal_config, num_patches=num_patches, dropout=dropout)
        self.classifier_image = torch.nn.Linear(model_cfg.modalities.image.hidden_dim,
                                                model_cfg.modalities.classification.num_classes)
        self.classifier_text = torch.nn.Linear(model_cfg.modalities.text.hidden_dim,
                                               model_cfg.modalities.classification.num_classes)
        self.classifier_fusion = modules.get_classifier_by_name(**model_cfg.modalities.classification)

        pos_weight = torch.tensor(model_cfg.pos_weight)
        self.image_criterion = BCEWithLogitsLoss(pos_weight=pos_weight)
        self.text_criterion = BCEWithLogitsLoss(pos_weight=pos_weight)
        self.fusion_criterion = BCEWithLogitsLoss(pos_weight=pos_weight)

        self.use_softadapt = model_cfg.get('use_softadapt', False)
        if self.use_softadapt:
            if LossWeightedSoftAdapt is None:
                self.use_softadapt = False
                print('SoftAdapt is not installed! Hence, will not be used!')
            else:
                self.image_criterion_history = []
                self.text_criterion_history = []
                self.fusion_criterion_history = []
                self.loss_weights = torch.tensor([1.0 / 3, 1.0 / 3, 1.0 / 3], device=self.device)
                self.update_loss_weights_per_epoch = model_cfg.get('update_loss_weights_per_epoch', 6)
                self.softadapt = LossWeightedSoftAdapt(beta=-0.1, accuracy_order=self.update_loss_weights_per_epoch - 1)

    def shared_step(self, batch, **kwargs):
        # Load data

        image = batch['image']
        text = batch['text']
        labels = batch['label']

        if kwargs.get('mode', None) == 'train':
            if self.freeze_modalities_on_epoch is not None and (self.current_epoch == self.freeze_modalities_on_epoch) \
                    and not self.modalities_freezed:
                print('Freezing modalities')
                for param in self.image_mixer.parameters():
                    param.requires_grad = False
                for param in self.text_mixer.parameters():
                    param.requires_grad = False
                for param in self.classifier_image.parameters():
                    param.requires_grad = False
                for param in self.classifier_text.parameters():
                    param.requires_grad = False
                self.modalities_freezed = True

            if self.random_modality_muting_on_freeze and (self.current_epoch >= self.freeze_modalities_on_epoch):
                self.mute = np.random.choice(['image', 'text', 'multimodal'], p=[self.muting_probs['image'],
                                                                                 self.muting_probs['text'],
                                                                                 self.muting_probs['multimodal']])

            if self.mute != 'multimodal':
                if self.mute == 'image':
                    image = torch.zeros_like(image)
                elif self.mute == 'text':
                    text = torch.zeros_like(text)

        # get modality encodings from feature extractors
        image_logits = self.image_mixer(image)
        text_logits = self.text_mixer(text)

        # fuse modalities
        fused_moalities = self.fusion_function(image_logits, text_logits)
        logits = self.fusion_mixer(fused_moalities)

        # logits = logits.reshape(logits.shape[0], -1, logits.shape[-1])
        text_logits = text_logits.reshape(text_logits.shape[0], -1, text_logits.shape[-1])
        image_logits = image_logits.reshape(image_logits.shape[0], -1, image_logits.shape[-1])

        # get logits for each modality
        image_logits = self.classifier_image(image_logits.mean(dim=1))
        text_logits = self.classifier_text(text_logits.mean(dim=1))
        logits = self.classifier_fusion(logits)

        # compute losses
        loss_image = self.image_criterion(image_logits, labels.float())
        loss_text = self.text_criterion(text_logits, labels.float())
        loss_fusion = self.fusion_criterion(logits, labels.float())

        if self.use_softadapt:
            loss = self.loss_weights[0] * loss_image + self.loss_weights[1] * loss_text + self.loss_weights[
                2] * loss_fusion
        else:
            loss = loss_image + loss_text + loss_fusion
        if self.modalities_freezed and kwargs.get('mode', None) == 'train':
            loss = loss_fusion

        # get predictions
        preds = torch.sigmoid(logits) > 0.5
        preds = preds.long()
        preds_image = torch.sigmoid(image_logits) > 0.5
        preds_image = preds_image.long()
        preds_text = torch.sigmoid(text_logits) > 0.5
        preds_text = preds_text.long()

        return {
            'preds': preds,
            'preds_image': preds_image,
            'preds_text': preds_text,
            'labels': labels,
            'loss': loss,
            'loss_image': loss_image,
            'loss_text': loss_text,
            'loss_fusion': loss_fusion,
            'image_logits': image_logits,
            'text_logits': text_logits,
            'logits': logits
        }

    def on_train_epoch_end(self) -> None:
        super().log_training_metrics()
        wandb.log(
            {'train_loss_image': torch.stack([x['loss_image'] for x in self.training_step_outputs]).mean().item()})
        wandb.log({'train_loss_text': torch.stack([x['loss_text'] for x in self.training_step_outputs]).mean().item()})
        wandb.log(
            {'train_loss_fusion': torch.stack([x['loss_fusion'] for x in self.training_step_outputs]).mean().item()})
        self.log('train_loss_fusion', torch.stack([x['loss_fusion'] for x in self.training_step_outputs]).mean().item())
        self.training_step_outputs.clear()

    def on_validation_epoch_end(self) -> None:
        super().log_validation_metrics()
        if self.use_softadapt:
            self.image_criterion_history.append(
                torch.stack([x['loss_image'] for x in self.validation_step_outputs]).mean().item())
            self.text_criterion_history.append(
                torch.stack([x['loss_text'] for x in self.validation_step_outputs]).mean().item())
            self.fusion_criterion_history.append(
                torch.stack([x['loss_fusion'] for x in self.validation_step_outputs]).mean().item())
            wandb.log({'loss_weight_image': self.loss_weights[0].item()})
            wandb.log({'loss_weight_text': self.loss_weights[1].item()})
            wandb.log({'loss_weight_fusion': self.loss_weights[2].item()})
            wandb.log({'val_loss_image': self.image_criterion_history[-1]})
            wandb.log({'val_loss_text': self.text_criterion_history[-1]})
            wandb.log({'val_loss_fusion': self.fusion_criterion_history[-1]})
            self.log('val_loss_fusion', self.fusion_criterion_history[-1])

            if self.current_epoch != 0 and (self.current_epoch % self.update_loss_weights_per_epoch == 0):
                print('[!] Updating loss weights')
                self.loss_weights = self.softadapt.get_component_weights(torch.tensor(self.image_criterion_history),
                                                                         torch.tensor(self.text_criterion_history),
                                                                         torch.tensor(self.fusion_criterion_history),
                                                                         verbose=True)
                print(f'[!] loss weights: {self.loss_weights}')
                self.image_criterion_history = list()
                self.text_criterion_history = list()
                self.fusion_criterion_history = list()
        self.validation_step_outputs.clear()

    def setup_criterion(self) -> torch.nn.Module:
        return None

    def setup_scores(self) -> List[torch.nn.Module]:
        train_scores = dict(f1w=F1Score(task="multilabel", num_labels=23, average='weighted'),
                            f1m=F1Score(task="multilabel", num_labels=23, average='macro'))
        val_scores = dict(f1w=F1Score(task="multilabel", num_labels=23, average='weighted'),
                          f1m=F1Score(task="multilabel", num_labels=23, average='macro'))
        test_scores = dict(f1w=F1Score(task="multilabel", num_labels=23, average='weighted'),
                           f1m=F1Score(task="multilabel", num_labels=23, average='macro'))

        return [train_scores, val_scores, test_scores]

    def on_test_epoch_end(self, save_preds=False):
        super().test_epoch_end(self.test_step_outputs, save_preds)
        preds = torch.cat([x['preds'] for x in self.test_step_outputs])
        preds_image = torch.cat([x['preds_image'] for x in self.test_step_outputs])
        preds_text = torch.cat([x['preds_text'] for x in self.test_step_outputs])
        labels = torch.cat([x['labels'] for x in self.test_step_outputs])
        image_logits = torch.cat([x['image_logits'] for x in self.test_step_outputs])
        text_logits = torch.cat([x['text_logits'] for x in self.test_step_outputs])
        logits = torch.cat([x['logits'] for x in self.test_step_outputs])

        if self.checkpoint_path is None:
            self.checkpoint_path = f'{self.logger.save_dir}/{self.logger.name}/version_{self.logger.version}/checkpoints/'
        save_path = path.dirname(self.checkpoint_path)
        torch.save(dict(preds=preds, preds_image=preds_image, preds_text=preds_text, labels=labels,
                        image_logits=image_logits, text_logits=text_logits, logits=logits),
                   save_path + '/test_preds.pt')
        print(f'[!] Saved test predictions to {save_path}/test_preds.pt')
        self.test_step_outputs.clear()

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
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
