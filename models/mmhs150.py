from os import path
from typing import Any, List, Optional

import numpy as np
import torch
import wandb
from omegaconf import DictConfig
from torch import nn
from torch.nn import BCEWithLogitsLoss
from torchmetrics import AUROC, Accuracy, F1Score, Precision, Recall

import modules
from modules.train_test_module import AbstractTrainTestModule

try:
    from softadapt import LossWeightedSoftAdapt
except ModuleNotFoundError:
    print('Warning: Could not import softadapt. LossWeightedSoftAdapt will not be available.')
    LossWeightedSoftAdapt = None

class MMHS150MultiLoss(AbstractTrainTestModule):
    def __init__(self, model_cfg: DictConfig, optimizer_cfg: DictConfig, **kwargs):
        self.num_classes = model_cfg.modalities.classification.get('num_classes', 3)
        super(MMHS150MultiLoss, self).__init__(optimizer_cfg, log_confusion_matrix=False, **kwargs)
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
        self.text_ocr_mixer = modules.get_block_by_name(**text_config, dropout=dropout)
        self.fusion_function = modules.get_fusion_by_name(**model_cfg.modalities.multimodal)
        num_patches = self.fusion_function.get_output_shape(self.image_mixer.num_patch, self.text_mixer.num_patch,
                                                            self.text_ocr_mixer.num_patch, dim=1)
        self.fusion_mixer = modules.get_block_by_name(**multimodal_config, num_patches=num_patches, dropout=dropout)
        self.classifier_image = torch.nn.Linear(model_cfg.modalities.image.hidden_dim,
                                                model_cfg.modalities.classification.num_classes)
        self.classifier_text = torch.nn.Linear(model_cfg.modalities.text.hidden_dim,
                                               model_cfg.modalities.classification.num_classes)
        self.classifier_text_ocr = torch.nn.Linear(model_cfg.modalities.text.hidden_dim,
                                                    model_cfg.modalities.classification.num_classes)
        self.classifier_fusion = modules.get_classifier_by_name(**model_cfg.modalities.classification)

        self.image_criterion = BCEWithLogitsLoss(pos_weight=torch.tensor([3.57]))
        self.text_criterion = BCEWithLogitsLoss(pos_weight=torch.tensor([3.57]))
        self.text_ocr_criterion = BCEWithLogitsLoss(pos_weight=torch.tensor([3.57]))
        self.fusion_criterion = BCEWithLogitsLoss(pos_weight=torch.tensor([3.57]))
        self.fusion_loss_weight = model_cfg.get('fusion_loss_weight', 1.0 / 4)
        self.fusion_loss_change = model_cfg.get('fusion_loss_change', 0)
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
        text_ocr = batch['ocr']
        labels = batch['label']

        # get modality encodings from feature extractors
        image_logits = self.image_mixer(image)
        text_logits = self.text_mixer(text)
        text_ocr_logits = self.text_ocr_mixer(text_ocr)

        # fuse modalities
        fused_moalities = self.fusion_function(image_logits, text_logits, text_ocr_logits)
        logits = self.fusion_mixer(fused_moalities)

        # logits = logits.reshape(logits.shape[0], -1, logits.shape[-1])
        text_logits = text_logits.reshape(text_logits.shape[0], -1, text_logits.shape[-1])
        image_logits = image_logits.reshape(image_logits.shape[0], -1, image_logits.shape[-1])
        text_ocr_logits = text_ocr_logits.reshape(text_ocr_logits.shape[0], -1, text_ocr_logits.shape[-1])

        # get logits for each modality
        image_logits = self.classifier_image(image_logits.mean(dim=1))
        text_logits = self.classifier_text(text_logits.mean(dim=1))
        text_ocr_logits = self.classifier_text_ocr(text_ocr_logits.mean(dim=1))
        logits = self.classifier_fusion(logits)

        # compute losses
        loss_image = self.image_criterion(image_logits, labels.unsqueeze(1).float())
        loss_text = self.text_criterion(text_logits * batch['use_features'],
                                        labels.unsqueeze(1).float() * batch['use_features'])
        loss_text_ocr = self.text_ocr_criterion(text_ocr_logits * batch['use_features_ocr'],
                                                labels.unsqueeze(1).float() * batch['use_features_ocr'])
        loss_fusion = self.fusion_criterion(logits, labels.unsqueeze(1).float())

        ow = (1 - self.fusion_loss_weight) / 3
        loss = self.fusion_loss_weight * loss_fusion + ow * loss_image + ow * loss_text + ow * loss_text_ocr

        # get predictions
        preds = (torch.sigmoid(logits) > 0.5).long()
        preds = torch.tensor(np.random.choice([0, 1], size=preds.shape, p=[0.5, 0.5])).long()
        preds_image = (torch.sigmoid(image_logits) > 0.5).long()
        preds_text = (torch.sigmoid(text_logits) > 0.5).long()
        preds_text_ocr = (torch.sigmoid(text_ocr_logits) > 0.5).long()

        return {
            'preds': preds,
            'preds_image': preds_image,
            'preds_text': preds_text,
            'preds_text_ocr': preds_text_ocr,
            'labels': labels.unsqueeze(1).long(),
            'loss': loss,
            'loss_image': loss_image,
            'loss_text': loss_text,
            'loss_fusion': loss_fusion,
            'loss_text_ocr': loss_text_ocr,
            'image_logits': image_logits,
            'text_logits': text_logits,
            'text_ocr_logits': text_ocr_logits,
            'logits': logits
        }

    def training_epoch_end(self, outputs) -> None:
        super().training_epoch_end(outputs)
        self.fusion_loss_weight = min(1, self.fusion_loss_weight + self.fusion_loss_change)
        wandb.log({'train_loss_image': torch.stack([x['loss_image'] for x in outputs]).mean().item()})
        wandb.log({'train_loss_text': torch.stack([x['loss_text'] for x in outputs]).mean().item()})
        wandb.log({'train_loss_fusion': torch.stack([x['loss_fusion'] for x in outputs]).mean().item()})
        self.log('train_loss_fusion', torch.stack([x['loss_fusion'] for x in outputs]).mean().item())

    def validation_epoch_end(self, outputs) -> None:
        super().validation_epoch_end(outputs)
        if self.use_softadapt:
            self.image_criterion_history.append(torch.stack([x['loss_image'] for x in outputs]).mean().item())
            self.text_criterion_history.append(torch.stack([x['loss_text'] for x in outputs]).mean().item())
            self.fusion_criterion_history.append(torch.stack([x['loss_fusion'] for x in outputs]).mean().item())
            wandb.log({'loss_weight_image': self.loss_weights[0].item()})
            wandb.log({'loss_weight_text': self.loss_weights[1].item()})
            wandb.log({'loss_weight_fusion': self.loss_weights[2].item()})
            wandb.log({'val_loss_image': self.image_criterion_history[-1]})
            wandb.log({'val_loss_text': self.text_criterion_history[-1]})
            wandb.log({'val_loss_fusion': self.fusion_criterion_history[-1]})
            self.log('val_loss_fusion', self.fusion_criterion_history[-1])

            if self.current_epoch != 0 and (self.current_epoch % self.update_loss_weights_per_epoch == 0):
                self._update_softadapt_weights()

    def _update_softadapt_weights(self):
        print('[!] Updating loss weights')
        self.loss_weights = self.softadapt.get_component_weights(torch.tensor(self.image_criterion_history),
                                                                 torch.tensor(self.text_criterion_history),
                                                                 torch.tensor(self.fusion_criterion_history),
                                                                 verbose=True)
        print(f'[!] loss weights: {self.loss_weights}')
        self.image_criterion_history = []
        self.text_criterion_history = []
        self.fusion_criterion_history = []

    def setup_criterion(self) -> torch.nn.Module:
        return None

    def setup_scores(self) -> List[torch.nn.Module]:
        train_scores = dict(f1=F1Score(task="binary"),
                            accuracy=Accuracy(task="binary"),
                            precision=Precision(task="binary"),
                            recall=Recall(task="binary"),
                            auc=AUROC(task="binary"))
        val_scores = dict(f1=F1Score(task="binary"),
                          accuracy=Accuracy(task="binary"),
                          precision=Precision(task="binary"),
                          recall=Recall(task="binary"),
                          auc=AUROC(task="binary"))
        test_scores = dict(f1=F1Score(task="binary"),
                           accuracy=Accuracy(task="binary"),
                           precision=Precision(task="binary"),
                           recall=Recall(task="binary"),
                           auc=AUROC(task="binary"))

        return [train_scores, val_scores, test_scores]

    def test_epoch_end(self, outputs, save_preds=False):
        super().test_epoch_end(outputs, save_preds)
        preds = torch.cat([x['preds'] for x in outputs])
        preds_image = torch.cat([x['preds_image'] for x in outputs])
        preds_text = torch.cat([x['preds_text'] for x in outputs])
        labels = torch.cat([x['labels'] for x in outputs])
        image_logits = torch.cat([x['image_logits'] for x in outputs])
        text_logits = torch.cat([x['text_logits'] for x in outputs])
        logits = torch.cat([x['logits'] for x in outputs])

        if self.checkpoint_path is None:
            self.checkpoint_path = f'{self.logger.save_dir}/{self.logger.name}/version_{self.logger.version}/checkpoints/'
        save_path = path.dirname(self.checkpoint_path)
        torch.save(
            dict(
                preds=preds,
                preds_image=preds_image,
                preds_text=preds_text,
                labels=labels,
                image_logits=image_logits,
                text_logits=text_logits,
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

    def init_weights(self):
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
