from typing import List, Dict, Any

import numpy as np
import torch
import wandb
from omegaconf import DictConfig
from sklearn.metrics import f1_score
from torch.nn import BCEWithLogitsLoss
import pytorch_lightning as pl
from torchmetrics.functional import accuracy

from models.mmixer import MultimodalMixer, MosiMixer


class MOSIMixerLearningModule(pl.LightningModule):
    def __init__(self, optimizer_cfg: DictConfig, model_cfg: DictConfig, **kwargs):
        super(MOSIMixerLearningModule, self).__init__(**kwargs)
        self.optimizer_cfg = optimizer_cfg
        self.model = MosiMixer(
            model_cfg.image,
            model_cfg.text,
            model_cfg.audio,
            model_cfg.multimodal,
            model_cfg.bottleneck,
            model_cfg.classification,
            dropout=model_cfg.dropout
        )
        self.criterion = BCEWithLogitsLoss()
        self.score = accuracy

    def shared_step(self, batch):
        vision, audio, label, tmp_label, features = batch
        label = label.squeeze(1)
        logits = self.model(vision, features, audio)
        loss = self.criterion(logits, label.float())
        preds = torch.sigmoid(logits)

        return {
            'preds': preds,
            'labels': label,
            'loss': loss
        }

    def compute_accuracy(self, outputs: List[Dict[str, Any]]):
        corr = 0
        all = 0
        for output in outputs:
            corr += output['corr']
            all += output['all']
        return {
            'acc': corr / all,
        }

    def training_step(self, batch, batch_idx):
        results = self.shared_step(batch)
        self.log('train_loss', results['loss'], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # self.log('train_score', self.train_score(results['preds'], results['labels']), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        score = self.score(results['preds'], results['labels'], 'binary')
        self.log('train_score_sk', score, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        wandb.log({'train_loss_step': results['loss'].cpu().item()})
        return results

    def training_epoch_end(self, outputs):
        # self.log('train_score', self.train_score.compute(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        preds = torch.cat([output['preds'] for output in outputs], dim=0)
        labels = torch.cat([output['labels'] for output in outputs], dim=0)
        score = self.score(preds, labels, 'binary')
        self.log('train_score_sk', score, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        wandb.log({'train_score': score})
        wandb.log({'train_loss': np.mean([output['loss'].cpu().item() for output in outputs])})

    def validation_step(self, batch, batch_idx):
        results = self.shared_step(batch)
        self.log('val_loss', results['loss'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        # self.log('val_score', self.val_score(results['preds'], results['labels']), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return results

    def validation_epoch_end(self, outputs):
        # self.log('val_score', self.val_score.compute(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        preds = torch.cat([output['preds'] for output in outputs], dim=0)
        labels = torch.cat([output['labels'] for output in outputs], dim=0)
        score = self.score(preds, labels, 'binary')
        self.log('val_score_sk', score, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        wandb.log({'val_score': score})
        wandb.log({'val_loss': np.mean([output['loss'].cpu().item() for output in outputs])})

    def test_step(self, batch, batch_idx):
        results = self.shared_step(batch)
        self.log('test_loss', results['loss'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        # self.log('test_score', self.test_score(results['preds'], results['labels']), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return results

    def test_epoch_end(self, outputs):
        # self.log('test_score', self.test_score.compute(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        preds = torch.cat([output['preds'] for output in outputs], dim=0)
        labels = torch.cat([output['labels'] for output in outputs], dim=0)
        score = self.score(preds, labels, 'binary')
        self.log('test_score_sk', score, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        wandb.log({'test_score': score})
    def configure_optimizers(self):
        optimizer_cfg = self.optimizer_cfg
        optimizer = torch.optim.Adam(self.parameters(), **optimizer_cfg)
        return optimizer