import abc
from typing import List, Dict, Any

import numpy as np
import torch
import wandb
from omegaconf import DictConfig
from torch.nn import BCEWithLogitsLoss
import pytorch_lightning as pl
from torchmetrics import F1Score


class AbstractTrainTestModule(pl.LightningModule, abc.ABC):
    def __init__(self, optimizer_cfg: DictConfig, **kwargs):
        super(AbstractTrainTestModule, self).__init__(**kwargs)
        self.optimizer_cfg = optimizer_cfg
        self.criterion = self.setup_criterion()
        self.train_score, self.val_score, self.test_score = self.setup_scores()

    @abc.abstractmethod
    def setup_criterion(self) -> torch.nn.Module:
        raise NotImplementedError

    @abc.abstractmethod
    def setup_scores(self) -> List[torch.nn.Module]:
        raise NotImplementedError
    @abc.abstractmethod
    def shared_step(self, batch) -> Dict[str, Any]:
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        results = self.shared_step(batch)
        self.log('train_loss', results['loss'], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        score = self.train_score(results['preds'], results['labels'])
        self.log('train_score', score, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        wandb.log({'train_loss_step': results['loss'].cpu().item()})
        return results

    def training_epoch_end(self, outputs):
        wandb.log({'train_score': self.train_score.compute()})
        wandb.log({'train_loss': np.mean([output['loss'].cpu().item() for output in outputs])})

    def validation_step(self, batch, batch_idx):
        results = self.shared_step(batch)
        self.log('val_loss', results['loss'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_score', self.val_score(results['preds'], results['labels']), on_step=False, on_epoch=True,
                 prog_bar=True, logger=True)
        return results

    def validation_epoch_end(self, outputs):
        wandb.log({'val_score': self.val_score.compute()})
        wandb.log({'val_loss': np.mean([output['loss'].cpu().item() for output in outputs])})

    def test_step(self, batch, batch_idx):
        results = self.shared_step(batch)
        self.log('test_loss', results['loss'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_score', self.test_score(results['preds'], results['labels']), on_step=False, on_epoch=True,
                 prog_bar=True, logger=True)
        return results

    def test_epoch_end(self, outputs):
        wandb.log({'test_score': self.test_score.compute()})

    def configure_optimizers(self):
        optimizer_cfg = self.optimizer_cfg
        optimizer = torch.optim.Adam(self.parameters(), **optimizer_cfg)
        return optimizer
