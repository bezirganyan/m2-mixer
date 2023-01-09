import wandb
from torch.nn import BCEWithLogitsLoss
from torchmetrics.classification import MulticlassF1Score

from datasets.multimodal import MMIMDBDataModule
from datasets.pnlp import PnlpMixerDataModule
from models.mmixer import MultimodalMixer
from models.pnlp import PnlpMixerSeqCls, PnlpMixerTokenCls
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from typing import Any, Dict, List
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from sklearn.metrics import f1_score

from utils.utils import deep_update


class MMIDB_Mixer(pl.LightningModule):
    def __init__(self, optimizer_cfg: DictConfig, model_cfg: DictConfig, **kwargs):
        super(MMIDB_Mixer, self).__init__(**kwargs)
        self.optimizer_cfg = optimizer_cfg
        self.model = MultimodalMixer(
            model_cfg.image,
            model_cfg.text,
            model_cfg.multimodal,
            model_cfg.bottleneck,
            model_cfg.classification,
            dropout=model_cfg.dropout
        )
        self.criterion = BCEWithLogitsLoss()
        self.train_score = MulticlassF1Score(model_cfg.classification.num_classes, average='weighted', task='multilabel')
        self.val_score = MulticlassF1Score(model_cfg.classification.num_classes, average='weighted', task='multilabel')
        self.test_score = MulticlassF1Score(model_cfg.classification.num_classes, average='weighted', task='multilabel')

    def shared_step(self, batch):
        image = batch['image']
        text = batch['text']
        labels = batch['label']
        logits = self.model(image, text)
        loss = self.criterion(logits, labels.float())
        preds = torch.sigmoid(logits) > 0.5

        return {
            'preds': preds,
            'labels': labels,
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
        score = f1_score(results['preds'].cpu().detach().numpy(), results['labels'].cpu().detach().numpy(), average='weighted', zero_division=1)
        self.log('train_score_sk', score, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        wandb.log({'train_loss_step': results['loss'].cpu().item()})
        return results

    def training_epoch_end(self, outputs):
        # self.log('train_score', self.train_score.compute(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        preds = torch.cat([output['preds'] for output in outputs], dim=0)
        labels = torch.cat([output['labels'] for output in outputs], dim=0)
        score = f1_score(preds.cpu().detach().numpy(), labels.cpu().detach().numpy(), average='weighted', zero_division=1)
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
        score = f1_score(preds.cpu().detach().numpy(), labels.cpu().detach().numpy(), average='weighted', zero_division=1)
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
        score = f1_score(preds.cpu().detach().numpy(), labels.cpu().detach().numpy(), average='weighted', zero_division=1)
        self.log('test_score_sk', score, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        wandb.log({'test_score': score})
    def configure_optimizers(self):
        optimizer_cfg = self.optimizer_cfg
        optimizer = torch.optim.Adam(self.parameters(), **optimizer_cfg)
        return optimizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg', type=str)
    parser.add_argument('-n', '--name', type=str)
    parser.add_argument('-p', '--ckpt', type=str)
    parser.add_argument('-m', '--mode', type=str, default='train')
    args, unknown = parser.parse_known_args()
    return args, unknown


def get_module_cls(type: str):
    if type == 'matis' or type == 'imdb':
        return MMIDB_Mixer


if __name__ == '__main__':
    args, unknown = parse_args()
    cfg = OmegaConf.load(args.cfg)
    vocab_cfg = cfg.vocab
    train_cfg = cfg.train
    model_cfg = cfg.model

    unknown = [u.replace('--', '') for u in unknown]
    ucfg = OmegaConf.from_cli(unknown)
    if 'model' in ucfg:
        deep_update(model_cfg, ucfg.model)
    if 'train' in ucfg:
        deep_update(train_cfg, ucfg.train)
    if 'vocab' in ucfg:
        deep_update(vocab_cfg, ucfg.vocab)

    wandb.init(project='MMixer', name=args.name, config=dict(cfg))

    module_cls = get_module_cls(train_cfg.dataset_type)
    if args.ckpt:
        train_module = module_cls.load_from_checkpoint(args.ckpt, optimizer_cfg=train_cfg.optimizer,
                                                       model_cfg=model_cfg)
    else:
        train_module = module_cls(train_cfg.optimizer, model_cfg)
    wandb.watch(train_module)
    data_module = MMIMDBDataModule('../output', 32, 8, cfg.vocab, train_cfg, model_cfg.projection)

    trainer = pl.Trainer(
        # accelerator='ddp',
        # amp_backend='native',
        # amp_level='O2',
        callbacks=[
            pl.callbacks.EarlyStopping(monitor='val_score_sk', patience=5, mode='max'),
            pl.callbacks.ModelCheckpoint(
                monitor='val_score_sk',
                save_last=True,
                save_top_k=5,
                mode='max'
            )
        ],
        # checkpoint_callback=True,
        gpus=-1,
        log_every_n_steps=train_cfg.log_interval_steps,
        logger=pl.loggers.TensorBoardLogger(train_cfg.tensorboard_path, args.name),
        max_epochs=train_cfg.epochs
    )
    if args.mode == 'train':
        trainer.fit(train_module, data_module)
    if args.mode == 'test':
        trainer.test(train_module, data_module)
