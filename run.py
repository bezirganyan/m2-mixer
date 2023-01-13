import argparse
import os
from typing import Type

import wandb
from datasets.multimodal import MMIMDBDataModule
from omegaconf import OmegaConf
import pytorch_lightning as pl

from models.mmimdb_gmlp import MMIDB_GMLP
from models.mmimdb_mixer import MMIDB_Mixer
from utils.utils import deep_update, todict


def get_model(model_type: str) -> type[pl.LightningModule]:
    if model_type == 'mmimdb_mixer':
        return MMIDB_Mixer
    elif model_type == 'mmimdb_gmlp':
        return MMIDB_GMLP
    else:
        raise NotImplementedError


def get_data_module(data_type: str) -> type[pl.LightningDataModule]:
    if data_type == 'mmimdb':
        return MMIMDBDataModule
    else:
        raise NotImplementedError


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg', type=str)
    parser.add_argument('-n', '--name', type=str)
    parser.add_argument('-p', '--ckpt', type=str)
    parser.add_argument('-m', '--mode', type=str, default='train')
    parser.add_argument('--disable-wandb', action='store_true', default=False)
    args, unknown = parser.parse_known_args()
    return args, unknown


if __name__ == '__main__':
    args, unknown = parse_args()
    cfg = OmegaConf.load(args.cfg)
    train_cfg = cfg.train
    dataset_cfg = cfg.dataset
    model_cfg = cfg.model
    pl.seed_everything(train_cfg.seed)
    unknown = [u.replace('--', '') for u in unknown]
    ucfg = OmegaConf.from_cli(unknown)
    if 'model' in ucfg:
        deep_update(model_cfg, ucfg.model)
    if 'train' in ucfg:
        deep_update(train_cfg, ucfg.train)
    if 'dataset' in ucfg:
        deep_update(dataset_cfg, ucfg.dataset)

    if args.disable_wandb:
        wandb.init(project='MMixer', name=args.name, config=todict(cfg), mode='disabled')
    else:
        wandb.init(project='MMixer', name=args.name, config=todict(cfg))

    model = get_model(model_cfg.type)
    if args.ckpt:
        train_module = model.load_from_checkpoint(args.ckpt, optimizer_cfg=train_cfg.optimizer,
                                                  model_cfg=model_cfg)
    else:
        train_module = model(train_cfg.optimizer, model_cfg)
    wandb.watch(train_module)
    data_module = get_data_module(dataset_cfg.type)
    data_module = data_module(**dataset_cfg.params)

    trainer = pl.Trainer(
        callbacks=[
            # pl.callbacks.EarlyStopping(monitor='val_score_sk', patience=5, mode='max'),
            pl.callbacks.ModelCheckpoint(
                monitor='val_score',
                save_last=True,
                save_top_k=5,
                mode='max'
            )
        ],
        gpus=-1,
        log_every_n_steps=train_cfg.log_interval_steps,
        logger=pl.loggers.TensorBoardLogger(train_cfg.tensorboard_path, args.name),
        max_epochs=train_cfg.epochs
    )
    wandb.config.update({"run_version": trainer.logger.version})
    if args.mode == 'train':
        trainer.fit(train_module, data_module)
    if args.mode == 'test':
        trainer.test(train_module, data_module)
