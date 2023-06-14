import argparse
import os

import torch
import wandb

import datasets
import models
from omegaconf import OmegaConf
import pytorch_lightning as pl

from utils.utils import deep_update, todict


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg', type=str)
    parser.add_argument('-n', '--name', type=str)
    parser.add_argument('-p', '--ckpt', type=str)
    parser.add_argument('-r', '--runs', type=int, default=10)
    parser.add_argument('-m', '--mode', type=str, default='train')
    parser.add_argument('--project', type=str, default='MMixer Significance Test')
    parser.add_argument('--patience', type=int, default=30)
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

    scores = []
    for r in range(args.runs):
        trainer = pl.Trainer(
            callbacks=[
                pl.callbacks.EarlyStopping(monitor='val_loss', patience=args.patience, mode='min'),
                pl.callbacks.ModelCheckpoint(
                    monitor=train_cfg.monitor,
                    save_last=True,
                    save_top_k=5,
                    mode=train_cfg.monitor_mode
                )
            ],
            accelerator='gpu',
            devices=-1,
            log_every_n_steps=train_cfg.log_interval_steps,
            logger=pl.loggers.TensorBoardLogger(train_cfg.tensorboard_path, args.name),
            max_epochs=train_cfg.epochs
        )

        model = models.get_model(model_cfg.type)
        if args.ckpt:
            train_module = model.load_from_checkpoint(args.ckpt, optimizer_cfg=train_cfg.optimizer,
                                                      model_cfg=model_cfg)
        else:
            train_module = model(model_cfg, train_cfg.optimizer)
        data_module = datasets.get_data_module(dataset_cfg.type)
        if dataset_cfg.params.num_workers == -1:
            dataset_cfg.params.num_workers = os.cpu_count()
        data_module = data_module(**dataset_cfg.params)

        if args.disable_wandb:
            wandb.init(project=args.project, name=f'{args.name} - {r}', config=todict(cfg), mode='disabled')
        else:
            wandb.init(project=args.project, name=f'{args.name} - {r}', config=todict(cfg))
        wandb.config.update({"run_version": trainer.logger.version})
        wandb.watch(train_module)
        if args.mode == 'train':
            trainer.fit(train_module, data_module)
        trainer.test(train_module, data_module, ckpt_path='best')
        score = trainer.callback_metrics['test_acc']
        scores.append(score)
        torch.save(scores, f'{trainer.logger.save_dir}/test_scores.pt')
        wandb.finish()
