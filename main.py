import argparse

import wandb
from datasets.multimodal import MMIMDBDataModule
from omegaconf import OmegaConf
import pytorch_lightning as pl

from models.mmimdb_mixer import MMIDB_Mixer
from utils.utils import deep_update, todict


def parse_args():
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
    vocab_cfg = cfg.vocab
    train_cfg = cfg.train
    model_cfg = cfg.model
    pl.seed_everything(train_cfg.seed)
    unknown = [u.replace('--', '') for u in unknown]
    ucfg = OmegaConf.from_cli(unknown)
    if 'model' in ucfg:
        deep_update(model_cfg, ucfg.model)
    if 'train' in ucfg:
        deep_update(train_cfg, ucfg.train)
    if 'vocab' in ucfg:
        deep_update(vocab_cfg, ucfg.vocab)

    if args.disable_wandb:
        wandb.init(project='MMixer', name=args.name, config=todict(cfg), mode='disabled')
    else:
        wandb.init(project='MMixer', name=args.name, config=todict(cfg))

    module_cls = MMIDB_Mixer(train_cfg.dataset_type)
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
    wandb.config.update({"run_version": trainer.logger.version})
    if args.mode == 'train':
        trainer.fit(train_module, data_module)
    if args.mode == 'test':
        trainer.test(train_module, data_module)
