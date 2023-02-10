import argparse
import os

import wandb

from datasets.avmnist import AVMnistDataModule
from datasets.get_processed_mmimdb import MMIMDBExtDataModule
from datasets.imagenet_dataset import ImagenetDataModule
from datasets.multimodal import MMIMDBDataModule
from omegaconf import OmegaConf
import pytorch_lightning as pl

from models.avmnist import AVMnistImagePooler, AVMnistImageMixer, AVMnistMixer, AVMnistAudioMixer, AVMnistgMLP, \
    AVMnistMixerLF, AVMnistMixerMultiLoss
from models.convnet import ConvNet
from models.gmlp_autoencoder import GMLPAutoencoder, MMIMDGMLPClassifier
from models.imagenet_mixer import ImagenetPooler
from models.mixer_autoencoder import MixerAutoencoder, MMIMDBMixerGMLPClassifier, MMIMDBEncoderClassifier
from models.mmimdb_gmlp import MMIDB_GMLP
from models.mmimdb_mixer import MMIDBMixer, MMIDBPooler
from utils.utils import deep_update, todict


def get_model(model_type: str) -> type[pl.LightningModule]:
    if model_type == 'mmimdb_mixer':
        return MMIDBMixer
    elif model_type == 'mmimdb_gmlp':
        return MMIDB_GMLP
    elif model_type == 'mmimdb_autoencoder':
        return MixerAutoencoder
    elif model_type == 'mmimdb_autoencoder_classifier':
        return MMIMDBEncoderClassifier
    elif model_type == 'mmimdb_gmlp_autoencoder':
        return GMLPAutoencoder
    elif model_type == 'mmimdb_autoencoder_classifier':
        return MMIMDBMixerGMLPClassifier
    elif model_type == 'mmimdb_autoencoder_gmlp_classifier':
        return MMIMDGMLPClassifier
    elif model_type == 'mmimdb_convnet':
        return ConvNet
    elif model_type == 'mmimdb_pooler':
        return MMIDBPooler
    elif model_type == 'imagenet_pooler':
        return ImagenetPooler
    elif model_type == 'avmnist_image_pooler':
        return AVMnistImagePooler
    elif model_type == 'avmnist_image_mixer':
        return AVMnistImageMixer
    elif model_type == 'avmnist_mixer':
        return AVMnistMixer
    elif model_type == 'avmnist_audio_mixer':
        return AVMnistAudioMixer
    elif model_type == 'avmnist_gmlp':
        return AVMnistgMLP
    elif model_type == 'avmnist_mixer_lf':
        return AVMnistMixerLF
    elif model_type == 'avmnist_mixer_3loss':
        return AVMnistMixerMultiLoss
    else:
        raise NotImplementedError


def get_data_module(data_type: str) -> type[pl.LightningDataModule]:
    if data_type == 'mmimdb':
        return MMIMDBDataModule
    elif data_type == 'mmimdb_ext':
        return MMIMDBExtDataModule
    elif data_type == 'imagenet':
        return ImagenetDataModule
    elif data_type == 'avmnist':
        return AVMnistDataModule
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
        train_module = model(model_cfg, train_cfg.optimizer)
    wandb.watch(train_module)
    data_module = get_data_module(dataset_cfg.type)
    if dataset_cfg.params.num_workers == -1:
        dataset_cfg.params.num_workers = os.cpu_count()
    data_module = data_module(**dataset_cfg.params)

    trainer = pl.Trainer(
        callbacks=[
            pl.callbacks.EarlyStopping(monitor='val_loss', patience=30, mode='min'),
            pl.callbacks.ModelCheckpoint(
                monitor=train_cfg.monitor,
                save_last=True,
                save_top_k=5,
                mode=train_cfg.monitor_mode
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
