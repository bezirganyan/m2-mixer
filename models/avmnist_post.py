from typing import List

import torch
from omegaconf import DictConfig
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import Accuracy, F1Score, Precision

import modules
from modules.train_test_module import AbstractTrainTestModule


class AVMnistMixerMultiLossTP(AbstractTrainTestModule):
    def __init__(self, model_cfg: DictConfig, optimizer_cfg: DictConfig, **kwargs):
        super(AVMnistMixerMultiLossTP, self).__init__(optimizer_cfg, log_confusion_matrix=True, **kwargs)
        self.model_cfg = model_cfg
        for param in self.parameters():
            param.requires_grad = False

        self.classifier = modules.get_classifier_by_name(**model_cfg.modalities.classification)

    def shared_step(self, batch, **kwargs):
        el = batch['data'].to(self.device).reshape(batch['data'].shape[0], -1)
        out = self.classifier(el).squeeze(1)
        pos_weigth = torch.tensor(self.model_cfg.pos_weight).repeat(batch['label'].shape[0]).to(self.device)
        pos_weigth = pos_weigth * batch['label'].long() + (1 - batch['label'].long())
        loss = torch.nn.functional.binary_cross_entropy_with_logits(out,
                                                                    batch['label'].to(self.device).float(),
                                                                    weight=pos_weigth)
        out = torch.sigmoid(out)
        return dict(
            loss=loss,
            preds=out,
            labels=batch['label']
        )

    def setup_criterion(self) -> torch.nn.Module:
        return None

    def setup_scores(self) -> List[torch.nn.Module]:
        train_scores = dict(acc=Accuracy(task="binary"),
                            f1=F1Score(task="binary"),
                            precision=Precision(task="binary"))
        val_scores = dict(acc=Accuracy(task="binary"),
                          f1=F1Score(task="binary"),
                          precision=Precision(task="binary"))
        test_scores = dict(acc=Accuracy(task="binary"),
                           f1=F1Score(task="binary"),
                           precision=Precision(task="binary"))

        return [train_scores, val_scores, test_scores]

    def configure_optimizers(self):
        optimizer_cfg = self.optimizer_cfg
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), **optimizer_cfg)
        scheduler = ReduceLROnPlateau(optimizer, patience=5, verbose=True)

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }


# class AVMNISTModalityPicker(AbstractTrainTestModule):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.image_model = AVMnistMixerMultiLossTP()
#
#     def setup_criterion(self) -> torch.nn.Module:
#         pass
#
#     def setup_scores(self) -> List[Dict[str, Metric]]:
#         pass
#
#     def shared_step(self, batch, **kwargs) -> Dict[str, Any]:
#         pass
