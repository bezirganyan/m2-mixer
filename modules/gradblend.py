from copy import deepcopy

import torch
from torch import nn
from torch.optim import Adam
from tqdm import tqdm


class MultiModalEncoder(nn.Module):
    def __init__(self, unimodal_encoders, multimodal_encoder):
        super().__init__()
        self.unimodal_encoders = nn.ModuleList(unimodal_encoders)
        self.multimodal_encoder = multimodal_encoder

    def forward(self, x):
        encs = [e(x[i]) for i, e in enumerate(self.unimodal_encoders)]
        max_dim = max([len(e.shape) for e in encs])
        encs = [e.unsqueeze(1) if len(e.shape) < max_dim else e for e in encs]
        cat = torch.cat(encs, dim=1)
        out = self.multimodal_encoder(cat)
        return out


class GradBlend:
    def __init__(self, model, unimodal_encoders, unimodal_heads, multimodal_encoder, multimodal_head,
                 loss, train_dataloader, val_dataloader,
                 label_name='labels', epochs=20, loss_args=None):
        if loss_args is None:
            loss_args = {}
        self.val_dataloader = val_dataloader
        self.train_dataloader = train_dataloader
        self.label_name = label_name
        self.epochs = epochs
        self.model = model
        self.unimodal_encoders = unimodal_encoders
        self.unimodal_heads = unimodal_heads
        self.multimodal_encoder = multimodal_encoder
        self.multimodal_head = multimodal_head
        self.losses = []
        for _ in range(len(unimodal_heads) + 1):
            self.losses.append(loss(**loss_args))

    def get_loss(self, enc, head, modality_ind, split='train', device='cuda'):
        m = modality_ind
        if split == 'train':
            dataloader = self.train_dataloader
        elif split == 'val':
            dataloader = self.val_dataloader
        else:
            raise ValueError('Split must be train or val')
        loss = 0
        for batch in dataloader:
            x = batch[m].to(device) if m is not None else [b.to(device) for b in batch[:-1]]
            y = batch[-1].to(device)
            x = enc(x)
            if len(x.shape) > 2:
                x = x.reshape(x.shape[0], -1, x.shape[-1]).mean(dim=1)
            pred = head(x)
            loss += self.losses[m](pred, y) if m is not None else self.losses[-1](pred, y)
        return loss

    def get_single_weight(self, enc, head, modality_ind, device='cuda', params=None):
        m = modality_ind
        if params is None:
            optimizer = Adam(list(enc.parameters()) + list(head.parameters()))
        else:
            optimizer = Adam(params)
        loss_N_train = self.get_loss(enc, head, m, split='train', device=device)
        loss_N_val = self.get_loss(enc, head, m, split='val', device=device)
        for e in tqdm(range(self.epochs), desc=f'Modality {m}', leave=False):
            for batch in tqdm(self.train_dataloader, desc=f'Epoch {e}', leave=False):
                optimizer.zero_grad()
                x = batch[m].to(device) if m is not None else [b.to(device) for b in batch[:-1]]
                y = batch[-1].to(device)
                x = enc(x)
                if len(x.shape) > 2:
                    x = x.reshape(x.shape[0], -1, x.shape[-1]).mean(dim=1)
                pred = head(x)
                loss = self.losses[m](pred, y) if m is not None else self.losses[-1](pred, y)
                loss.backward()
                optimizer.step()
        loss_Nn_train = self.get_loss(enc, head, m, split='train', device=device)
        loss_Nn_val = self.get_loss(enc, head, m, split='val', device=device)

        ON = loss_N_val - loss_N_train
        ONn = loss_Nn_val - loss_Nn_train
        O = ONn - ON
        G = loss_Nn_val - loss_N_val
        w = abs(O / (G ** 2))
        return w

    def get_weights(self, device='cuda'):
        weights = []
        for m in range(len(self.unimodal_encoders)):
            enc = deepcopy(self.unimodal_encoders[m])
            head = deepcopy(self.unimodal_heads[m])
            w = self.get_single_weight(enc, head, m, device=device)
            weights.append(w)
        enc = deepcopy(self.multimodal_encoder)
        uenc = [deepcopy(e) for e in self.unimodal_encoders]
        head = deepcopy(self.multimodal_head)
        mmodel = MultiModalEncoder(uenc, enc)
        for p in mmodel.unimodal_encoders.parameters():
            p.requires_grad = False
        w = self.get_single_weight(mmodel, head, None, device=device)
        weights.append(w)
        z = sum(weights)
        weights = [(w / z).item() for w in weights]
        return weights

    def __call__(self, *args, **kwargs):
        return self.get_weights(*args, **kwargs)
