import torch
from torch import nn


class EDLMSELoss(nn.Module):
    def __init__(self, num_classes, annealing_step):
        super().__init__()
        self.num_classes = num_classes
        self.annealing_step = annealing_step
        self.ohe = torch.eye(self.num_classes)

    def forward(self, output, y, epoch_num):
        self.ohe = self.ohe.to(output.device)
        target = self.ohe[y]
        annealing_coef = torch.min(
            torch.tensor(1.0, dtype=torch.float32),
            torch.tensor(epoch_num / self.annealing_step, dtype=torch.float32),
        )
        evidence = nn.functional.relu(output)
        loss = self.squared_error_bayes_risk(evidence, target) + annealing_coef * 0 * self.kl_divergence_loss(evidence,
                                                                                                          target)
        return loss

    def squared_error_bayes_risk(self, evidence: torch.Tensor, target: torch.Tensor):
        alpha = evidence + 1.
        strength = alpha.sum(dim=-1)
        p = alpha / strength[:, None]
        err = (target - p) ** 2
        var = p * (1 - p) / (strength[:, None] + 1)
        loss = (err + var).sum(dim=-1)
        return loss.mean()

    def kl_divergence_loss(self, evidence: torch.Tensor, target: torch.Tensor):
        alpha = evidence + 1.
        n_classes = evidence.shape[-1]
        alpha_tilde = target + (1 - target) * alpha
        strength_tilde = alpha_tilde.sum(dim=-1)
        first = (torch.lgamma(alpha_tilde.sum(dim=-1))
                 - torch.lgamma(alpha_tilde.new_tensor(float(n_classes)))
                 - (torch.lgamma(alpha_tilde)).sum(dim=-1))

        second = (
                (alpha_tilde - 1) *
                (torch.digamma(alpha_tilde) - torch.digamma(strength_tilde)[:, None])
        ).sum(dim=-1)

        loss = first + second

        return loss.mean()


def kl_divergence_loss(evidence: torch.Tensor, target: torch.Tensor):
    alpha = evidence + 1.
    n_classes = evidence.shape[-1]
    alpha_tilde = target + (1 - target) * alpha
    strength_tilde = alpha_tilde.sum(dim=-1)
    first = (torch.lgamma(alpha_tilde.sum(dim=-1))
             - torch.lgamma(alpha_tilde.new_tensor(float(n_classes)))
             - (torch.lgamma(alpha_tilde)).sum(dim=-1))

    second = (
            (alpha_tilde - 1) *
            (torch.digamma(alpha_tilde) - torch.digamma(strength_tilde)[:, None])
    ).sum(dim=-1)

    loss = first + second

    return loss.mean()


class EDLCELoss(nn.Module):
    def __init__(self, num_classes, annealing_step):
        super().__init__()
        self.num_classes = num_classes
        self.annealing_step = annealing_step
        self.ohe = torch.eye(self.num_classes)

    def forward(self, output, y, epoch_num):
        self.ohe = self.ohe.to(output.device)
        target = self.ohe[y]
        annealing_coef = torch.min(
            torch.tensor(1.0, dtype=torch.float32),
            torch.tensor(epoch_num / self.annealing_step, dtype=torch.float32),
        )
        evidence = nn.functional.relu(output)
        loss = self.cross_entropy_bayes_risk(evidence, target)
        return loss

    def cross_entropy_bayes_risk(self, evidence: torch.Tensor, target: torch.Tensor):
        alpha = evidence + 1.
        strength = alpha.sum(dim=-1)
        loss = (target * (torch.digamma(strength)[:, None] - torch.digamma(alpha))).sum(dim=-1)
        return loss.mean()
