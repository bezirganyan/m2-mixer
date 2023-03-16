import torch
from torch import nn


class EDLMSELoss(nn.Module):
    def __init__(self, num_classes, annealing_step):
        super().__init__()
        self.num_classes = num_classes
        self.annealing_step = annealing_step
        self.ohe = torch.eye(self.num_classes)


    def forward(self, output, target, epoch_num):
        evidence = nn.functional.relu(output)
        alpha = evidence + 1
        loss = torch.mean(
            self.mse_loss(target, alpha, epoch_num, self.annealing_step)
        )
        return loss

    def mse_loss(self, yn, alpha, epoch_num, annealing_step):
        self.ohe = self.ohe.to(yn.device)
        y = self.ohe[yn]
        loglikelihood = self.loglikelihood_loss(y, alpha)

        annealing_coef = torch.min(
            torch.tensor(1.0, dtype=torch.float32),
            torch.tensor(epoch_num / annealing_step, dtype=torch.float32),
        )

        kl_alpha = (alpha - 1) * (1 - y) + 1
        kl_div = annealing_coef * self.kl_divergence(kl_alpha)
        return loglikelihood + kl_div

    def kl_divergence(self, alpha):
        ones = torch.ones([1, self.num_classes], dtype=torch.float32).to(alpha.device)
        sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
        first_term = (
                torch.lgamma(sum_alpha)
                - torch.lgamma(alpha).sum(dim=1, keepdim=True)
                + torch.lgamma(ones).sum(dim=1, keepdim=True)
                - torch.lgamma(ones.sum(dim=1, keepdim=True))
        )
        second_term = (
            (alpha - ones)
            .mul(torch.digamma(alpha) - torch.digamma(sum_alpha))
            .sum(dim=1, keepdim=True)
        )
        kl = first_term + second_term
        return kl

    @staticmethod
    def loglikelihood_loss(y, alpha):
        S = torch.sum(alpha, dim=1, keepdim=True)
        loglikelihood_err = torch.sum((y - (alpha / S)) ** 2, dim=1, keepdim=True)
        loglikelihood_var = torch.sum(
            alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True
        )
        loglikelihood = loglikelihood_err + loglikelihood_var
        return loglikelihood
