# losses.py (new)
import torch, torch.nn as nn
class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_pos=0, gamma_neg=4, clip=0.05, eps=1e-8):
        super().__init__()
        self.gp, self.gn, self.clip, self.eps = gamma_pos, gamma_neg, clip, eps
    def forward(self, logits, targets):
        x = torch.sigmoid(logits)
        if self.clip is not None and self.clip > 0:
            x = torch.clamp(x, self.clip, 1.0 - self.clip)
        xs_pos, xs_neg = x, 1 - x
        pt = targets * xs_pos + (1 - targets) * xs_neg
        w = (targets * (1 - xs_pos))**self.gp + ((1 - targets) * xs_pos)**self.gn
        loss = - (targets * torch.log(xs_pos + self.eps) + (1 - targets) * torch.log(xs_neg + self.eps))
        return (loss * w).mean()
