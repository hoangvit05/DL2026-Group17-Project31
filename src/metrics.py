import torch
import torch.nn as nn


class ComboLoss(nn.Module):
    """
    Compound loss combining Binary Cross-Entropy (BCE) and Soft Dice Loss:
    Loss = 0.5 * BCE + 0.5 * (1 - Dice)
    """
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits, targets):
        bce = self.bce(logits, targets)
        p = torch.sigmoid(logits).view(-1)
        t = targets.view(-1)
        inter = (p * t).sum()
        dice = (2.0 * inter + 1e-6) / (p.sum() + t.sum() + 1e-6)
        return self.bce_weight * bce + self.dice_weight * (1.0 - dice)


@torch.no_grad()
def compute_dice_iou(logits, targets, threshold=0.5):
    """
    Computes batch-averaged Dice Similarity Coefficient and Intersection over Union (IoU).
    """
    preds = (torch.sigmoid(logits) > threshold).float().view(logits.size(0), -1)
    targets = targets.view(targets.size(0), -1)
    inter = (preds * targets).sum(dim=1)
    union = preds.sum(dim=1) + targets.sum(dim=1)
    dice = (2.0 * inter + 1e-6) / (union + 1e-6)
    iou = (inter + 1e-6) / (union - inter + 1e-6)
    return dice.mean().item(), iou.mean().item()
