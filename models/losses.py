import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class RadiantLoss(nn.Module):
    """
    Unified training loss formulation for Radiant (Paper Section III & IV).
    Combines:
      1. Detection classification loss (L_det) supervising fault identification;
      2. Behavioral evolution prediction loss (L_pred) supervising expected future states;
      3. Expert load-balancing regularization loss (L_bal) preventing router collapse.
    """

    def __init__(
        self,
        cls_weight: float = 1.0,
        pred_weight: float = 0.1,
        moe_balance_weight: float = 0.05,
    ) -> None:
        super().__init__()
        self.cls_weight = cls_weight
        self.pred_weight = pred_weight
        self.moe_balance_weight = moe_balance_weight
        self.cls_criterion = nn.CrossEntropyLoss()

    def forward(
        self,
        cls_logits: torch.Tensor,
        cls_targets: torch.Tensor,
        pred_last: torch.Tensor,
        actual_last: torch.Tensor,
        moe_balance_loss: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        # 1. Detection classification loss over service/window states
        cls_loss = self.cls_criterion(cls_logits, cls_targets)

        # 2. Behavioral evolution prediction loss in latent representation space
        pred_loss = F.mse_loss(pred_last, actual_last.detach())

        total_loss = self.cls_weight * cls_loss + self.pred_weight * pred_loss

        # 3. Router load-balancing regularization
        if moe_balance_loss is not None and self.moe_balance_weight > 0:
            total_loss = total_loss + self.moe_balance_weight * moe_balance_loss

        return {
            "total_loss": total_loss,
            "cls_loss": cls_loss.detach(),
            "pred_loss": pred_loss.detach(),
            "moe_balance_loss": moe_balance_loss.detach() if moe_balance_loss is not None else torch.zeros(()),
        }
