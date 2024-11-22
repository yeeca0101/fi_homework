import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from loss import SSIMLoss,dice_loss

class ValMetric(nn.Module):
    def __init__(self,):
        super(ValMetric, self).__init__()
        self.ssim_fn = SSIMLoss(data_range=1,n_channels=3)
        self.dice_fn = dice_loss

    @torch.no_grad()
    def forward(self, pred, target,pred_mask,target_mask):
        B, C, H, W = pred.shape

        ssim = F.relu(1-self.ssim_fn(pred,target))
        dice = F.relu(1-self.dice_fn(pred_mask,target_mask))
        pred = pred.view(B, -1)
        target = target.view(B, -1)

        # MAE 계산
        mae = torch.mean(torch.abs(pred - target))
        return {'mae':mae,'ssim':ssim,'dice':dice}

    def compute_metrics(self, pred, target,pred_mask,target_mask):
        with torch.no_grad():
            return self.forward(pred, target,pred_mask,target_mask)


if __name__ == "__main__":
    pred = torch.randn((4,3,32,32),requires_grad=True)
    target  =torch.randint(0, 2, (4,3, 32, 32)).float()

    ir_metrics = ValMetric()

    print("개별 메트릭:", ir_metrics.compute_metrics(pred, target))