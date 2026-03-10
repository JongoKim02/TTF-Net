import torch
import torch.nn as nn
import torch.nn.functional as F

class InlineTriPlaneTextFuser(nn.Module):
    def __init__(self, in_ch: int, text_dim: int, d: int = 64):
        super().__init__()
        self.ax_proj = nn.Conv2d(in_ch, d, kernel_size=1, bias=False)  # (H,W)
        self.sa_proj = nn.Conv2d(in_ch, d, kernel_size=1, bias=False)  # (D,W)
        self.co_proj = nn.Conv2d(in_ch, d, kernel_size=1, bias=False)  # (D,H)
        self.text_proj = nn.Linear(text_dim, d, bias=False)

    def forward(self, feat3d: torch.Tensor, sel_text: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = feat3d.shape
        axial_summary = feat3d.amax(dim=2)  # [B,C,H,W]
        sagittal_summary = feat3d.amax(dim=3)  # [B,C,D,W]
        coronal_summary = feat3d.amax(dim=4)  # [B,C,D,H]

        ax = self.ax_proj(axial_summary).flatten(2)  # [B,d,H*W]
        sa = self.sa_proj(sagittal_summary).flatten(2)  # [B,d,D*W]
        co = self.co_proj(coronal_summary).flatten(2)  # [B,d,D*H]

        k = F.normalize(self.text_proj(sel_text), dim=-1)  # [B,d]
        k = k.unsqueeze(-1)  # [B,d,1]

        # [B,d,n] x [B,d,1] -> [B,n]
        ax_ker = F.softmax((ax * k).sum(dim=1), dim=-1).view(B, 1, H, W)
        sa_ker = F.softmax((sa * k).sum(dim=1), dim=-1).view(B, 1, D, W)
        co_ker = F.softmax((co * k).sum(dim=1), dim=-1).view(B, 1, D, H)

        ax_w_3d = feat3d * ax_ker.unsqueeze(2)
        sa_w_3d = feat3d * sa_ker.unsqueeze(3)
        co_w_3d = feat3d * co_ker.unsqueeze(4)
        return (1*feat3d + 1*ax_w_3d + 1*sa_w_3d + 1*co_w_3d) / 4.00
