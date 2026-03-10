# networks/UXNet_3D/tri_plane_text_fusion.py
import torch, torch.nn as nn, torch.nn.functional as F

class TriPlaneTextFusion3D(nn.Module):
    def __init__(self, c_in, txt_dim, d_model=256, heads=4, K=12,
                 mlp_ratio=2, fuse_mode="weighted", use_post3d=True, init_gate=0.0):
        super().__init__()
        self.K, self.fuse_mode, self.use_post3d = K, fuse_mode, use_post3d

        # Q/K/V proj
        self.q_proj = nn.Linear(c_in, d_model)
        self.k_proj = nn.Linear(txt_dim, d_model)
        self.v_proj = nn.Linear(txt_dim, d_model)

        self.norm_q = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, heads, batch_first=True)
        self.ffn  = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, mlp_ratio*d_model),
            nn.GELU(),
            nn.Linear(mlp_ratio*d_model, d_model),
        )
        self.back = nn.Linear(d_model, c_in)

        if fuse_mode == "weighted":
            self.w_dh = nn.Parameter(torch.tensor(1.0))
            self.w_hw = nn.Parameter(torch.tensor(1.0))
            self.w_dw = nn.Parameter(torch.tensor(1.0))
            self.fuse_conv = None
        elif fuse_mode == "conv1x1x1":
            self.fuse_conv = nn.Conv3d(3*c_in, c_in, kernel_size=1)
        else:
            raise ValueError

        if use_post3d:
            self.post3d = nn.Sequential(
                nn.Conv3d(c_in, c_in, 3, padding=1), nn.GELU(),
                nn.Conv3d(c_in, c_in, 1)
            )


        self.g_spa = nn.Parameter(torch.tensor(float(init_gate)))
        nn.init.zeros_(self.back.weight)
        nn.init.zeros_(self.back.bias)


        self._text = None  # (B, L, txt_dim) or None

    @torch.no_grad()
    def set_text(self, T):   # T: (B, L, txt_dim) or (B, txt_dim)
        if T is None:
            self._text = None
        else:
            if T.dim() == 2:  # (B, d)
                T = T.unsqueeze(1)
            self._text = T

    def _tri_planes(self, X):
        X_DH = X.mean(dim=4)  # W avg -> (B,C,D,H)
        X_HW = X.mean(dim=2)  # D avg -> (B,C,H,W)
        X_DW = X.mean(dim=3)  # H avg -> (B,C,D,W)
        return X_DH, X_HW, X_DW

    def _pool_KK(self, X2):
        return F.adaptive_avg_pool2d(X2, (self.K, self.K))

    def _lift_DH(self, Z2, D,H,W):  # (D,H) plane -> broadcast over W
        z3 = Z2.unsqueeze(4)
        return F.interpolate(z3, size=(D,H,W), mode="trilinear", align_corners=False)

    def _lift_HW(self, Z2, D,H,W):  # (H,W) plane -> broadcast over D
        z3 = Z2.unsqueeze(2)
        return F.interpolate(z3, size=(D,H,W), mode="trilinear", align_corners=False)

    def _lift_DW(self, Z2, D,H,W):  # (D,W) plane -> broadcast over H
        z3 = Z2.unsqueeze(3)
        return F.interpolate(z3, size=(D,H,W), mode="trilinear", align_corners=False)

    def forward(self, X):  # keep signature simple
        if (self._text is None) or (X is None):
            return X
        B,C,D,H,W = X.shape
        K = self.K

        # 1) tri-plane -> KxK
        X_DH, X_HW, X_DW = self._tri_planes(X)
        P_DH, P_HW, P_DW = self._pool_KK(X_DH), self._pool_KK(X_HW), self._pool_KK(X_DW)

        # 2) tokens: (B, 3K^2, C)
        Q = torch.stack([P_DH, P_HW, P_DW], dim=1)      # (B,3,C,K,K)
        Q = Q.permute(0,1,3,4,2).reshape(B, 3*K*K, C)

        # 3) CA + FFN
        q = self.q_proj(Q)
        k = self.k_proj(self._text)
        v = self.v_proj(self._text)
        h,_ = self.attn(self.norm_q(q), k, v)
        z = q + h
        z = z + self.ffn(z)


        Z = self.back(z).reshape(B, 3, K, K, C).permute(0,1,4,2,3)  # (B,3,C,K,K)
        Z_DH, Z_HW, Z_DW = Z[:,0], Z[:,1], Z[:,2]


        Y_DH = self._lift_DH(Z_DH, D,H,W)
        Y_HW = self._lift_HW(Z_HW, D,H,W)
        Y_DW = self._lift_DW(Z_DW, D,H,W)


        if self.fuse_mode == "weighted":
            w1 = torch.relu(self.w_dh); w2 = torch.relu(self.w_hw); w3 = torch.relu(self.w_dw)
            Y = (w1*Y_DH + w2*Y_HW + w3*Y_DW) / (w1+w2+w3+1e-6)
        else:
            Y = torch.cat([Y_DH, Y_HW, Y_DW], dim=1)
            Y = self.fuse_conv(Y)

        if self.use_post3d:
            Y = self.post3d(Y)

        g = torch.clamp(self.g_spa, 0.0, 1.0)
        return X + g*Y
