# DiffResCrime.py
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import einsum
from inspect import isfunction
from typing import Optional

# -----------------------
# Utility
# -----------------------
def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

class ModelMeanType:
    START_X = 0
    EPSILON = 1

# -----------------------
# create_lambda_schedule
# -----------------------
def create_lambda_schedule(n_T: int, lambda_start: float = 0.001, lambda_end: float = 0.999, power: float = 2.0, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    稳健返回长度 n_T 的 sqrt_lambdas(torch.float32)。
    - 在 log 空间线性插值，避免直接 b0 ** large_exp 导致 overflow。
    - 保证 beta_t 在 [0,1]。
    """
    eps = 1e-12
    lambda_start = float(max(lambda_start, eps))
    lambda_end = float(min(max(lambda_end, eps), 1.0 - eps))
    # ts in [0,1]
    ts = np.linspace(0.0, 1.0, n_T, endpoint=True, dtype=np.float64)
    beta_t = np.clip(ts ** float(power), 0.0, 1.0)  # in [0,1]

    sqrt_lambda_start = math.sqrt(lambda_start)
    sqrt_lambda_end = math.sqrt(lambda_end)

    # log-space interpolation: log(sqrt_lambda) = log(start) + beta*(log(end)-log(start))
    log_start = math.log(max(sqrt_lambda_start, eps))
    log_end = math.log(max(sqrt_lambda_end, eps))
    log_vals = log_start + beta_t * (log_end - log_start)
    sqrt_lambda = np.exp(log_vals).astype(np.float32)

    t = torch.from_numpy(sqrt_lambda)  # CPU tensor
    if device is not None:
        t = t.to(device)
    return t

# -----------------------
# Small NN blocks
# -----------------------
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)
    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * nn.functional.gelu(gate)

class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)
        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )
    def forward(self, x):
        return self.net(x)

class FeatureEmbedder(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(FeatureEmbedder, self).__init__()
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.fc = nn.Sequential(
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        )
    def forward(self, x):
        return self.fc(x)

# -----------------------
# Time-step aware fusion
# -----------------------

class TemporalCrossAttentionFuser(nn.Module):
    def __init__(self, query_dim, x_h_dim, heads=1, dim_head=256, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.t_transform = nn.Linear(query_dim, inner_dim)

        self.key_transform_x = nn.Linear(x_h_dim, inner_dim, bias=False)
        self.value_transform_x = nn.Linear(x_h_dim, inner_dim, bias=False)
        self.layerNorm_x = nn.LayerNorm(x_h_dim)

        self.key_transform_H = nn.Linear(x_h_dim, inner_dim, bias=False)
        self.value_transform_H = nn.Linear(x_h_dim, inner_dim, bias=False)
        self.layerNorm_H = nn.LayerNorm(x_h_dim)

        self.key_transform_S = nn.Linear(x_h_dim, inner_dim, bias=False)
        self.value_transform_S = nn.Linear(x_h_dim, inner_dim, bias=False)
        self.layerNorm_S = nn.LayerNorm(x_h_dim)

        self.key_transform_poi = nn.Linear(x_h_dim, inner_dim, bias=False)
        self.value_transform_poi = nn.Linear(x_h_dim, inner_dim, bias=False)
        self.layerNorm_poi = nn.LayerNorm(x_h_dim)

        self.key_transform_M = nn.Linear(x_h_dim, inner_dim, bias=False)
        self.value_transform_M = nn.Linear(x_h_dim, inner_dim, bias=False)
        self.layerNorm_M = nn.LayerNorm(x_h_dim)

        self.ff = FeedForward(inner_dim, mult=1, glu=True)
        self.layerNorm_result = nn.LayerNorm(inner_dim)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, x_h_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, h, s, poi, m, t, imageWidth):
        # x,h,s,poi : [B, C, H, W]
        # t: [B, 1] or [B]
        assert x.shape[2] == x.shape[3] == imageWidth
        assert h.shape[2] == h.shape[3] == imageWidth
        assert s.shape[2] == s.shape[3] == imageWidth
        assert poi.shape[2] == poi.shape[3] == imageWidth
        assert m.shape[2] == m.shape[3] == imageWidth

        # prepare query
        q = self.t_transform(t).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, imageWidth, imageWidth)
        query = rearrange(q, 'b c h w -> b (h w) c')

        x = rearrange(x, 'b c h w -> b (h w) c')
        h = rearrange(h, 'b c h w -> b (h w) c')
        s = rearrange(s, 'b c h w -> b (h w) c')
        poi = rearrange(poi, 'b c h w -> b (h w) c')
        m = rearrange(m, 'b c h w -> b (h w) c')

        x = self.layerNorm_x(x)
        h = self.layerNorm_H(h)
        s = self.layerNorm_S(s)
        poi = self.layerNorm_poi(poi)
        m = self.layerNorm_M(m)

        k_x = self.key_transform_x(x); v_x = self.value_transform_x(x)
        k_h = self.key_transform_H(h); v_h = self.value_transform_H(h)
        k_s = self.key_transform_S(s); v_s = self.value_transform_S(s)
        k_poi = self.key_transform_poi(poi); v_poi = self.value_transform_poi(poi)
        k_m = self.key_transform_M(m); v_m = self.value_transform_M(m)

        sim_x = einsum('b i d, b j d -> b i j', query, k_x) * self.scale
        sim_h = einsum('b i d, b j d -> b i j', query, k_h) * self.scale
        sim_s = einsum('b i d, b j d -> b i j', query, k_s) * self.scale
        sim_poi = einsum('b i d, b j d -> b i j', query, k_poi) * self.scale
        sim_m = einsum('b i d, b j d -> b i j', query, k_m) * self.scale

        combined_keys = torch.cat([sim_x, sim_h, sim_s, sim_poi, sim_m], dim=-1)
        combined_scores = combined_keys.softmax(dim=-1)

        split_size = combined_scores.size(-1) // 5
        attn_scores_x, attn_scores_H, attn_scores_S, attn_scores_poi, attn_scores_M = torch.split(combined_scores, split_size, dim=-1)

        out_x = einsum('b i j, b j d -> b i d', attn_scores_x, v_x)
        out_h = einsum('b i j, b j d -> b i d', attn_scores_H, v_h)
        out_s = einsum('b i j, b j d -> b i d', attn_scores_S, v_s)
        out_poi = einsum('b i j, b j d -> b i d', attn_scores_poi, v_poi)
        out_m = einsum('b i j, b j d -> b i d', attn_scores_M, v_m)

        x = self.ff(self.layerNorm_result(out_x + out_h + out_s + out_poi + out_m))
        x = self.to_out(x)
        return rearrange(x, 'b (h w) c -> b c h w', h=imageWidth)

# -----------------------
# Residual block, SE block, Down/Up blocks (copied/adapted)
# -----------------------
class ResConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, is_res: bool = False):
        super().__init__()
        self.same_channels = in_channels == out_channels
        self.is_res = is_res
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_res:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            if self.same_channels:
                out = x + x2
            else:
                out = x1 + x2
            return out / 1.414
        else:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            return x2

class CondSEBlock(nn.Module):
    def __init__(self, channel, cond_dim, reduction=64):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channel, channel // reduction, bias=True)
        self.cond_to_weight = nn.Linear(cond_dim, (channel//reduction) * channel)
        self.cond_to_bias   = nn.Linear(cond_dim, channel)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    def forward(self, x, cond):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.relu(self.fc1(y))
        w = self.cond_to_weight(cond)
        b2 = self.cond_to_bias(cond)
        w = w.view(b, self.fc1.out_features, c)
        y = torch.bmm(y.unsqueeze(1), w).squeeze(1) + b2
        y = self.sigmoid(y).view(b, c, 1, 1)
        return x * y

# The BlendDownsampling / BlendUpsampling classes are kept intact with minor formatting.
# For brevity, we include them as in your original code (omitted here would break STFusionNet).
# We'll include DownBlockLevel1..4 and UpBlockLevel1..4 exactly as before.

class DownBlockLevel1(nn.Module):
    def __init__(self, in_channels, out_channels, context_channels, his_channel, his_size, poi_channels, x_dim):
        super(DownBlockLevel1, self).__init__()
        self.out_channels = out_channels
        self.timeEmb = FeatureEmbedder(1, out_channels)
        self.conv1 = nn.Sequential(
            nn.BatchNorm2d(num_features=in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1)
        )
        self.conEmbMap = nn.Sequential(
            nn.BatchNorm2d(num_features=context_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=context_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1)
        )
        self.conEmbSatellite = nn.Sequential(
            nn.BatchNorm2d(num_features=context_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=context_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1)
        )
        self.poiDown = nn.Sequential(
            nn.BatchNorm2d(num_features=poi_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=poi_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1),
            CondSEBlock(out_channels, x_dim)
        )
        self.TAMF = TemporalCrossAttentionFuser(query_dim=1, x_h_dim=out_channels)
        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        )
        self.convInit = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=2)
        self.hisDown = nn.Sequential(
            nn.BatchNorm2d(num_features=in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        )
        self.w1 = nn.Parameter(torch.randn(his_channel, his_size, his_size))
        self.w2 = nn.Parameter(torch.randn(his_channel, his_size, his_size))
        self.b = nn.Parameter(torch.randn(his_channel, his_size, his_size))

    def forward(self, x, s, m, his, poi, t, cond):
        xInit = self.convInit(x)
        x = self.conv1(x)
        condMapEmb = self.conEmbMap(m)
        condSatelliteEmb = self.conEmbSatellite(s)
        his = self.hisDown(his)
        poi = self.poiDown[0](poi)  # BatchNorm
        poi = self.poiDown[1](poi)  # ReLU
        poi = self.poiDown[2](poi)  # Conv
        poi = self.poiDown[3](poi, cond)  # CondSEBlock
        x = self.TAMF(x, his, condSatelliteEmb, poi, condMapEmb, t, imageWidth=8) + x
        x = self.conv2(x)
        x = x + xInit
        feg = nn.Sigmoid()(torch.matmul(self.w1, x) + torch.matmul(self.w2, his) + self.b)
        return torch.multiply(x, feg) + torch.multiply(his, 1 - feg), his, poi

class DownBlockLevel2(nn.Module):
    def __init__(self, in_channels, out_channels, context_channels, his_channel, his_size, poi_channels, x_dim):
        super(DownBlockLevel2, self).__init__()
        self.out_channels = out_channels
        self.timeEmb = FeatureEmbedder(1, out_channels)
        self.conv1 = nn.Sequential(
            nn.BatchNorm2d(num_features=in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1)
        )
        self.conEmbMap = nn.Sequential(
            nn.BatchNorm2d(num_features=context_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=context_channels, out_channels=context_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=context_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=context_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1)
        )
        self.conEmbSatellite = nn.Sequential(
            nn.BatchNorm2d(num_features=context_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=context_channels, out_channels=context_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=context_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=context_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1)
        )
        self.poiDown = nn.Sequential(
            nn.BatchNorm2d(num_features=poi_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=poi_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1),
            CondSEBlock(out_channels, x_dim)
        )
        self.TAMF = TemporalCrossAttentionFuser(query_dim=1, x_h_dim=out_channels)
        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        )
        self.convInit = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=2)
        self.hisDown = nn.Sequential(
            nn.BatchNorm2d(num_features=in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        )
        self.w1 = nn.Parameter(torch.randn(his_channel, his_size, his_size))
        self.w2 = nn.Parameter(torch.randn(his_channel, his_size, his_size))
        self.b = nn.Parameter(torch.randn(his_channel, his_size, his_size))

    def forward(self, x, s, m, his, poi, t, cond):
        xInit = self.convInit(x)
        x = self.conv1(x)
        condMapEmb = self.conEmbMap(m)
        condSatelliteEmb = self.conEmbSatellite(s)
        his = self.hisDown(his)
        poi = self.poiDown[0](poi)
        poi = self.poiDown[1](poi)
        poi = self.poiDown[2](poi)
        poi = self.poiDown[3](poi, cond)
        x = self.TAMF(x, his, condSatelliteEmb, poi, condMapEmb, t, imageWidth=4) + x
        x = self.conv2(x)
        x = x + xInit
        feg = nn.Sigmoid()(torch.matmul(self.w1, x) + torch.matmul(self.w2, his) + self.b)
        return torch.multiply(x, feg) + torch.multiply(his, 1 - feg), his, poi

class DownBlockLevel3(nn.Module):
    def __init__(self, in_channels, out_channels, context_channels, his_channel, his_size, poi_channels, x_dim):
        super(DownBlockLevel3, self).__init__()
        self.out_channels = out_channels
        self.timeEmb = FeatureEmbedder(1, out_channels)
        self.conv1 = nn.Sequential(
            nn.BatchNorm2d(num_features=in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1)
        )
        self.conEmbMap = nn.Sequential(
            nn.BatchNorm2d(num_features=context_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=context_channels, out_channels=context_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=context_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=context_channels, out_channels=context_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=context_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=context_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1)
        )
        self.conEmbSatellite = nn.Sequential(
            nn.BatchNorm2d(num_features=context_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=context_channels, out_channels=context_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=context_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=context_channels, out_channels=context_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=context_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=context_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1)
        )
        self.poiDown = nn.Sequential(
            nn.BatchNorm2d(num_features=poi_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=poi_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1),
            CondSEBlock(out_channels, x_dim)
        )
        self.TAMF = TemporalCrossAttentionFuser(query_dim=1, x_h_dim=out_channels)
        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        )
        self.convInit = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=2)
        self.hisDown = nn.Sequential(
            nn.BatchNorm2d(num_features=in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        )
        self.w1 = nn.Parameter(torch.randn(his_channel, his_size, his_size))
        self.w2 = nn.Parameter(torch.randn(his_channel, his_size, his_size))
        self.b = nn.Parameter(torch.randn(his_channel, his_size, his_size))

    def forward(self, x, s, m, his, poi, t, cond):
        xInit = self.convInit(x)
        x = self.conv1(x)
        condMapEmb = self.conEmbMap(m)
        condSatelliteEmb = self.conEmbSatellite(s)
        his = self.hisDown(his)
        poi = self.poiDown[0](poi)
        poi = self.poiDown[1](poi)
        poi = self.poiDown[2](poi)
        poi = self.poiDown[3](poi, cond)
        x = self.TAMF(x, his, condSatelliteEmb, poi, condMapEmb, t, imageWidth=2) + x
        x = self.conv2(x)
        x = x + xInit
        feg = nn.Sigmoid()(torch.matmul(self.w1, x) + torch.matmul(self.w2, his) + self.b)
        return torch.multiply(x, feg) + torch.multiply(his, 1 - feg), his, poi

class DownBlockLevel4(nn.Module):
    def __init__(self, in_channels, out_channels, poi_channels, x_dim):
        super(DownBlockLevel4, self).__init__()
        self.out_channels = out_channels
        self.timeEmb = FeatureEmbedder(1, out_channels)
        self.conv1 = nn.Sequential(
            nn.BatchNorm2d(num_features=in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1)
        )
        self.hisDown = nn.Sequential(
            nn.BatchNorm2d(num_features=in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1)
        )
        self.poiDown = nn.Sequential(
            nn.BatchNorm2d(num_features=poi_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=poi_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1),
            CondSEBlock(out_channels, x_dim)
        )

    def forward(self, x, his, poi, cond):
        x = self.conv1(x)
        his = self.hisDown(his)
        poi = self.poiDown[0](poi)
        poi = self.poiDown[1](poi)
        poi = self.poiDown[2](poi)
        poi = self.poiDown[3](poi, cond)
        return x, his, poi

class UpBlockLevel4(nn.Module):
    def __init__(self, in_channels, out_channels, poi_channels, x_dim):
        super(UpBlockLevel4, self).__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.hisUp = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.poiUp = nn.Sequential(
            nn.ConvTranspose2d(poi_channels, out_channels, 4, 2, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            CondSEBlock(out_channels, x_dim)
        )

    def forward(self, x, his, poi, cond):
        x = self.conv(x)
        his = self.hisUp(his)
        poi = self.poiUp[0](poi)
        poi = self.poiUp[1](poi)
        poi = self.poiUp[2](poi)
        poi = self.poiUp[3](poi, cond)
        return x, his, poi

class UpBlockLevel3(nn.Module):
    def __init__(self, in_channels, out_channels, context_channels, his_channel, his_size, poi_channels, x_dim):
        super(UpBlockLevel3, self).__init__()
        self.out_channels = out_channels
        self.timeEmb = FeatureEmbedder(1, out_channels)
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
            ResConvBlock(out_channels, out_channels),
            ResConvBlock(out_channels, out_channels)
        )
        self.conEmbMap = nn.Sequential(
            nn.BatchNorm2d(num_features=context_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=context_channels, out_channels=context_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=context_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=context_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1)
        )
        self.conEmbSatellite = nn.Sequential(
            nn.BatchNorm2d(num_features=context_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=context_channels, out_channels=context_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=context_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=context_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1)
        )
        self.poiUp = nn.Sequential(
            nn.ConvTranspose2d(poi_channels, out_channels, 4, 2, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            CondSEBlock(out_channels, x_dim)
        )
        self.TAMF = TemporalCrossAttentionFuser(query_dim=1, x_h_dim=out_channels)
        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        )
        self.convInit = nn.ConvTranspose2d(in_channels, out_channels, 2, 2)
        self.hisUp = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
            ResConvBlock(out_channels, out_channels),
            ResConvBlock(out_channels, out_channels),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        )
        self.w1 = nn.Parameter(torch.randn(out_channels, his_size, his_size))
        self.w2 = nn.Parameter(torch.randn(out_channels, his_size, his_size))
        self.b = nn.Parameter(torch.randn(out_channels, his_size, his_size))

    def forward(self, x, skip, s, m, his, poi, t, cond):
        x = torch.cat((x, skip), 1)
        xInit = self.convInit(x)
        x = self.conv1(x)
        condMapEmb = self.conEmbMap(m)
        condSatelliteEmb = self.conEmbSatellite(s)
        his = self.hisUp(his)
        poi = self.poiUp[0](poi)
        poi = self.poiUp[1](poi)
        poi = self.poiUp[2](poi)
        poi = self.poiUp[3](poi, cond)
        x = self.TAMF(x, his, condSatelliteEmb, poi, condMapEmb, t, imageWidth=4) + x
        x = self.conv2(x)
        x = x + xInit
        feg = nn.Sigmoid()(torch.matmul(self.w1, x) + torch.matmul(self.w2, his) + self.b)
        return torch.multiply(x, feg) + torch.multiply(his, 1 - feg), his, poi

class UpBlockLevel2(nn.Module):
    def __init__(self, in_channels, out_channels, context_channels, his_channel, his_size, poi_channels, x_dim):
        super(UpBlockLevel2, self).__init__()
        self.out_channels = out_channels
        self.timeEmb = FeatureEmbedder(1, out_channels)
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
            ResConvBlock(out_channels, out_channels),
            ResConvBlock(out_channels, out_channels)
        )
        self.conEmbMap = nn.Sequential(
            nn.BatchNorm2d(num_features=context_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=context_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1)
        )
        self.conEmbSatellite = nn.Sequential(
            nn.BatchNorm2d(num_features=context_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=context_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1)
        )
        self.poiUp = nn.Sequential(
            nn.ConvTranspose2d(poi_channels, out_channels, 4, 2, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            CondSEBlock(out_channels, x_dim)
        )
        self.TAMF = TemporalCrossAttentionFuser(query_dim=1, x_h_dim=out_channels)
        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        )
        self.convInit = nn.ConvTranspose2d(in_channels, out_channels, 2, 2)
        self.hisUp = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
            ResConvBlock(out_channels, out_channels),
            ResConvBlock(out_channels, out_channels),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        )
        self.w1 = nn.Parameter(torch.randn(out_channels, his_size, his_size))
        self.w2 = nn.Parameter(torch.randn(out_channels, his_size, his_size))
        self.b = nn.Parameter(torch.randn(out_channels, his_size, his_size))

    def forward(self, x, skip, s, m, his, poi, t, cond):
        x = torch.cat((x, skip), 1)
        xInit = self.convInit(x)
        x = self.conv1(x)
        condMapEmb = self.conEmbMap(m)
        condSatelliteEmb = self.conEmbSatellite(s)
        his = self.hisUp(his)
        poi = self.poiUp[0](poi)
        poi = self.poiUp[1](poi)
        poi = self.poiUp[2](poi)
        poi = self.poiUp[3](poi, cond)
        x = self.TAMF(x, his, condSatelliteEmb, poi, condMapEmb, t, imageWidth=8) + x
        x = self.conv2(x)
        x = x + xInit
        feg = nn.Sigmoid()(torch.matmul(self.w1, x) + torch.matmul(self.w2, his) + self.b)
        return torch.multiply(x, feg) + torch.multiply(his, 1 - feg), his, poi

class UpBlockLevel1(nn.Module):
    def __init__(self, in_channels, out_channels, context_channels, his_channel, his_size, poi_channels, x_dim):
        super(UpBlockLevel1, self).__init__()
        self.out_channels = out_channels
        self.timeEmb = FeatureEmbedder(1, out_channels)
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
            ResConvBlock(out_channels, out_channels),
            ResConvBlock(out_channels, out_channels)
        )
        self.conEmbMap = ResConvBlock(context_channels, out_channels)
        self.conEmbSatellite = ResConvBlock(context_channels, out_channels)
        self.poiUp = nn.Sequential(
            nn.ConvTranspose2d(poi_channels, out_channels, 4, 2, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            CondSEBlock(out_channels, x_dim)
        )
        self.TAMF = TemporalCrossAttentionFuser(query_dim=1, x_h_dim=out_channels)
        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        )
        self.convInit = nn.ConvTranspose2d(in_channels, out_channels, 2, 2)
        self.hisUp = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
            ResConvBlock(out_channels, out_channels),
            ResConvBlock(out_channels, out_channels),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        )
        self.w1 = nn.Parameter(torch.randn(out_channels, his_size, his_size))
        self.w2 = nn.Parameter(torch.randn(out_channels, his_size, his_size))
        self.b = nn.Parameter(torch.randn(out_channels, his_size, his_size))

    def forward(self, x, skip, s, m, his, poi, t, cond):
        x = torch.cat((x, skip), 1)
        xInit = self.convInit(x)
        x = self.conv1(x)
        condMapEmb = self.conEmbMap(m)
        condSatelliteEmb = self.conEmbSatellite(s)
        his = self.hisUp(his)
        poi = self.poiUp[0](poi)
        poi = self.poiUp[1](poi)
        poi = self.poiUp[2](poi)
        poi = self.poiUp[3](poi, cond)
        x = self.TAMF(x, his, condSatelliteEmb, poi, condMapEmb, t, imageWidth=16) + x
        x = self.conv2(x)
        x = x + xInit
        feg = nn.Sigmoid()(torch.matmul(self.w1, x) + torch.matmul(self.w2, his) + self.b)
        return torch.multiply(x, feg) + torch.multiply(his, 1 - feg), his, poi

# -----------------------
# STFusionNet backbone (adapted from provided code)
# -----------------------
class STFusionNet(nn.Module):
    def __init__(self, in_channels, n_feat, context_out_channels=64, poi_channels=13, x_dim=256):
        super(STFusionNet, self).__init__()
        self.in_channels = in_channels
        self.n_feat = n_feat
        self.contextDim = 32
        self.x_dim = x_dim
        self.encoderWideMap = nn.Sequential(
            nn.Conv2d(3, context_out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(context_out_channels),
            nn.ReLU(),
            nn.Conv2d(context_out_channels, context_out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(context_out_channels),
            nn.ReLU(),
            nn.Conv2d(context_out_channels, context_out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(context_out_channels),
            nn.ReLU(),
            nn.Conv2d(context_out_channels, context_out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(context_out_channels),
            nn.ReLU(),
            nn.Conv2d(context_out_channels, context_out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(context_out_channels),
            nn.ReLU(),
            nn.Conv2d(context_out_channels, context_out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(context_out_channels),
            nn.ReLU(),
            nn.Conv2d(context_out_channels, context_out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(context_out_channels),
            nn.ReLU(),
            nn.Conv2d(context_out_channels, context_out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(context_out_channels),
            nn.ReLU(),
            nn.Conv2d(context_out_channels, context_out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(context_out_channels),
            nn.ReLU(),
            nn.Conv2d(context_out_channels, context_out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(context_out_channels),
            nn.ReLU(),
        )
        self.encoderWideSatellite = nn.Sequential(
            nn.Conv2d(3, context_out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(context_out_channels),
            nn.ReLU(),
            nn.Conv2d(context_out_channels, context_out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(context_out_channels),
            nn.ReLU(),
            nn.Conv2d(context_out_channels, context_out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(context_out_channels),
            nn.ReLU(),
            nn.Conv2d(context_out_channels, context_out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(context_out_channels),
            nn.ReLU(),
            nn.Conv2d(context_out_channels, context_out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(context_out_channels),
            nn.ReLU(),
            nn.Conv2d(context_out_channels, context_out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(context_out_channels),
            nn.ReLU(),
            nn.Conv2d(context_out_channels, context_out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(context_out_channels),
            nn.ReLU(),
            nn.Conv2d(context_out_channels, context_out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(context_out_channels),
            nn.ReLU(),
            nn.Conv2d(context_out_channels, context_out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(context_out_channels),
            nn.ReLU(),
            nn.Conv2d(context_out_channels, context_out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(context_out_channels),
            nn.ReLU(),
        )
        self.poi_encoder = ResConvBlock(poi_channels, n_feat, is_res=True)
        self.init_conv = ResConvBlock(in_channels, n_feat, is_res=True)
        self.init_his = ResConvBlock(in_channels, n_feat, is_res=True)
        self.down1 = DownBlockLevel1(n_feat, n_feat, context_out_channels, n_feat, 8, n_feat, x_dim)
        self.down2 = DownBlockLevel2(n_feat, 2 * n_feat, context_out_channels, 2 * n_feat, 4, n_feat, x_dim)
        self.down3 = DownBlockLevel3(2 * n_feat, 4 * n_feat, context_out_channels, 4 * n_feat, 2, 2 * n_feat, x_dim)
        self.down4 = DownBlockLevel4(4 * n_feat,  8 * n_feat, 4 * n_feat, x_dim)
        self.up4 = UpBlockLevel4(8 * n_feat, 4 * n_feat, 8 * n_feat, x_dim)
        self.up3 = UpBlockLevel3(8 * n_feat, 2 * n_feat, context_out_channels, 2  * n_feat, 4, 4 * n_feat, x_dim)
        self.up2 = UpBlockLevel2(4 * n_feat, n_feat, context_out_channels, n_feat, 8, 2 * n_feat, x_dim)
        self.up1 = UpBlockLevel1(2 * n_feat, n_feat , context_out_channels, n_feat, 16, n_feat, x_dim)
        
        self.out = nn.Sequential(
            nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1),
            nn.GroupNorm(8, n_feat),
            nn.ReLU(),
            nn.Conv2d(n_feat, 2 * self.in_channels, 3, 1, 1),
            # nn.Conv2d(n_feat, self.in_channels, 3, 1, 1),
        )

        self.hisOut = nn.Sequential(
            nn.Conv2d(2 * n_feat, n_feat, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, n_feat),
            nn.ReLU(),
            ResConvBlock(n_feat, n_feat, is_res=True),
            nn.GroupNorm(8, n_feat),
            nn.ReLU(),
            nn.Conv2d(n_feat,  self.in_channels, kernel_size=3, stride=1, padding=1)
        )

        self.cond_embed = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(n_feat, x_dim),
            nn.ReLU(),
            nn.Linear(x_dim, x_dim)
        )

    def forward(self, x, c, x_hist, poi, t, context_mask):
        # t : [B] or [B,1] (fraction or scalar); context_mask:  [B]
        # Apply mask semantics: in your earlier code context_mask was inverted; we keep same invert behavior
        t = t.unsqueeze(-1) if t.ndim == 1 else t
        context_mask = context_mask.view(-1, 1, 1, 1)
        context_mask = 1 - context_mask  # invert: 0->1,1->0

        c = c * context_mask
        x_hist = x_hist * context_mask
        poi = poi * context_mask

        # satellite expects HWC input in earlier code: here keep same transform
        c =  c.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
        # 分离出satellite和map模态
        csatellite = c[:, 0:3, :, :]  # 假设前3通道是satellite
        cmap = c[:, 3:6, :, :]  # 假设后3通道是map
        csatellite = self.encoderWideSatellite(csatellite)
        cmap = self.encoderWideMap(cmap)  # 新增Map模态编码

        x = self.init_conv(x)
        x_hist = self.init_his(x_hist)
        poi = self.poi_encoder(poi)
        
        cond = self.cond_embed(x_hist)
        
        down1, x_hist1, poi1 = self.down1(x, csatellite, cmap, x_hist, poi, t, cond)
        down2, x_hist2, poi2 = self.down2(down1, csatellite, cmap, x_hist1, poi1, t, cond)
        down3, x_hist3, poi3 = self.down3(down2, csatellite, cmap, x_hist2, poi2, t, cond)
        down4, x_hist4, poi4 = self.down4(down3, x_hist3, poi3, cond)
        up4, x_hist5, poi5 =  self.up4(down4, x_hist4, poi4, cond)
        up3, x_hist6, poi6 = self.up3(up4, down3, csatellite, cmap, torch.cat((x_hist3, x_hist5), 1), poi5, t, cond)
        up2, x_hist7, poi7 = self.up2(up3, down2, csatellite, cmap, torch.cat((x_hist2, x_hist6), 1), poi6, t, cond)
        up1, x_hist8, poi8 = self.up1(up2, down1, csatellite, cmap, torch.cat((x_hist1, x_hist7), 1), poi7, t, cond)
        out = self.out(torch.cat((up1, x), 1))

        if x_hist8.shape[2:] != x_hist.shape[2:]:
            x_hist8 = F.interpolate(x_hist8, size=x_hist.shape[2:], mode='bilinear', align_corners=False)
        history_cat = torch.cat((x_hist, x_hist8), dim=1)  # [B, 2*n_feat, H, W]
        x_hist = self.hisOut(history_cat)  # [B, n_feat, H, W]
        pred = out[:, 1:2, :, :]
        wgt = out[:, 0:1, :, :]
        wgt = nn.Sigmoid()(wgt)
        out = pred * wgt + x_hist * (1.0 - wgt)

        return out

    # compatibility wrapper: explicit predict_x_future method
    def predict_x_future(self, x_t, c, x_hist, poi, t_idx, context_mask):
        # convert t_idx to fractional t if needed
        t_frac = (t_idx.float() / max(1, self.n_T if hasattr(self, 'n_T') else 1)).to(x_t.device)
        # The forward expects t shaped appropriately; if t_idx is scalar or vector, adapt
        return self.forward(x_t, c, x_hist, poi, t_frac, context_mask)

# -----------------------
# FeatureBridge for eps -> x_future conversion
# -----------------------
class FeatureBridge(nn.Module):
    """
    Wrap an original backbone that predicts epsilon (noise) and convert to x_future estimates.
    Usage:
    adapter = FeatureBridge(orig_net, diffusion_module)
    diffusion = DiffRes(adapter, ...)
    """
    def __init__(self, orig_net: nn.Module, diffusion_module):
        super().__init__()
        self.orig_net = orig_net
        self.diff = diffusion_module

    def predict_x_future(self, x_t, cond, x_hist, poi, t_idx, context_mask):
        # orig_net is expected to return eps_pred (same shape as x_t)
        t_frac = (t_idx.float() / max(1, self.diff.n_T)).to(x_t.device)
        eps_pred = self.orig_net(x_t, cond, x_hist, poi, t_frac, context_mask)
        # index sqrt_lambda and lambda
        if t_idx.dim() == 0:
            t_idx = t_idx.unsqueeze(0)
        t_idx = t_idx.long().to(x_t.device)
        sqrt_lambda_t = self.diff.sqrt_lambdas[t_idx].view(-1, *([1] * (x_t.dim() - 1))).to(x_t.device)
        lambda_t = self.diff.lambdas[t_idx].view(-1, *([1] * (x_t.dim() - 1))).to(x_t.device)
        sigma_t = self.diff.gamma * sqrt_lambda_t # Updated kappa -> gamma
        # Rearranged formula to solve x_future: x_t = x_hist + (1-lambda_t)*(x_future - x_hist) + sigma_t * z
        # => x_t = x_hist + (1-lambda_t)*x_future - (1-lambda_t)*x_hist + sigma_t * z
        # => x_t = lambda_t * x_hist + (1-lambda_t) * x_future + sigma_t * z
        # => (1-lambda_t) * x_future = x_t - lambda_t * x_hist - sigma_t * z
        # => x_future = (x_t - lambda_t * x_hist - sigma_t * z) / (1 - lambda_t)
        # Here z approximated by eps_pred
        # x_future_hat = (x_t - lambda_t * x_hist - sigma_t * eps_pred) / (1.0 - lambda_t)
        # Simplified form from Methodology.md Appendix A.3 (Eq A.14 rearranged):
        # x_t = x_hist + (1-lambda_t) * Delta_res + sigma_t * eps => Delta_res = (x_t - x_hist - sigma_t * eps) / (1-lambda_t)
        # x_future = Delta_res + x_hist => x_future = (x_t - x_hist - sigma_t * eps) / (1-lambda_t) + x_hist
        # x_future_hat = (x_t - sigma_t * eps_pred + lambda_t * x_hist) / (1.0 + lambda_t) # Incorrect from old code
        # Correcting based on forward process: x_t = x_hist + (1-lambda_t)*(x_future - x_hist) + sigma*eps
        # => x_t - x_hist = (1-lambda_t)*(x_future - x_hist) + sigma*eps
        # => (x_t - x_hist - sigma*eps) / (1-lambda_t) = x_future - x_hist
        # => x_future = x_hist + (x_t - x_hist - sigma*eps) / (1-lambda_t)
        # => x_future = ( (1-lambda_t)*x_hist + x_t - x_hist - sigma*eps ) / (1-lambda_t)
        # => x_future = ( x_t - lambda_t*x_hist - sigma*eps ) / (1-lambda_t)
        
        x_future_hat = (x_t - lambda_t * x_hist - sigma_t * eps_pred) / (1.0 - lambda_t) # Corrected formula and name
        # x_future_hat = (x_t - sigma_t * eps_pred + lambda_t * x_hist) / (1.0 + lambda_t)
        return x_future_hat

# -----------------------
# DiffRes
# -----------------------
class DiffRes(nn.Module):
    def __init__(self, backbone: nn.Module, n_T: int = 15, device: str = "cpu", drop_prob: float = 0.1, gamma: float = 1.0, lambda_start: float = 0.001, lambda_end: float = 0.999, power: float = 2.0): # Updated kappa -> gamma
        super().__init__()
        self.n_T = int(n_T)
        self.device = torch.device(device) if isinstance(device, str) else device
        self.drop_prob = float(drop_prob)
        self.gamma = float(gamma) # Updated kappa -> gamma
        # backbone - user may pass a backbone that already implements predict_x_future, or an eps-backbone wrapped in FeatureBridge
        self.backbone = backbone.to(self.device)

        # create schedules and register as buffers (on CPU initially)
        # sqrt_lambdas = create_lambda_schedule(self.n_T, lambda_start=lambda_start, lambda_end=lambda_end, power=power, device=None)  # cpu tensor
         # lambdas = sqrt_lambdas ** 2
        # lambdas_full = torch.cat((torch.tensor([0.0], dtype=torch.float32), lambdas), dim=0)
        # sqrt_lambdas_full = torch.cat((torch.tensor([0.0], dtype=torch.float32), sqrt_lambdas), dim=0)
        # lambdas_prev_full = torch.cat((torch.tensor([0.0], dtype=torch.float32), lambdas[:-1]), dim=0)
        # alpha_full = lambdas_full - lambdas_prev_full

        # ---- START REPLACEMENT ----
        # 生成 sqrt_lambdas（长度 n_T，CPU），然后到 device
        sqrt_lambdas_t = create_lambda_schedule(self.n_T, lambda_start=lambda_start, lambda_end=lambda_end, power=power, device=None)  #  CPU tensor length n_T
        lambdas_t = sqrt_lambdas_t ** 2  # length n_T

        # 构造 length n_T+1 的向量 [0, lambda1, lambda2, ..., lambda_T]
        zero = torch.tensor([0.0], dtype=torch.float32)
        lambdas_full = torch.cat((zero, lambdas_t.to(zero.device)), dim=0).to(self.device)         # length n_T+1, moved to self.device
        sqrt_lambdas_full = torch.cat((zero, sqrt_lambdas_t.to(zero.device)), dim=0).to(self.device)

        # alpha_diff 长度为 n_T: alpha_diff[i-1] = lambdas_full[i] - lambdas_full[i-1], for i=1..n_T
        alpha_diff = lambdas_full[1:] - lambdas_full[:-1]   # length n_T

        # alpha_full 插入前导 0，长度 n_T+1，与 lambdas_full 对齐
        alpha_full = torch.cat((zero.to(self.device), alpha_diff), dim=0)  # length n_T+1
        # ---- END REPLACEMENT ----


        self.register_buffer("lambdas", lambdas_full)       # shape (n_T+1,) # Updated etas -> lambdas
        self.register_buffer("sqrt_lambdas", sqrt_lambdas_full) # Updated sqrt_etas -> sqrt_lambdas
        # self.register_buffer("lambdas_prev", lambdas_prev_full) # Removed as not used directly
        self.register_buffer("alpha", alpha_full) # Updated alpha calculation

        # move model to device (buffers will move with .to)
        self.to(self.device)

    def _predict_x0_from_backbone(self, x_t, cond, x_hist, poi, t_idx, context_mask): # Updated history -> x_hist
        if hasattr(self.backbone, "predict_x_future"): # Updated method name check
            return self.backbone.predict_x_future(x_t, cond, x_hist, poi, t_idx, context_mask) # Updated method name call
        # else assume backbone(x, c, x_hist, poi, t_frac, context_mask) returns x_future directly # Updated history -> x_hist
        t_frac = (t_idx.float() / max(1, self.n_T)).to(x_t.device)
        return self.backbone(x_t, cond, x_hist, poi, t_frac, context_mask) # Updated history -> x_hist

    def q_sample(self, x_future: torch.Tensor, x_hist: torch.Tensor, t: torch.LongTensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor: # Updated x_start -> x_future, history -> x_hist
        B = x_future.shape[0] # Updated x_start -> x_future
        if noise is None:
            noise = torch.randn_like(x_future) # Updated x_start -> x_future
        lambda_t = self.lambdas[t].view(B, *([1] * (x_future.dim() - 1))).to(x_future.device) # Updated etas -> lambdas
        sqrt_lambda_t = self.sqrt_lambdas[t].view_as(lambda_t).to(x_future.device) # Updated sqrt_etas -> sqrt_lambdas
        sigma_t = self.gamma * sqrt_lambda_t # Updated kappa -> gamma
        # x_t = x_future + lambda_t * (x_future - x_hist) + sigma_t * noise # Incorrect old formula
        # Correct formula from Methodology.md Eq 3.2: z_t = x_hist + (1-lambda_t) * Delta_res + sigma_t * noise
        # Where Delta_res = x_future - x_hist
        # => z_t = x_hist + (1-lambda_t) * (x_future - x_hist) + sigma_t * noise
        # => z_t = x_hist + (1-lambda_t)*x_future - (1-lambda_t)*x_hist + sigma_t * noise
        # => z_t = lambda_t * x_hist + (1-lambda_t) * x_future + sigma_t * noise
        
        x_t = lambda_t * x_hist + (1 - lambda_t) * x_future + sigma_t * noise # Corrected formula
        # x_t = x_future + lambda_t * (x_future - x_hist) + sigma_t * noise
        return x_t

    def spatially_weighted_asymmetric_loss(self, pred, target, pos_weight=10.0, asymmetry_alpha=2.0):
        """
        Spatially Weighted Asymmetric MSE (SWAM Loss) for Sparse Crime Prediction.
        1. Spatial Weighting: Heavily penalize errors in regions where crime actually happened (target > 0).
        2. Asymmetry: Penalize under-prediction (missing a crime) more than over-prediction.
        """
        diff = pred - target
        mse = diff ** 2
        
        # 1. Spatial Weighting
        # Identify positive regions (Ground Truth > epsilon).
        # Assuming crime maps are normalized or count-based.
        pos_mask = (target > 1e-4).float()
        
        # Apply weight: pos_weight for positive pixels, 1.0 for background
        spatial_weights = 1.0 + (pos_weight - 1.0) * pos_mask
        
        # 2. Asymmetric Penalty
        # If target > pred (Under-prediction), apply extra penalty alpha.
        # If target <= pred (Over-prediction), weight is 1.0.
        # This forces the model to increase Recall (AP).
        under_prediction_mask = (target > pred).float()
        asymmetric_weights = 1.0 + (asymmetry_alpha - 1.0) * under_prediction_mask
        
        # Combine weights
        total_weights = spatial_weights * asymmetric_weights
        
        weighted_loss = (mse * total_weights).mean()
        return weighted_loss

    def forward(self, x_future: torch.Tensor, cond, x_hist: torch.Tensor, poi) -> torch.Tensor: # Updated x_start -> x_future, history -> x_hist
        B = x_future.shape[0] # Updated x_start -> x_future
        t = torch.randint(1, self.n_T + 1, (B,), device=self.device).long()
        x_t = self.q_sample(x_future, x_hist, t) # Updated x_start -> x_future, history -> x_hist
        context_mask = torch.bernoulli(torch.zeros(B, device=self.device) + self.drop_prob).to(self.device)
        pred_x_future = self._predict_x0_from_backbone(x_t, cond, x_hist, poi, t, context_mask) # Updated x_start -> x_future, history -> x_hist, pred_x0 -> pred_x_future
        
        # # --- 修改开始 ---
        # loss = self.spatially_weighted_asymmetric_loss(pred_x_future, x_future)
        # # --- 修改结束 ---
        
        loss = F.mse_loss(pred_x_future, x_future) # Updated pred_x0 -> pred_x_future, x_start -> x_future
        return loss

    @torch.no_grad()
    def sample(self, n_sample: int, size: tuple, cond: torch.Tensor, x_hist: torch.Tensor, poi: torch.Tensor, guide_w: float = 0.0, device: Optional[torch.device] = None) -> torch.Tensor: # Updated history -> x_hist
        device = device if device is not None else x_hist.device # Updated history -> x_hist
        cond_dup = cond.repeat(2, 1, 1, 1).to(device)
        x_hist_dup = x_hist.repeat(2, 1, 1, 1).to(device) # Updated history -> x_hist
        poi_dup = poi.repeat(2, 1, 1, 1).to(device)
        context_mask = torch.zeros(n_sample, device=device).repeat(2)
        context_mask[n_sample:] = 1.0

        lambda_T = self.lambdas[self.n_T].to(device) # Updated etas -> lambdas
        # sigma_T = self.gamma * torch.sqrt(lambda_T * (1 - self.lambdas[self.n_T - 1])) # From Methodology.md Algo 2, but schedule might not guarantee lambda_{T-1} exists if n_T=1
        sigma_T = self.gamma * lambda_T.sqrt() # Simplified initialization, consistent with old code's intent for t=T
        # x_t = x_hist[:n_sample].to(device) + torch.randn(n_sample, *size, device=device) * (self.gamma * lambda_T.sqrt()) # Incorrect old code
        x_t = x_hist[:n_sample].to(device) + torch.randn(n_sample, *size, device=device) * sigma_T # Corrected initialization from Methodology.md Algo 2 line 7

        for t in range(self.n_T, 0, -1):
            print('\r' + f'sampling timestep {t}', end='')
            t_idx = torch.tensor([t] * (2 * n_sample), device=device).long()
            x_in = x_t.repeat(2, 1, 1, 1)
            pred_all = self._predict_x0_from_backbone(x_in, cond_dup, x_hist_dup, poi_dup, t_idx, context_mask) # Updated history -> x_hist
            pred_cond = pred_all[:n_sample]
            pred_uncond = pred_all[n_sample:]
            pred_x_future = (1 + guide_w) * pred_cond - guide_w * pred_uncond if guide_w != 0.0 else pred_cond # Updated pred_x0 -> pred_x_future

            x_hist0 = x_hist[:n_sample].to(device) # Updated history -> x_hist
            # e0_hat = pred_x_future - x_hist0 # Delta_res_hat

            if t > 1:
                # # lambda_prev = self.lambdas[t-1].view(1,1,1,1).to(device) # Updated etas -> lambdas
                # # alpha_t = (self.lambdas[t] - self.lambdas[t-1]).view(1,1,1,1).to(device) # Updated etas -> lambdas
                # # ========== 修复 1: 修正均值计算中的符号错误 ==========
                # # 原错误代码: mean_prev = pred_x_future +  lambda_prev * e0_hat
                # # 根据前向过程 x_t = x_hist + (1-lambda_t) * (x_future - x_hist) 推导，反向均值应为:
                # # x_{t-1} ~ x_hist + (1-lambda_{t-1}) * (x_hat_future - x_hist)
                # # mean_prev = x_hist0 + (1 - lambda_prev) * e0_hat # e0_hat is Delta_res_hat

                # # ========== 修复 2: 修正噪声项以匹配理论推导 ==========
                # # 原近似代码: z = torch.sqrt(alpha_t) * torch.randn_like(x_t)
                # # 根据 ResShift 论文附录A，后验方差为: γ² * (λ_{t-1} / λ_t) * α_t
                # # 因此，采样时应添加的噪声标准差为: γ * sqrt( (λ_{t-1} * α_t) / λ_t )
                lambda_prev = self.lambdas[t-1].view(1,1,1,1).to(device) # Updated etas -> lambdas
                alpha_t = (self.lambdas[t] - self.lambdas[t-1]).view(1,1,1,1).to(device) # Updated etas -> lambdas
                mean_prev = x_hist0 + (1 - lambda_prev) * (pred_x_future - x_hist0) # Updated calculation based on Methodology.md Algo 2 line 18

                z = torch.randn_like(x_t)
                # posterior_std = self.gamma * torch.sqrt((self.lambdas[t-1] * alpha_t) / self.lambdas[t]) # Updated etas -> lambdas, kappa -> gamma
                # Simplified form matching Methodology.md Algo 2 line 18 (sigma_{t-1})
                lambda_prev_for_sigma = self.lambdas[t-1].to(device) # λ_{t-1}
                lambda_prev_minus_1_for_sigma = self.lambdas[t-2].to(device) if t > 2 else torch.tensor(0.0, device=device) # λ_{t-2} or 0
                sigma_prev = self.gamma * torch.sqrt(lambda_prev_for_sigma * (1 - lambda_prev_minus_1_for_sigma)).view(1,1,1,1) # σ_{t-1}
                x_t = mean_prev + sigma_prev * z # Updated noise term
                # lambda_prev = self.lambdas[t-1].view(1,1,1,1).to(device) # Updated etas -> lambdas
                # mean_prev = (1 - lambda_prev) * pred_x_future + lambda_prev * x_hist0 # Updated calculation, aligns with code segment 2 structure
                # z = torch.randn_like(x_t)
                # lambda_t = self.lambdas[t].view(1,1,1,1).to(device) # 需要当前步的 lambda
                # alpha_t = (lambda_t - lambda_prev).view(1,1,1,1) # alpha_t = lambda_t - lambda_{t-1}
                # posterior_std = self.gamma * torch.sqrt((lambda_prev * alpha_t) / lambda_t) # Updated noise term, matches ResShift paper Eq. (A.18)
                # x_t = mean_prev + posterior_std * z # Updated noise term
            else:
                x_t = pred_x_future # Updated pred_x0 -> pred_x_future

        return x_t

# Backwards compatibility alias
# DDPM = DiffRes

# -----------------------
# End of file
# -----------------------
