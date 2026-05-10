from __future__ import annotations  

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class BackboneConfig:
    audio_group_dims: dict[str, int] = field(default_factory=dict)
    audio_pooled_group_dims: dict[str, int] = field(default_factory=dict)
    video_group_dims: dict[str, int] = field(default_factory=dict)

    d_adapter: int = 64
    d_model: int = 256
    tcn_layers: int = 4
    tcn_kernel_size: int = 3
    asp_alpha: float = 0.5
    asp_beta: float = 0.5
    dropout: float = 0.1

    n_sessions: int = 4
    d_session: int = 16
    d_shared: int = 256

class GroupAdapter(nn.Module):
    def __init__(self, d_in: int, d_out: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_in)
        self.proj = nn.Linear(d_in, d_out)
        self.act = nn.GELU()
        self.norm2 = nn.LayerNorm(d_out)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(self.norm(x))
        x = self.act(x)
        x = self.drop(x)
        return self.norm2(x)

class ModalityFusion(nn.Module):
    def __init__(self, n_groups: int, d_adapter: int, d_model: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(n_groups * d_adapter)
        self.proj = nn.Linear(n_groups * d_adapter, d_model)
        self.act = nn.GELU()
        self.drop = nn.Dropout(0.1)

    def forward(self, groups: list[torch.Tensor]) -> torch.Tensor:
        x = torch.cat(groups, dim=-1)
        x = self.norm(x)
        x = self.proj(x)
        x = self.act(x)
        return self.drop(x)


class DilatedResidualBlock(nn.Module):
    def __init__(
        self, d_model: int, kernel_size: int, dilation: int, dropout: float
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(d_model, d_model, kernel_size, dilation=dilation, padding=padding)
        )
        self.conv2 = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(d_model, d_model, kernel_size, dilation=dilation, padding=padding)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        residual = x
        T = x.size(1)

        h = self.norm1(x)
        h = h.transpose(1, 2)
        h = self.conv1(h)[:, :, :T]
        h = F.gelu(h)
        h = self.drop(h)

        # block 2
        h = h.transpose(1, 2)
        h = self.norm2(h)
        h = h.transpose(1, 2)
        h = self.conv2(h)[:, :, :T]
        h = self.drop(h)
        h = h.transpose(1, 2)

        out = h + residual
        out = out * mask.unsqueeze(-1).float()
        return out


class TCN(nn.Module):
    def __init__(
        self, d_model: int, n_layers: int, kernel_size: int, dropout: float
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([
            DilatedResidualBlock(d_model, kernel_size, dilation=2**i, dropout=dropout)
            for i in range(n_layers)
        ])

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, mask)
        return x


class CrossAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int = 2, dropout: float = 0.2):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim = d_model // num_heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, query_mask: torch.Tensor, key_mask: torch.Tensor):
        """
        query: (B, T_q, D)
        key: (B, T_k, D)
        value: (B, T_k, D)
        query_mask: (B, T_q) bool
        key_mask: (B, T_k) bool
        """
        B, T_q, _ = query.size()
        B, T_k, _ = key.size()

        Q = self.q_proj(query).view(B, T_q, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, T_q, head_dim)
        K = self.k_proj(key).view(B, T_k, self.num_heads, self.head_dim).transpose(1, 2)    # (B, H, T_k, head_dim)
        V = self.v_proj(value).view(B, T_k, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, T_k, head_dim)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        query_mask_exp = query_mask.unsqueeze(1).unsqueeze(-1)  # (B, 1, T_q, 1)
        key_mask_exp = key_mask.unsqueeze(1).unsqueeze(-2)      # (B, 1, 1, T_k)
        attn_mask = query_mask_exp & key_mask_exp  # (B, 1, T_q, T_k)

        attn_scores = attn_scores.masked_fill(~attn_mask, float("-1e9"))
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = attn_weights.masked_fill(~attn_mask, 0.0)
        attn_weights = attn_weights / attn_weights.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, V)  # (B, H, T_q, head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T_q, self.d_model)  # (B, T_q, D)

        output = self.out_proj(attn_output)
        return output


class ASP(nn.Module):
    """Attentive Statistics Pooling with VAD and quality control signals."""

    def __init__(self, d_model: int, alpha: float = 0.5, beta: float = 0.5) -> None:
        super().__init__()
        self.attn = nn.Linear(d_model, 1)
        self.alpha = nn.Parameter(torch.tensor(alpha))
        self.beta = nn.Parameter(torch.tensor(beta))

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        vad: torch.Tensor,
        qc: torch.Tensor,
    ) -> torch.Tensor:
        """
        x    : (B, T, D)
        mask : (B, T) bool
        vad  : (B, T) float
        qc   : (B, T) float
        Returns: (B, 2*D)
        """
        e = self.attn(x).squeeze(-1)
        e = e + self.alpha * vad + self.beta * qc

        # mask invalid positions
        e = e.masked_fill(~mask, float("-inf"))
        w = F.softmax(e, dim=-1)
        w = w.masked_fill(~mask, 0.0)   # to avoid NaN in mean/std when all masked

        w_unsq = w.unsqueeze(-1)
        mean = (w_unsq * x).sum(dim=1)

        diff = x - mean.unsqueeze(1)
        var = (w_unsq * diff ** 2).sum(dim=1)
        std = torch.sqrt(var.clamp(min=1e-8))

        return torch.cat([mean, std], dim=-1)

class MTCNBackbone(nn.Module):
    def __init__(self, cfg: BackboneConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.audio_adapters = nn.ModuleDict({
            name: GroupAdapter(d_in, cfg.d_adapter, cfg.dropout)
            for name, d_in in cfg.audio_group_dims.items()
        })
        self.audio_pooled_adapters = nn.ModuleDict({
            name: nn.Sequential(
                nn.LayerNorm(d_in),
                nn.Linear(d_in, cfg.d_adapter),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
            )
            for name, d_in in cfg.audio_pooled_group_dims.items()
        })
        self.video_adapters = nn.ModuleDict({
            name: GroupAdapter(d_in, cfg.d_adapter, cfg.dropout)
            for name, d_in in cfg.video_group_dims.items()
        })
        self.audio_group_names = sorted(cfg.audio_group_dims.keys())
        self.audio_pooled_group_names = sorted(cfg.audio_pooled_group_dims.keys())
        self.video_group_names = sorted(cfg.video_group_dims.keys())

        self.audio_fusion = ModalityFusion(
            len(self.audio_group_names), cfg.d_adapter, cfg.d_model
        )
        self.video_fusion = ModalityFusion(
            len(self.video_group_names), cfg.d_adapter, cfg.d_model
        )

        self.audio_tcn = TCN(cfg.d_model, cfg.tcn_layers, cfg.tcn_kernel_size, cfg.dropout)
        self.video_tcn = TCN(cfg.d_model, cfg.tcn_layers, cfg.tcn_kernel_size, cfg.dropout)

        # Enhanced cross-modal attention and fusion
        self.cross_attn_a2v = CrossAttention(cfg.d_model, num_heads=2, dropout=0.1)
        self.cross_attn_v2a = CrossAttention(cfg.d_model, num_heads=2, dropout=0.1)

        self.ln_a = nn.LayerNorm(cfg.d_model)
        self.ln_v = nn.LayerNorm(cfg.d_model)

        self.audio_asp = ASP(cfg.d_model, cfg.asp_alpha, cfg.asp_beta)
        self.video_asp = ASP(cfg.d_model, cfg.asp_alpha, cfg.asp_beta)

        fusion_in = 2 * cfg.d_model * 2  
        fusion_in += len(self.audio_pooled_group_names) * cfg.d_adapter
        fusion_in += cfg.d_session 

        self.session_embed = nn.Embedding(cfg.n_sessions, cfg.d_session)

        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_in, cfg.d_shared),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_shared, cfg.d_shared),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        audio_adapted = [
            self.audio_adapters[n](batch["audio_groups"][n])
            for n in self.audio_group_names
        ]
        video_adapted = [
            self.video_adapters[n](batch["video_groups"][n])
            for n in self.video_group_names
        ]

        a = self.audio_fusion(audio_adapted)
        v = self.video_fusion(video_adapted)

        mask_a = batch["mask_audio"]
        mask_v = batch["mask_video"]
        a = a * mask_a.unsqueeze(-1).float()
        v = v * mask_v.unsqueeze(-1).float()

        a = self.audio_tcn(a, mask_a)
        v = self.video_tcn(v, mask_v)

        # Cross-modal attention with enhanced gated fusion
        a2v = self.cross_attn_a2v(query=a, key=v, value=v, query_mask=mask_a, key_mask=mask_v)
        v2a = self.cross_attn_v2a(query=v, key=a, value=a, query_mask=mask_v, key_mask=mask_a)

        # Simple residual fusion
        a_fused = self.ln_a(a + a2v)
        v_fused = self.ln_v(v + v2a)

        a, v = a_fused, v_fused

        vad = batch["vad_signal"]
        qc = batch["qc_quality"]
        z_a = self.audio_asp(a, mask_a, vad, qc)
        z_v = self.video_asp(v, mask_v, vad, qc)

        parts = [z_a, z_v]
        parts.extend(
            self.audio_pooled_adapters[name](batch["audio_pooled_groups"][name])
            for name in self.audio_pooled_group_names
        )
        parts.append(self.session_embed(batch["session_idx"]))

        z = torch.cat(parts, dim=-1)
        output = self.fusion_mlp(z)

        # Return both final output and features for consistency loss
        return {
            "output": output,
            "audio_features": a,  # fused audio features before ASP
            "video_features": v,  # fused video features before ASP
            "audio_mask": mask_a,
            "video_mask": mask_v,
        }
