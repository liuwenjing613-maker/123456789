from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class A1Head(nn.Module):

    def __init__(self, d_in: int, bias_init: list[float] | None = None) -> None:
        super().__init__()
        self.fc = nn.Linear(d_in, 3)
        if bias_init is not None:
            with torch.no_grad():
                self.fc.bias.copy_(torch.tensor(bias_init, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)

    @staticmethod
    def predict_probs(logits: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(logits)


class A2OrdinalHead(nn.Module):
    def __init__(self, d_in: int, n_items: int = 21, n_thresholds: int = 3) -> None:
        super().__init__()
        self.n_items = n_items
        self.n_thresholds = n_thresholds
        self.fc = nn.Linear(d_in, n_items * n_thresholds)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        return self.fc(x).view(B, self.n_items, self.n_thresholds)

    @staticmethod
    def predict_int(logits: torch.Tensor) -> torch.Tensor:
        return (torch.sigmoid(logits) > 0.5).long().sum(dim=-1)

    @staticmethod
    def predict_int_monotonic(logits: torch.Tensor) -> torch.Tensor:
        s = torch.sigmoid(logits)  

        p1 = s[..., 0]
        p2 = torch.min(s[..., 1], p1)
        p3 = torch.min(s[..., 2], p2)

        P0 = 1.0 - p1
        P1 = p1 - p2
        P2 = p2 - p3
        P3 = p3
        class_probs = torch.stack([P0, P1, P2, P3], dim=-1)  
        return class_probs.argmax(dim=-1) 

    @staticmethod
    def predict_expectation(logits: torch.Tensor) -> torch.Tensor:
        s = torch.sigmoid(logits)
        p1 = s[..., 0]
        p2 = torch.min(s[..., 1], p1)
        p3 = torch.min(s[..., 2], p2)
        E = p1 + p2 + p3  
        return E.round().long().clamp(0, 3)

    @staticmethod
    def build_ordinal_targets(labels: torch.Tensor, n_thresholds: int = 3) -> torch.Tensor:
        B, I = labels.shape
        thresholds = torch.arange(1, n_thresholds + 1, device=labels.device).float()
        targets = (labels.unsqueeze(-1).float() >= thresholds.view(1, 1, -1)).float()
        return targets



def a1_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pos_weight: torch.Tensor | None = None,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    if label_smoothing > 0.0:
        targets = targets.float() * (1.0 - label_smoothing) + 0.5 * label_smoothing
    return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)


def a2_ordinal_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    pos_weight: torch.Tensor | None = None,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    targets = A2OrdinalHead.build_ordinal_targets(labels, n_thresholds=logits.size(-1))
    if label_smoothing > 0.0:
        targets = targets * (1.0 - label_smoothing) + 0.5 * label_smoothing
    return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)


# def cross_modal_consistency_loss(
#     audio_features: torch.Tensor,
#     video_features: torch.Tensor,
#     audio_mask: torch.Tensor,
#     video_mask: torch.Tensor,
# ) -> torch.Tensor:
#     """
#     Cross-modal consistency loss to encourage alignment between modalities.
#     Uses cosine similarity between pooled audio and video features.
#     """
#     # Simple mean pooling with masking
#     audio_pooled = (audio_features * audio_mask.unsqueeze(-1)).sum(dim=1) / audio_mask.sum(dim=1, keepdim=True).clamp(min=1)
#     video_pooled = (video_features * video_mask.unsqueeze(-1)).sum(dim=1) / video_mask.sum(dim=1, keepdim=True).clamp(min=1)
# 
#     # Cosine similarity
#     audio_norm = F.normalize(audio_pooled, dim=-1)
#     video_norm = F.normalize(video_pooled, dim=-1)
#     similarity = (audio_norm * video_norm).sum(dim=-1)
# 
#     # Encourage higher similarity (less negative loss)
#     return -similarity.mean()
