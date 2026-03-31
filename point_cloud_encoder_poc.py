"""Proof-of-concept point cloud encoder 

This module loads the SeamGPT-style JSON exported by the Blender addon,
encodes vertex and edge point streams with mask-aware pooling, embeds the seam
length ratio R with sinusoidal encoding, and concatenates both embeddings.

Usage:
    python point_cloud_encoder_poc.py --input C:/path/to/Data.json
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from typing import Any, Dict

import torch
import torch.nn as nn


EXPECTED_VERTEX_POINTS = 30720
EXPECTED_EDGE_POINTS = 30720
DEFAULT_RATIO_MIN = 0.1
DEFAULT_RATIO_MAX = 0.35


def sinusoidal_scalar_encoding(values: torch.Tensor, dim: int) -> torch.Tensor:
    """Encode scalar values with sinusoidal features.

    Args:
        values: Tensor with shape [B, 1].
        dim: Output feature dimension.

    Returns:
        Tensor of shape [B, dim].
    """
    if dim < 2:
        raise ValueError("dim must be >= 2")

    half = dim // 2
    if half == 1:
        inv_freq = torch.ones(1, device=values.device, dtype=values.dtype)
    else:
        exponent = torch.arange(half, device=values.device, dtype=values.dtype)
        exponent = exponent / float(half - 1)
        inv_freq = torch.exp(-math.log(10000.0) * exponent)

    angles = values * inv_freq.unsqueeze(0)
    encoded = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)

    if dim % 2 == 1:
        encoded = torch.cat([encoded, torch.zeros_like(encoded[:, :1])], dim=-1)

    return encoded


class MaskedPointStreamEncoder(nn.Module):
    """Memory-aware masked point stream encoder (PointNet-style PoC)."""

    def __init__(
        self,
        point_dim: int = 3,
        hidden_dim: int = 256,
        chunk_size: int = 4096,
    ) -> None:
        super().__init__()
        self.chunk_size = int(chunk_size)
        self.hidden_dim = int(hidden_dim)
        self.point_mlp = nn.Sequential(
            nn.Linear(point_dim, 64),
            nn.GELU(),
            nn.Linear(64, 128),
            nn.GELU(),
            nn.Linear(128, hidden_dim),
            nn.GELU(),
        )

    def forward(self, points: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Encode points with mask-aware mean/max pooling.

        Args:
            points: [B, N, 3]
            mask: [B, N], 1 for valid points and 0 for padding

        Returns:
            [B, hidden_dim * 2]
        """
        if points.ndim != 3 or points.size(-1) != 3:
            raise ValueError("points must have shape [B, N, 3]")
        if mask.ndim == 3 and mask.size(-1) == 1:
            mask = mask.squeeze(-1)
        if mask.ndim != 2:
            raise ValueError("mask must have shape [B, N]")
        if points.size(0) != mask.size(0) or points.size(1) != mask.size(1):
            raise ValueError("points and mask batch/sequence dimensions must match")

        mask = mask.to(dtype=points.dtype)
        batch_size, num_points, _ = points.shape
        device = points.device

        sum_feat = torch.zeros(batch_size, self.hidden_dim, device=device, dtype=points.dtype)
        count = torch.zeros(batch_size, 1, device=device, dtype=points.dtype)
        max_feat = torch.full(
            (batch_size, self.hidden_dim),
            float("-inf"),
            device=device,
            dtype=points.dtype,
        )

        for start in range(0, num_points, self.chunk_size):
            end = min(start + self.chunk_size, num_points)
            chunk_points = points[:, start:end, :]
            chunk_mask = mask[:, start:end].unsqueeze(-1)

            chunk_feat = self.point_mlp(chunk_points)

            sum_feat = sum_feat + (chunk_feat * chunk_mask).sum(dim=1)
            count = count + chunk_mask.sum(dim=1)

            chunk_max = chunk_feat.masked_fill(chunk_mask < 0.5, float("-inf")).max(dim=1).values
            max_feat = torch.maximum(max_feat, chunk_max)

        mean_feat = sum_feat / count.clamp_min(1.0)
        max_feat = torch.where(torch.isfinite(max_feat), max_feat, torch.zeros_like(max_feat))

        return torch.cat([mean_feat, max_feat], dim=-1)


class LengthRatioEmbedding(nn.Module):
    """Embed scalar ratio R with sinusoidal encoding followed by an MLP."""

    def __init__(
        self,
        sinusoidal_dim: int = 64,
        output_dim: int = 128,
        ratio_min: float = DEFAULT_RATIO_MIN,
        ratio_max: float = DEFAULT_RATIO_MAX,
    ) -> None:
        super().__init__()
        self.sinusoidal_dim = int(sinusoidal_dim)
        self.ratio_min = float(ratio_min)
        self.ratio_max = float(ratio_max)
        self.projector = nn.Sequential(
            nn.Linear(self.sinusoidal_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, ratio_r: torch.Tensor) -> torch.Tensor:
        if ratio_r.ndim == 1:
            ratio_r = ratio_r.unsqueeze(-1)
        if ratio_r.ndim != 2 or ratio_r.size(-1) != 1:
            raise ValueError("ratio_r must have shape [B] or [B, 1]")

        ratio_r = ratio_r.to(dtype=torch.float32)
        ratio_r = ratio_r.clamp(min=self.ratio_min, max=self.ratio_max)

        encoded = sinusoidal_scalar_encoding(ratio_r, self.sinusoidal_dim)
        return self.projector(encoded)


class SeamPointCloudEncoderPOC(nn.Module):
    """PoC encoder with vertex/edge streams and length conditioning."""

    def __init__(
        self,
        shape_latent_dim: int = 512,
        length_embed_dim: int = 128,
        stream_hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.vertex_stream = MaskedPointStreamEncoder(hidden_dim=stream_hidden_dim)
        self.edge_stream = MaskedPointStreamEncoder(hidden_dim=stream_hidden_dim)

        stream_out_dim = stream_hidden_dim * 2
        self.shape_fuser = nn.Sequential(
            nn.Linear(stream_out_dim * 2, shape_latent_dim),
            nn.GELU(),
            nn.LayerNorm(shape_latent_dim),
            nn.Linear(shape_latent_dim, shape_latent_dim),
        )
        self.length_embedder = LengthRatioEmbedding(output_dim=length_embed_dim)

    def forward(
        self,
        vertex_points: torch.Tensor,
        vertex_mask: torch.Tensor,
        edge_points: torch.Tensor,
        edge_mask: torch.Tensor,
        ratio_r: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        vertex_feat = self.vertex_stream(vertex_points, vertex_mask)
        edge_feat = self.edge_stream(edge_points, edge_mask)

        shape_latent = self.shape_fuser(torch.cat([vertex_feat, edge_feat], dim=-1))
        length_embedding = self.length_embedder(ratio_r)
        conditioned_latent = torch.cat([shape_latent, length_embedding], dim=-1)

        return {
            "shape_latent": shape_latent,
            "length_embedding": length_embedding,
            "conditioned_latent": conditioned_latent,
        }


@dataclass
class SeamGPTBatch:
    vertex_points: torch.Tensor
    vertex_mask: torch.Tensor
    edge_points: torch.Tensor
    edge_mask: torch.Tensor
    ratio_r: torch.Tensor
    raw_ratio_r: float
    clamped_ratio_r: float
    has_seams: bool
    seam_segment_count: int
    seam_token_count: int


def _as_float_tensor(values: Any, field_name: str) -> torch.Tensor:
    try:
        return torch.tensor(values, dtype=torch.float32)
    except Exception as exc:  # pragma: no cover - defensive parse guard
        raise ValueError(f"Failed to parse field '{field_name}' as float tensor: {exc}") from exc


def load_seamgpt_batch(json_path: str, ratio_source: str = "clamped") -> SeamGPTBatch:
    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    shape_context = payload.get("shape_context") or {}
    vertex_points = _as_float_tensor(shape_context.get("vertex_points", []), "shape_context.vertex_points")
    edge_points = _as_float_tensor(shape_context.get("edge_points", []), "shape_context.edge_points")

    if vertex_points.ndim != 2 or vertex_points.size(-1) != 3:
        raise ValueError("shape_context.vertex_points must have shape [N, 3]")
    if edge_points.ndim != 2 or edge_points.size(-1) != 3:
        raise ValueError("shape_context.edge_points must have shape [N, 3]")

    vertex_mask_values = shape_context.get("vertex_padding_mask")
    edge_mask_values = shape_context.get("edge_padding_mask")

    if vertex_mask_values is None:
        vertex_mask_values = [1.0] * int(vertex_points.size(0))
    if edge_mask_values is None:
        edge_mask_values = [1.0] * int(edge_points.size(0))

    vertex_mask = _as_float_tensor(vertex_mask_values, "shape_context.vertex_padding_mask")
    edge_mask = _as_float_tensor(edge_mask_values, "shape_context.edge_padding_mask")

    if vertex_mask.ndim != 1 or vertex_mask.size(0) != vertex_points.size(0):
        raise ValueError("vertex mask length must match vertex point count")
    if edge_mask.ndim != 1 or edge_mask.size(0) != edge_points.size(0):
        raise ValueError("edge mask length must match edge point count")

    length_conditioning = payload.get("length_conditioning") or {}
    raw_ratio_r = float(length_conditioning.get("ratio_R", DEFAULT_RATIO_MIN))
    clamped_ratio_r = float(length_conditioning.get("clamped_ratio_R", raw_ratio_r))

    ratio_key = ratio_source.strip().lower()
    if ratio_key not in {"raw", "clamped"}:
        raise ValueError("ratio_source must be 'raw' or 'clamped'")
    ratio_value = raw_ratio_r if ratio_key == "raw" else clamped_ratio_r
    ratio_r = _as_float_tensor([ratio_value], f"length_conditioning.{ratio_key}_ratio_R")

    labels = payload.get("labels") or {}
    has_seams = bool(labels.get("has_seams", labels.get("seam_segment_count", 0) > 0))
    seam_segment_count = int(labels.get("seam_segment_count", 0))
    seam_token_count = int(labels.get("seam_token_count", 0))

    return SeamGPTBatch(
        vertex_points=vertex_points.unsqueeze(0),
        vertex_mask=vertex_mask.unsqueeze(0),
        edge_points=edge_points.unsqueeze(0),
        edge_mask=edge_mask.unsqueeze(0),
        ratio_r=ratio_r,
        raw_ratio_r=raw_ratio_r,
        clamped_ratio_r=clamped_ratio_r,
        has_seams=has_seams,
        seam_segment_count=seam_segment_count,
        seam_token_count=seam_token_count,
    )


def _resolve_device(name: str) -> torch.device:
    requested = name.strip().lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _summarize_batch(batch: SeamGPTBatch) -> str:
    vertex_count = int(batch.vertex_points.size(1))
    edge_count = int(batch.edge_points.size(1))
    vertex_valid = int(batch.vertex_mask.sum().item())
    edge_valid = int(batch.edge_mask.sum().item())

    vertex_valid_pct = (100.0 * vertex_valid / max(vertex_count, 1))
    edge_valid_pct = (100.0 * edge_valid / max(edge_count, 1))

    expected = (
        f"expected vertex/edge = {EXPECTED_VERTEX_POINTS}/{EXPECTED_EDGE_POINTS}, "
        f"got {vertex_count}/{edge_count}"
    )
    validity = (
        f"valid vertex/edge points = {vertex_valid}/{edge_valid} "
        f"({vertex_valid_pct:.2f}%/{edge_valid_pct:.2f}%)"
    )
    label_info = (
        f"has_seams = {batch.has_seams}, seam segments = {batch.seam_segment_count}, "
        f"seam tokens = {batch.seam_token_count}"
    )
    ratio_info = (
        f"ratio R (raw/clamped/used) = {batch.raw_ratio_r:.6f}/"
        f"{batch.clamped_ratio_r:.6f}/{float(batch.ratio_r.item()):.6f}"
    )
    return "\n".join([expected, validity, label_info, ratio_info])


def _quality_notes(batch: SeamGPTBatch) -> str:
    notes = []

    vertex_count = int(batch.vertex_points.size(1))
    edge_count = int(batch.edge_points.size(1))
    vertex_valid = int(batch.vertex_mask.sum().item())
    edge_valid = int(batch.edge_mask.sum().item())

    vertex_valid_pct = vertex_valid / max(vertex_count, 1)
    edge_valid_pct = edge_valid / max(edge_count, 1)

    expected_tokens = 2 + (6 * int(batch.seam_segment_count))
    if expected_tokens != int(batch.seam_token_count):
        notes.append(
            f"- Token count mismatch: expected {expected_tokens} from seam segments but got {batch.seam_token_count}."
        )
    else:
        notes.append("- Token count is consistent with segment count (2 + 6 * segments).")

    if not batch.has_seams or batch.seam_segment_count == 0:
        notes.append(
            "- No seam supervision signal in this sample (only SOS/EOS target); good for pipeline sanity check, weak for learning seams."
        )

    if vertex_valid_pct < 0.05 or edge_valid_pct < 0.05:
        notes.append(
            "- Heavy padding detected (>95% in at least one stream); masked pooling handles it, but geometric diversity per sample is low."
        )

    if (not batch.has_seams) and batch.raw_ratio_r <= 1e-9 and batch.clamped_ratio_r >= DEFAULT_RATIO_MIN:
        notes.append(
            "- Raw ratio is 0 while clamped ratio is at minimum bound; this is expected from clamping but can be misleading for zero-seam samples."
        )

    if not notes:
        notes.append("- No obvious data-quality issues detected for this sample.")

    return "\n".join(notes)


def run_demo(args: argparse.Namespace) -> None:
    device = _resolve_device(args.device)
    batch = load_seamgpt_batch(args.input, ratio_source=args.ratio_source)

    model = SeamPointCloudEncoderPOC(
        shape_latent_dim=args.shape_latent_dim,
        length_embed_dim=args.length_embed_dim,
        stream_hidden_dim=args.stream_hidden_dim,
    ).to(device)
    model.eval()

    with torch.no_grad():
        outputs = model(
            vertex_points=batch.vertex_points.to(device),
            vertex_mask=batch.vertex_mask.to(device),
            edge_points=batch.edge_points.to(device),
            edge_mask=batch.edge_mask.to(device),
            ratio_r=batch.ratio_r.to(device),
        )

    print("Loaded batch summary:")
    print(_summarize_batch(batch))
    print("")
    print("Quality notes:")
    print(_quality_notes(batch))
    print("")
    print("Encoder output shapes:")
    print(f"shape_latent:      {tuple(outputs['shape_latent'].shape)}")
    print(f"length_embedding:  {tuple(outputs['length_embedding'].shape)}")
    print(f"conditioned_latent:{tuple(outputs['conditioned_latent'].shape)}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run point cloud encoder PoC on SeamGPT export JSON")
    parser.add_argument("--input", required=True, help="Path to SeamGPT export JSON (Data*.json)")
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device: auto, cpu, cuda, cuda:0, ...",
    )
    parser.add_argument(
        "--ratio-source",
        choices=["raw", "clamped"],
        default="clamped",
        help="Which length ratio to feed into the embedding.",
    )
    parser.add_argument("--shape-latent-dim", type=int, default=512)
    parser.add_argument("--length-embed-dim", type=int, default=128)
    parser.add_argument("--stream-hidden-dim", type=int, default=256)
    return parser


if __name__ == "__main__":
    parsed_args = build_arg_parser().parse_args()
    run_demo(parsed_args)
