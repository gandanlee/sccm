"""RoPE-circular with a *stochastic gain* on the rotation angles.

Motivation
----------
`RoPE2DCircular` applies a position-dependent rotation of fixed strength to Q, K.
Once trained, that strength is baked in: the backbone features co-adapt to it and
the prior can no longer be dialled down at inference.  Measured consequence
(holo360d, 400 pairs, MP3D checkpoint): scaling theta -> gamma*theta at inference
degrades monotonically (pck@1 0.0940 / 0.0881 / 0.0844 / 0.0780 / 0.0612 for
gamma = 1 / .75 / .5 / .25 / 0), i.e. the model is *brittle* to the prior's
strength rather than merely reliant on it.

This module samples gamma during training so the network must work across a range
of prior strengths.  After such training gamma becomes a usable inference-time
knob (and a prerequisite for a content-conditioned gate).

Contract
--------
* `gain_random_range = None` (default) -> exactly `RoPE2DCircular` (same tables,
  same numerics).  Safe drop-in.
* Training: gamma ~ U[lo, hi], resampled per forward call.
* Eval: gamma = `eval_gain` (default 1.0), deterministic.

Angles are kept as buffers and cos/sin computed on the fly, so the rebuild is
exact for any gamma (no table-reconstruction round-trip).
"""

from __future__ import annotations

import math
import torch

from sccm.models.rope_circular import RoPE2DCircular


class RoPE2DCircularGain(RoPE2DCircular):
    """RoPE2DCircular whose rotation angles are scaled by a (stochastic) gain."""

    def __init__(
        self,
        *args,
        gain_random_range: tuple[float, float] | None = None,
        eval_gain: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.gain_random_range = (
            None if gain_random_range is None
            else (float(gain_random_range[0]), float(gain_random_range[1]))
        )
        self.eval_gain = float(eval_gain)

        # Recover the raw angles from the parent's cos/sin tables is lossy for
        # |angle| > pi, so rebuild them from the same closed form the parent uses.
        H, W, hd = self.H, self.W, self.head_dim
        half = hd // 2
        n_pairs = half // 2
        lat_freqs = kwargs.get("lat_theta_base", 10000.0) ** (
            -torch.arange(0, n_pairs, dtype=torch.float32) * 2.0 / half
        )
        if kwargs.get("lon_geometric", False):
            lon_freqs = lat_freqs.clone()
        else:
            max_int = n_pairs if kwargs.get("lon_int_freq_max") is None else int(kwargs["lon_int_freq_max"])
            lon_freqs = torch.arange(1, max_int + 1, dtype=torch.float32)
            if lon_freqs.numel() < n_pairs:
                reps = (n_pairs + lon_freqs.numel() - 1) // lon_freqs.numel()
                lon_freqs = lon_freqs.repeat(reps)[:n_pairs]
            else:
                lon_freqs = lon_freqs[:n_pairs]

        v_norm = torch.linspace(-1.0 + 1.0 / H, 1.0 - 1.0 / H, H)
        u_norm = torch.linspace(-1.0 + 1.0 / W, 1.0 - 1.0 / W, W)
        lat = -(math.pi / 2.0) * v_norm
        if kwargs.get("equal_area_latitudes", False):
            sin_lat = torch.sin(lat)
            lat = torch.asin(torch.linspace(
                float(sin_lat[0]), float(sin_lat[-1]), H
            ).clamp(-1 + 1e-6, 1 - 1e-6))
        lon = math.pi * u_norm
        lat_grid = lat[:, None].expand(H, W).reshape(H * W)
        lon_grid = lon[None, :].expand(H, W).reshape(H * W)

        lat_ang = lat_grid[:, None] * lat_freqs[None, :]
        lon_ang = lon_grid[:, None] * lon_freqs[None, :]

        # Sanity: the reconstruction must reproduce the parent's tables at gain 1.
        assert torch.allclose(lat_ang.cos(), self.lat_cos.float(), atol=1e-4), \
            "RoPE2DCircularGain: latitude angle reconstruction mismatch"
        assert torch.allclose(lon_ang.cos(), self.lon_cos.float(), atol=1e-4), \
            "RoPE2DCircularGain: longitude angle reconstruction mismatch"

        self.register_buffer("lat_ang", lat_ang, persistent=False)
        self.register_buffer("lon_ang", lon_ang, persistent=False)
        self._gain = float(eval_gain)

    # ── gain lifecycle ────────────────────────────────────────────────────
    # `apply()` is invoked once per (q_A, q_B, k_A, k_B) slice, so sampling there
    # would give the two views *different* gains and destroy the relative-position
    # semantics of RoPE.  The gain is therefore held in `self._gain` and refreshed
    # exactly once per model forward by `resample_gain()`, which the owner wires up
    # as a forward-pre-hook.
    def resample_gain(self) -> float:
        if self.training and self.gain_random_range is not None:
            lo, hi = self.gain_random_range
            self._gain = float(lo + (hi - lo) * torch.rand(()).item())
        else:
            self._gain = float(self.eval_gain)
        return self._gain

    def set_gain(self, gain: float) -> None:
        """Pin the eval-time gain (used by inference sweeps)."""
        self.eval_gain = float(gain)
        if not self.training:
            self._gain = float(gain)

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        g = float(getattr(self, "_gain", self.eval_gain))
        if g == 1.0:
            return super().apply(x)
        half = self.head_dim // 2
        la = g * self.lat_ang
        lo = g * self.lon_ang
        x_lat = self._rotate_half(x[..., :half], la.cos().to(x.dtype), la.sin().to(x.dtype))
        x_lon = self._rotate_half(x[..., half:], lo.cos().to(x.dtype), lo.sin().to(x.dtype))
        return torch.cat([x_lat, x_lon], dim=-1)
