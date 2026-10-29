"""Dense versus mean-direction patching of the same eligible token coordinates."""

import torch


def patch_hidden(hidden, donor, eligible, mode):
    if hidden.shape != donor.shape or eligible.shape != hidden.shape[:2]:
        raise ValueError("Donor/recipient token shapes differ")
    if mode not in ("dense", "pooled", "pooled_norm", "self"):
        raise ValueError(mode)
    mask = eligible[..., None]
    difference = (donor.float() - hidden.float()) * mask
    dense_norm = difference.norm()
    ratio = 1.0
    if mode == "self" or not bool(torch.count_nonzero(difference)):
        edited = hidden
    elif mode == "dense":
        # Exact replacement on jointly valid positions; padding is untouched.
        edited = torch.where(mask, donor, hidden)
    else:
        mean = difference.sum(dim=1, keepdim=True) / eligible.sum(dim=1)[:, None, None].clamp(min=1)
        uniform = mean * mask
        if mode == "pooled_norm":
            if float(uniform.norm()) <= 1e-12:
                raise ValueError("Cannot norm-match a zero pooled direction")
            ratio = float(dense_norm / uniform.norm())
            uniform = uniform * ratio
        edited = hidden + uniform.to(hidden.dtype)
    realized = edited.float() - hidden.float()
    if not bool(torch.isfinite(edited).all()):
        raise ValueError("Nonfinite patched hidden states")
    return edited, {
        "dense_delta_norm": float(dense_norm), "realized_delta_norm": float(realized.norm()),
        "realized_vs_dense_norm": float(realized.norm() / dense_norm) if dense_norm > 0 else 1.0,
        "pooled_rescale": ratio, "eligible_tokens": int(eligible.sum()),
        "total_tokens": hidden.shape[1], "applied": bool(torch.count_nonzero(realized)),
    }
