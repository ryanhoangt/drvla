"""Error-preserving edits of pooled SAE coordinates, broadcast over tokens.

This is a pooled-direction pilot, not a token-level SAE intervention. For the
recipient pooled x, keep its normalization scale and reconstruction error:
    x' = x + scale(x) * W_dec @ (f' - f).
The same delta is added to every token; original token differences are retained.
"""

import torch

MODES = (
    "baseline", "identity", "suppress_old", "boost_new", "patch_old",
    "random_suppress", "general_suppress", "random_boost", "general_boost",
)


def pooled_scale(sae, pooled):
    centered = pooled - sae.pre_bias
    return (centered - centered.mean(-1, keepdim=True)).norm(dim=-1, keepdim=True).clamp(min=1e-8)


def feature_delta(sae, features, pair, global_mean, mode, alpha=1.0, donor=None):
    """Return edited feature difference and the intervention's feature IDs."""
    if mode not in MODES:
        raise ValueError(f"Unknown mode {mode}")
    if alpha < 0 or (("suppress" in mode or mode == "patch_old") and alpha > 1):
        raise ValueError("Suppression alpha must be in [0, 1]; boost/patch alpha must be nonnegative")
    difference = torch.zeros_like(features)
    if mode in ("baseline", "identity"):
        return difference, []
    suppress = mode.endswith("suppress") or mode == "suppress_old"
    controls = mode.startswith("random_") or mode.startswith("general_")
    target_ids = pair["old" if suppress or mode == "patch_old" else "new"]
    if controls:
        key = mode.split("_")[0] + ("_old" if suppress else "_new")
        ids = pair[key]
    else:
        ids = target_ids
    if suppress:
        difference[:, ids] = -alpha * features[:, ids]
    elif mode == "patch_old":
        if donor is None:
            raise ValueError("patch_old needs a same-observation, old-prompt donor")
        difference[:, ids] = alpha * (donor[:, ids] - features[:, ids])
    else:
        reference = global_mean if controls else pair["new_mean"]
        values = torch.as_tensor(reference, device=features.device, dtype=features.dtype)
        difference[:, ids] = alpha * (values[ids] - features[:, ids]).clamp(min=0)

    if controls:
        # Match the actual decoded perturbation norm, at this recipient observation,
        # to the corresponding targeted intervention (not just feature count).
        target_mode = "suppress_old" if suppress else "boost_new"
        target, _ = feature_delta(sae, features, pair, global_mean, target_mode, alpha)
        target_norm = sae.decoder(target).norm(dim=-1, keepdim=True)
        control_norm = sae.decoder(difference).norm(dim=-1, keepdim=True)
        if suppress and bool((control_norm < target_norm).any()):
            # Use a predeclared ID-only candidate pool if the fixed set cannot
            # supply enough suppression energy. Never create negative latents.
            pool = pair[mode.split("_")[0] + "_pool"]
            strengths = features[0, pool]
            order = torch.argsort(strengths, descending=True)[: len(target_ids)]
            ids = [pool[int(j)] for j in order]
            difference.zero_()
            difference[:, ids] = -alpha * features[:, ids]
            control_norm = sae.decoder(difference).norm(dim=-1, keepdim=True)
        elif not suppress and bool(((control_norm <= 1e-12) & (target_norm > 1e-12)).any()):
            difference.zero_()
            values = torch.as_tensor(global_mean, device=features.device, dtype=features.dtype)
            difference[:, ids] = alpha * values[ids]
            control_norm = sae.decoder(difference).norm(dim=-1, keepdim=True)
        ratio = target_norm / control_norm.clamp(min=1e-12)
        if suppress:
            ratio = ratio.clamp(max=1 / max(alpha, 1e-12))
        difference *= ratio
    return difference, ids


def apply_pooled_delta(hidden, pooled, sae, difference):
    delta = pooled_scale(sae, pooled) * sae.decoder(difference)
    # A zero edit returns the exact tensor object; avoid a dtype round trip.
    if not bool(torch.count_nonzero(delta)):
        return hidden, delta
    return hidden + delta[:, None, :].to(hidden.dtype), delta
