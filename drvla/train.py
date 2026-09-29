import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from drvla.sae import SAE_FILENAME, SAEConfig, TopKSAE
from drvla.store import ActivationStore

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    expansion_ratio: float = 1.0
    k: int = 100
    auxk: int = 512
    auxk_coef: float = 1 / 32
    dead_steps_threshold: int = 500
    lr: float = 1e-4
    batch_size: int = 4096
    num_epochs: int = 100
    warmup_fraction: float = 0.1
    grad_clip: float = 1.0
    adam_eps: float = 6.25e-10
    adam_betas: tuple[float, float] = (0.9, 0.999)
    geometric_median_samples: int = 10000
    seed: int = 0
    save_every: int = 25


def load_layer_activations(activation_dirs: list[str | Path], layer: str) -> torch.Tensor:
    """Stack every episode of ``layer`` from one or more activation directories."""
    arrays = []
    for activation_dir in activation_dirs:
        store = ActivationStore(activation_dir)
        arrays.extend(store.activations(layer, episode_id) for episode_id in store.episode_ids)
    data = np.concatenate(arrays).astype(np.float32, copy=False)
    return torch.from_numpy(data)


def train_sae(
    activations: torch.Tensor,
    config: TrainConfig,
    output_dir: str | Path,
    device: str = "cuda",
) -> TopKSAE:
    """Train an SAE on a (N, d) float32 tensor and write checkpoints to ``output_dir``.

    ``output_dir/sae.pt`` holds the epoch with the lowest training loss;
    ``output_dir/sae_epoch_<n>.pt`` is written every ``config.save_every`` epochs.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    num_samples, input_dim = activations.shape
    num_features = input_dim * config.expansion_ratio
    if num_features != int(num_features):
        raise ValueError(f"expansion_ratio={config.expansion_ratio} gives a non-integer width for d={input_dim}.")
    sae_config = SAEConfig(
        input_dim=input_dim,
        num_features=int(num_features),
        k=config.k,
        auxk=config.auxk,
        auxk_coef=config.auxk_coef,
        dead_steps_threshold=config.dead_steps_threshold,
    )
    sae = TopKSAE(sae_config).to(device)

    loader = torch.utils.data.DataLoader(activations, batch_size=config.batch_size, shuffle=True)
    total_steps = config.num_epochs * len(loader)
    if config.dead_steps_threshold >= total_steps:
        raise ValueError(
            f"dead_steps_threshold={config.dead_steps_threshold} >= total training steps {total_steps}: "
            f"AuxK would never run. Lower it (e.g. to {max(10, total_steps // 10)}) or train longer."
        )
    logger.info(
        "Training SAE: %d samples, d=%d, %d features, k=%d, %d steps",
        num_samples, input_dim, sae_config.num_features, config.k, total_steps,
    )

    # Geometric-median pre-bias and MSE normalization constant from the first shuffled batches.
    num_init_batches = min(max(10, config.geometric_median_samples // config.batch_size + 1), len(loader))
    init_batches = []
    for batch in loader:
        init_batches.append(batch.to(device))
        if len(init_batches) == num_init_batches:
            break
    init_samples = torch.cat(init_batches)
    sae.init_pre_bias(init_samples[: config.geometric_median_samples])
    mse_scale = ((init_samples - init_samples.mean(dim=0)) ** 2).mean().item()
    del init_batches, init_samples

    optimizer = torch.optim.Adam(sae.parameters(), lr=config.lr, eps=config.adam_eps, betas=config.adam_betas)
    warmup_steps = int(total_steps * config.warmup_fraction)

    log_path = output_dir / "train_log.jsonl"
    log_path.unlink(missing_ok=True)
    best_loss = float("inf")
    step = 0
    sae.train()
    for epoch in range(1, config.num_epochs + 1):
        start = time.time()
        sums = {"loss": 0.0, "recon": 0.0, "auxk": 0.0, "dead_fraction": 0.0}
        for batch in loader:
            batch = batch.to(device, non_blocking=True)
            out = sae(batch)
            recon_mse = torch.nn.functional.mse_loss(out.reconstruction, batch)
            recon_loss = recon_mse / mse_scale
            auxk_loss = sae.auxk_loss(out)
            if auxk_loss.item() > 0:
                auxk_loss = auxk_loss / (recon_mse.item() + 1e-8)
            loss = recon_loss + config.auxk_coef * auxk_loss

            if step < warmup_steps:
                for group in optimizer.param_groups:
                    group["lr"] = config.lr * (step + 1) / warmup_steps
            optimizer.zero_grad()
            loss.backward()
            sae.project_decoder_grads_()
            if config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(sae.parameters(), config.grad_clip)
            optimizer.step()
            sae.normalize_decoder_()
            step += 1

            sums["loss"] += loss.item()
            sums["recon"] += recon_loss.item()
            sums["auxk"] += auxk_loss.item()
            sums["dead_fraction"] += sae.dead_mask().float().mean().item()

        record = {"epoch": epoch, **{key: value / len(loader) for key, value in sums.items()}}
        record["seconds"] = time.time() - start
        with open(log_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        logger.info(
            "epoch %d/%d loss %.5f recon %.5f auxk %.5f dead %.1f%%",
            epoch, config.num_epochs, record["loss"], record["recon"], record["auxk"],
            100 * record["dead_fraction"],
        )

        train_info = {"train_config": asdict(config), "epoch": epoch, "train_loss": record["loss"]}
        if record["loss"] < best_loss:
            best_loss = record["loss"]
            sae.save(output_dir / SAE_FILENAME, **train_info)
        if epoch % config.save_every == 0:
            sae.save(output_dir / f"sae_epoch_{epoch}.pt", **train_info)

    return sae.eval()
