"""Shared helpers for the dashboard pages: configuration, cached data access, plots."""

import io
import sys
import tomllib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch

from drvla.classifier import GeneralityClassifier
from drvla.index import FeatureIndex
from drvla.layers import short_name
from drvla.sae import encode_batched, load_sae
from drvla.store import ActivationStore

REPO_ROOT = Path(__file__).resolve().parent.parent
PAPER_CLASSIFIERS = REPO_ROOT / "data" / "classifiers"
PAPER_LABELS = REPO_ROOT / "data" / "labels"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------- configuration


def _config_path() -> Path:
    argv = sys.argv[1:]
    if "--config" not in argv or argv.index("--config") + 1 >= len(argv):
        st.error("Start the dashboard with:  `streamlit run dashboard/Home.py -- --config <config.toml>`")
        st.stop()
    return Path(argv[argv.index("--config") + 1]).resolve()


@st.cache_resource
def _load_config(path: str) -> dict:
    raw = tomllib.loads(Path(path).read_text())
    base = Path(path).parent
    if "workspace" not in raw:
        raise ValueError(f"{path}: set 'workspace', the directory where labels and classifiers are saved.")
    if not raw.get("models"):
        raise ValueError(f"{path}: define at least one [[models]] entry.")
    models = {}
    for entry in raw["models"]:
        missing = [key for key in ("name", "activations", "saes", "index") if key not in entry]
        if missing:
            raise ValueError(f"{path}: model entry {entry} is missing {missing}.")
        model = {"name": entry["name"]}
        for key in ("activations", "saes", "index", "classifier"):
            if key in entry:
                model[key] = str((base / entry[key]).resolve())
        models[entry["name"]] = model
    workspace = (base / raw["workspace"]).resolve()
    for sub in ("labels", "classifiers"):
        (workspace / sub).mkdir(parents=True, exist_ok=True)
    return {"models": models, "workspace": workspace}


def config() -> dict:
    return _load_config(str(_config_path()))


# ---------------------------------------------------------------- cached data


@st.cache_resource
def get_store(path: str) -> ActivationStore:
    return ActivationStore(path)


@st.cache_resource
def get_index(path: str) -> FeatureIndex:
    return FeatureIndex(path)


@st.cache_resource
def get_sae(sae_dir: str, layer: str):
    return load_sae(sae_dir, layer, device=DEVICE)


@st.cache_data(max_entries=64, show_spinner=False)
def episode_features(sae_dir: str, activation_dir: str, layer: str, episode_id: int) -> np.ndarray:
    """(T, num_features) SAE activations of one episode."""
    activations = get_store(activation_dir).activations(layer, episode_id)
    return encode_batched(get_sae(sae_dir, layer), activations).numpy()


@st.cache_resource(max_entries=32)
def get_frames(activation_dir: str, episode_id: int):
    return get_store(activation_dir).frames(episode_id)


def features_of(model: dict, layer: str, episode_id: int) -> np.ndarray:
    return episode_features(model["saes"], model["activations"], layer, episode_id)


# ---------------------------------------------------------------- widgets


def persistent_selectbox(label, options, key, container=None, **kwargs):
    """A selectbox whose value survives switching between pages."""
    container = container or st.sidebar
    stored = st.session_state.get(f"_{key}")
    index = options.index(stored) if stored in options else 0
    value = container.selectbox(label, options, index=index, key=f"widget_{key}", **kwargs)
    st.session_state[f"_{key}"] = value
    return value


def sidebar_model_and_layer(cfg: dict) -> tuple[dict, str]:
    model = cfg["models"][persistent_selectbox("Model", list(cfg["models"]), "model")]
    layers = get_index(model["index"]).layers
    layer = persistent_selectbox("Layer", layers, "layer", format_func=lambda l: f"{short_name(l)}  ({l})")
    return model, layer


def available_classifiers(cfg: dict, model: dict) -> dict[str, Path]:
    """Classifiers to choose from, the model's configured default first."""
    options = {}
    if "classifier" in model:
        options[f"{Path(model['classifier']).stem} (default)"] = Path(model["classifier"])
    for path in sorted(PAPER_CLASSIFIERS.glob("*.json")):
        options.setdefault(f"{path.stem} (paper)", path)
    for path in sorted((cfg["workspace"] / "classifiers").glob("*.json")):
        options.setdefault(f"{path.stem} (workspace)", path)
    return options


def classifier_select(cfg: dict, model: dict, container=None) -> GeneralityClassifier:
    options = available_classifiers(cfg, model)
    if not options:
        st.error("No classifier found. Add one to data/classifiers or train one on the Feature Classification page.")
        st.stop()
    choice = persistent_selectbox("Classifier", list(options), "classifier", container=container)
    return GeneralityClassifier.load(options[choice])


def go_to_feature(feature_id: int):
    st.session_state["_goto_feature"] = int(feature_id)
    st.switch_page("pages/2_Feature_Search.py")


def go_to_episode(episode_id: int):
    st.session_state["_episode"] = int(episode_id)
    st.switch_page("pages/1_Activation_Viewer.py")


# ---------------------------------------------------------------- plots


def smooth(matrix: np.ndarray, window: int) -> np.ndarray:
    """Moving average along the last axis."""
    if window <= 1:
        return matrix
    kernel = np.ones(window) / window
    return np.stack([np.convolve(row, kernel, mode="same") for row in np.atleast_2d(matrix)])


def heatmap(matrix: np.ndarray, feature_ids, height: int | None = None) -> go.Figure:
    """Features x timesteps heatmap."""
    fig = go.Figure(
        go.Heatmap(
            z=matrix,
            x=np.arange(matrix.shape[1]),
            y=[f"F{f}" for f in feature_ids],
            colorscale="Viridis",
            colorbar=dict(title="activation", thickness=12),
            hovertemplate="%{y}  t=%{x}<br>activation %{z:.3f}<extra></extra>",
        )
    )
    fig.update_yaxes(type="category", autorange="reversed")
    fig.update_layout(
        height=height or max(260, 22 * len(feature_ids) + 80),
        margin=dict(l=10, r=10, t=10, b=40),
        xaxis_title="timestep",
    )
    return fig


def trace(values: np.ndarray, marks=(), height: int = 160) -> go.Figure:
    """Activation of one feature over an episode, with optional marked timesteps."""
    fig = go.Figure(go.Scatter(y=values, mode="lines", line=dict(width=1.5), hovertemplate="t=%{x}<br>%{y:.3f}<extra></extra>"))
    for t in marks:
        fig.add_vline(x=t, line_dash="dot", line_width=1, opacity=0.6)
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=30), xaxis_title="timestep", yaxis_title="act.")
    return fig


def composite_figure(matrix: np.ndarray, feature_ids, images: list[np.ndarray], image_timesteps, dpi: int = 200):
    """Paper-style figure: a strip of camera frames above a feature heatmap, sharing the time axis."""
    num_features, num_steps = matrix.shape
    strip = np.concatenate(images, axis=1)
    strip_aspect = strip.shape[0] / strip.shape[1]
    width = 12.0
    heat_height = max(0.9, 0.2 * num_features)
    fig, (ax_img, ax_heat) = plt.subplots(
        2, 1, figsize=(width, width * strip_aspect + heat_height), dpi=dpi,
        gridspec_kw={"height_ratios": [width * strip_aspect, heat_height], "hspace": 0.03},
    )
    ax_img.imshow(strip)
    ax_img.axis("off")
    image = ax_heat.imshow(matrix, aspect="auto", cmap="viridis", interpolation="nearest",
                           extent=[-0.5, num_steps - 0.5, num_features - 0.5, -0.5])
    ax_heat.set_yticks(range(num_features))
    ax_heat.set_yticklabels([f"F{f}" for f in feature_ids], fontsize=7)
    ax_heat.set_xticks(image_timesteps)
    ax_heat.tick_params(axis="x", labelsize=7)
    ax_heat.set_xlabel("timestep", fontsize=8)
    fig.colorbar(image, ax=[ax_img, ax_heat], shrink=0.6, pad=0.01)
    return fig


def figure_bytes(fig, fmt: str) -> bytes:
    buffer = io.BytesIO()
    fig.savefig(buffer, format=fmt, bbox_inches="tight")
    return buffer.getvalue()
