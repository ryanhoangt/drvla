"""Episode -> features: the most active SAE features of one episode over time."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import (
    classifier_select,
    composite_figure,
    config,
    features_of,
    figure_bytes,
    get_frames,
    get_index,
    get_store,
    go_to_feature,
    heatmap,
    persistent_selectbox,
    sidebar_model_and_layer,
    smooth,
)

st.set_page_config(page_title="Activation Viewer", layout="wide")
cfg = config()
model, layer = sidebar_model_and_layer(cfg)
store = get_store(model["activations"])

query = st.sidebar.text_input("Filter episodes by task")
episode_ids = [e for e in store.episode_ids if query.lower() in store.episode(e)["task"].lower()]
if not episode_ids:
    st.warning(f"No episode task contains {query!r}.")
    st.stop()
episode_id = persistent_selectbox(
    "Episode", episode_ids, "episode", format_func=lambda e: f"{e}: {store.episode(e)['task'][:70]}"
)
top_n = st.sidebar.slider("Features shown", 5, 60, 15)
window = st.sidebar.slider("Smoothing window (timesteps)", 1, 9, 1)
clf = classifier_select(cfg, model)

meta = store.episode(episode_id)
features = features_of(model, layer, episode_id)
num_steps, num_features = features.shape
peak = features.max(axis=0)
top = [int(f) for f in np.argsort(-peak)[:top_n] if peak[f] > 0]

st.title("Activation Viewer")
st.subheader(f"Episode {episode_id}: {meta['task']}")
st.caption(f"{num_steps} timesteps · {int((peak > 0).sum())} of {num_features} features active · {layer}")
if not top:
    st.warning("No feature is active in this episode.")
    st.stop()

has_frames = store.has_frames(episode_id)
if has_frames:
    frames = get_frames(model["activations"], episode_id)
    camera = st.radio("Camera", ["main", "wrist"], horizontal=True)
    strip_steps = np.linspace(0, num_steps - 1, 8).round().astype(int)
    for col, t in zip(st.columns(len(strip_steps)), strip_steps):
        col.image(frames[camera, int(t)], caption=f"t={t}", width="stretch")
else:
    st.info("No frames were stored for this episode.")

st.markdown(f"**Top {len(top)} features by peak activation in this episode**")
st.plotly_chart(heatmap(smooth(features[:, top].T, window), top), width="stretch")

st.markdown("**Timestep**")
t = st.slider("Timestep", 0, num_steps - 1, int(np.argmax(features[:, top[0]])), label_visibility="collapsed")
cols = st.columns([1, 1, 2] if has_frames else [1])
if has_frames:
    cols[0].image(frames["main", t], caption=f"main, t={t}", width="stretch")
    cols[1].image(frames["wrist", t], caption=f"wrist, t={t}", width="stretch")
active_now = [int(f) for f in np.argsort(-features[t])[:15] if features[t, f] > 0]
bars = go.Figure(go.Bar(x=features[t, active_now], y=[f"F{f}" for f in active_now], orientation="h"))
bars.update_yaxes(type="category", autorange="reversed")
bars.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=30), title=f"Most active features at t={t}")
cols[-1].plotly_chart(bars, width="stretch")

st.markdown("**Features in this episode**")
metrics = get_index(model["index"]).metrics(layer).set_index("feature_id")
table = metrics.loc[top, ["episode_coverage", "mean_onset_count", "mean_activation_magnitude", "relative_run_length"]].copy()
table.insert(0, "peak in episode", peak[top])
table["P(general)"] = clf.predict_proba(table)
table.index.name = "feature"
selection = st.dataframe(
    table.style.format(precision=3), on_select="rerun", selection_mode="single-row", width="stretch"
)
if selection.selection.rows:
    feature = top[selection.selection.rows[0]]
    if st.button(f"Open F{feature} in Feature Search"):
        go_to_feature(feature)

with st.expander("Export a paper-style figure"):
    if not has_frames:
        st.info("Needs stored frames.")
    else:
        chosen = st.multiselect("Features", top, default=top[:5])
        num_images = st.slider("Frames", 3, 12, 6)
        fig_camera = st.radio("Figure camera", ["main", "wrist"], horizontal=True)
        if chosen and st.button("Render"):
            steps = np.linspace(0, num_steps - 1, num_images).round().astype(int)
            fig = composite_figure(
                smooth(features[:, chosen].T, window), chosen, [frames[fig_camera, int(s)] for s in steps], steps
            )
            st.pyplot(fig)
            name = f"episode{episode_id}_{layer}"
            st.download_button("PNG", figure_bytes(fig, "png"), f"{name}.png", "image/png")
            st.download_button("PDF", figure_bytes(fig, "pdf"), f"{name}.pdf", "application/pdf")
