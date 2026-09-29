"""Feature -> episodes: the episodes and timesteps where one SAE feature activates most."""

import numpy as np
import streamlit as st

from common import (
    classifier_select,
    config,
    features_of,
    get_frames,
    get_index,
    get_store,
    go_to_episode,
    sidebar_model_and_layer,
    trace,
)
from drvla.metrics import METRIC_LABELS, METRICS

st.set_page_config(page_title="Feature Search", layout="wide")
cfg = config()
model, layer = sidebar_model_and_layer(cfg)
store = get_store(model["activations"])
index = get_index(model["index"])
metrics = index.metrics(layer)
active_ids = metrics.loc[metrics["num_active_episodes"] > 0, "feature_id"].to_numpy()

if "_goto_feature" in st.session_state:
    st.session_state["feature_input"] = st.session_state.pop("_goto_feature")
if st.sidebar.button("Random active feature"):
    st.session_state["feature_input"] = int(np.random.choice(active_ids))
feature = int(st.sidebar.number_input("Feature", 0, len(metrics) - 1, key="feature_input"))
clf = classifier_select(cfg, model)

st.title(f"Feature Search: F{feature}")
row = metrics.iloc[feature]
if row["num_active_episodes"] == 0:
    st.warning(f"F{feature} never activates on this dataset.")
    st.stop()

active = metrics[metrics["num_active_episodes"] > 0].copy()
active["p_general"] = clf.predict_proba(active)
p_general = float(active.loc[active["feature_id"] == feature, "p_general"].iloc[0])
rank = int((active["p_general"] > p_general).sum()) + 1
cols = st.columns(len(METRICS) + 1)
for col, metric in zip(cols, METRICS):
    col.metric(METRIC_LABELS[metric], f"{row[metric]:.3f}")
cols[-1].metric(f"P(general): {'general' if p_general >= 0.5 else 'memorized'}", f"{p_general:.3f}")
st.caption(
    f"Active in {int(row['num_active_episodes'])} episodes · on in {row['activation_frequency']:.2%} of timesteps · "
    f"max activation {row['max_activation']:.3f} · generality rank {rank} of {len(active)} active features "
    f"(classifier: {clf.description or 'unnamed'})"
)

tab_episodes, tab_timesteps = st.tabs(["Top episodes", "Top timesteps"])

with tab_episodes:
    st.caption("Episodes ranked by the feature's maximum activation, with the activation over the whole episode.")
    for rank_in_list, entry in enumerate(index.top_episodes(layer, feature), start=1):
        episode_id = entry["episode_id"]
        meta = store.episode(episode_id)
        values = features_of(model, layer, episode_id)[:, feature]
        peak_t = entry["timesteps"][0]
        with st.container(border=True):
            st.markdown(f"**{rank_in_list}. Episode {episode_id}** · peak {entry['max_activation']:.3f} at t={peak_t} · {meta['task']}")
            cols = st.columns([1, 1, 3])
            if store.has_frames(episode_id):
                frames = get_frames(model["activations"], episode_id)
                cols[0].image(frames["main", peak_t], caption=f"main, t={peak_t}", width="stretch")
                cols[1].image(frames["wrist", peak_t], caption=f"wrist, t={peak_t}", width="stretch")
            else:
                cols[0].caption("No frames stored for this episode.")
            cols[-1].plotly_chart(trace(values, marks=[peak_t]), width="stretch", key=f"trace_{episode_id}")
            if cols[-1].button("Open in Activation Viewer", key=f"open_{episode_id}"):
                go_to_episode(episode_id)

with tab_timesteps:
    n = st.slider("Timesteps", 6, index.info["layers"][layer]["top_k"], 24, step=6)
    top = index.top_activations(layer, feature, n)
    st.caption(
        f"The {len(top)} highest activations over all timesteps, from "
        f"{top['episode_id'].nunique()} distinct episodes."
    )
    camera = st.radio("Camera", ["main", "wrist"], horizontal=True, key="timestep_camera")
    grid = st.columns(6)
    for i, item in enumerate(top.itertuples()):
        task = store.episode(item.episode_id)["task"]
        caption = f"ep {item.episode_id}, t={item.timestep} · {item.activation:.3f}\n{task[:60]}"
        with grid[i % 6]:
            if store.has_frames(item.episode_id):
                st.image(get_frames(model["activations"], item.episode_id)[camera, item.timestep], caption=caption, width="stretch")
            else:
                st.caption(f"{caption}\n(no frames stored)")
