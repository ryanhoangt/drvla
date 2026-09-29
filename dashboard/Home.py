"""Dr. VLA dashboard.

    streamlit run dashboard/Home.py -- --config dashboard/config.toml
"""

import streamlit as st

from common import config, get_index, get_store
from drvla.layers import short_name

st.set_page_config(page_title="Dr. VLA", layout="wide")
st.title("Dr. VLA")
st.markdown(
    "Browse sparse autoencoder features of Vision-Language-Action models.\n\n"
    "- **Activation Viewer**: pick an episode and see which features fire over time.\n"
    "- **Feature Search**: pick a feature and see the episodes and timesteps that activate it most.\n"
    "- **Feature Classification**: rank features from general to memorized with the generality "
    "classifier, label features and fit your own classifier."
)

cfg = config()
st.subheader("Models")
for model in cfg["models"].values():
    store = get_store(model["activations"])
    index = get_index(model["index"])
    info = store.collection
    with st.container(border=True):
        st.markdown(f"**{model['name']}**  ·  {info['checkpoint']} on {info['dataset']}")
        cols = st.columns(4)
        cols[0].metric("Episodes", len(store))
        cols[1].metric("Timesteps", f"{sum(store.episode(e)['length'] for e in store.episode_ids):,}")
        cols[2].metric("Indexed layers", len(index.layers))
        cols[3].metric("Episodes with frames", sum(store.has_frames(e) for e in store.episode_ids))
        st.caption("Layers: " + ", ".join(
            f"{short_name(layer)} ({index.info['layers'][layer]['num_features']} features)" for layer in index.layers
        ))
st.caption(f"Labels and classifiers you create are saved in `{cfg['workspace']}`.")
