"""Rank features from general to memorized, label features, and fit new classifiers."""

import re

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from common import PAPER_LABELS, classifier_select, config, get_index, go_to_feature, sidebar_model_and_layer
from drvla.classifier import labeled_metric_table, load_labels, save_labels, train_classifier
from drvla.layers import short_name
from drvla.metrics import METRIC_LABELS, METRICS

st.set_page_config(page_title="Feature Classification", layout="wide")
cfg = config()
model, layer = sidebar_model_and_layer(cfg)
index = get_index(model["index"])
workspace = cfg["workspace"]

st.title("Feature Classification")
st.caption(
    "P(general) = sigmoid(b0 + sum_i b_i m_i) over the generality metrics m: episode coverage, mean onset count, "
    "mean activation magnitude and relative run length. Features that never activate are excluded."
)
clf = classifier_select(cfg, model, container=st)
coefficients = ", ".join(f"{m} {clf.coefficients[m]:+.2f}" for m in clf.metrics)
st.caption(f"{clf.description} · intercept {clf.intercept:+.2f} · {coefficients}")

metrics = index.metrics(layer)
active = metrics[metrics["num_active_episodes"] > 0].copy()
active["P(general)"] = clf.predict_proba(active)
general = active["P(general)"] >= 0.5

cols = st.columns(4)
cols[0].metric("Features", len(metrics))
cols[1].metric("Active features", len(active))
cols[2].metric("Classified general", int(general.sum()))
cols[3].metric("Share general", f"{general.mean():.2%}")

left, right = st.columns(2)
hist = px.histogram(active, x="P(general)", nbins=50, log_y=True, title="Distribution of P(general)")
hist.add_vline(x=0.5, line_dash="dash")
hist.update_yaxes(dtick=1, title="features (log scale)")
left.plotly_chart(hist, width="stretch")
scatter = px.scatter(
    active, x="episode_coverage", y="mean_onset_count", color="P(general)", hover_data=["feature_id"],
    color_continuous_scale="Viridis", title="Coverage vs onset count", render_mode="svg",
)
right.plotly_chart(scatter, width="stretch")

SHORT_NAMES = {"feature_id": "feature", "episode_coverage": "coverage", "mean_onset_count": "onsets",
               "mean_activation_magnitude": "magnitude", "relative_run_length": "run length"}
display_columns = ["feature_id", "P(general)", *METRICS]


def ranked_table(title: str, table: pd.DataFrame, key: str):
    st.markdown(f"**{title}**")
    selection = st.dataframe(
        table[display_columns].rename(columns=SHORT_NAMES).style.format(precision=3), on_select="rerun", selection_mode="single-row",
        hide_index=True, width="stretch", key=key,
    )
    if selection.selection.rows:
        feature = int(table.iloc[selection.selection.rows[0]]["feature_id"])
        if st.button(f"Open F{feature} in Feature Search", key=f"{key}_open"):
            go_to_feature(feature)


n_show = st.slider("Features per list", 10, 200, 25)
left, right = st.columns(2)
with left:
    ranked_table("Most general", active.sort_values("P(general)", ascending=False).head(n_show), "most_general")
with right:
    ranked_table("Most memorized", active.sort_values("P(general)").head(n_show), "most_memorized")

# ------------------------------------------------------------------ labeling and training
st.divider()
st.header("Label features and train a classifier")
st.caption(
    f"Labels store each feature's metrics, so a label set can mix layers and models. Label sets and classifiers "
    f"are saved in `{workspace}`; classifiers saved there appear in the classifier menu above."
)
labels: list[dict] = st.session_state.setdefault("labels", [])

label_files = {f"{p.stem} (paper)": p for p in sorted(PAPER_LABELS.glob("*.json"))}
label_files.update({f"{p.stem} (workspace)": p for p in sorted((workspace / "labels").glob("*.json"))})
load_col, add_col = st.columns(2)
with load_col:
    st.markdown("**Load a label set**")
    if label_files:
        choice = st.selectbox("Label set", list(label_files), label_visibility="collapsed")
        mode = st.radio("Mode", ["Replace current labels", "Append"], horizontal=True, label_visibility="collapsed")
        if st.button("Load"):
            loaded = load_labels(label_files[choice])["labels"]
            st.session_state["labels"] = loaded if mode.startswith("Replace") else labels + loaded
            st.rerun()
    else:
        st.caption("No saved label sets.")
with add_col:
    st.markdown(f"**Label a feature of {model['name']} {short_name(layer)}**")
    with st.form("add_label", clear_on_submit=True, border=False):
        feature = st.number_input("Feature", 0, len(metrics) - 1, step=1)
        label = st.radio("Label", ["general", "memorized"], horizontal=True)
        if st.form_submit_button("Add label"):
            row = metrics.iloc[int(feature)]
            if row["num_active_episodes"] == 0:
                st.error(f"F{int(feature)} never activates, so it has no metrics to learn from.")
                st.stop()
            entry = {
                "sae": model["name"], "layer": layer, "feature_id": int(feature), "label": label,
                "metrics": {m: float(row[m]) for m in METRICS},
            }
            key = (entry["sae"], layer, entry["feature_id"])
            labels[:] = [e for e in labels if (e.get("sae"), e["layer"], e["feature_id"]) != key] + [entry]
            st.rerun()

if labels:
    table = pd.DataFrame(
        [{"sae": e.get("sae", ""), "layer": short_name(e["layer"]), "feature": e["feature_id"], "label": e["label"], **e["metrics"]} for e in labels]
    )
    counts = table["label"].value_counts()
    st.markdown(f"**Current labels:** {counts.get('general', 0)} general, {counts.get('memorized', 0)} memorized")
    selection = st.dataframe(
        table.style.format(precision=3), on_select="rerun", selection_mode="multi-row", hide_index=True, width="stretch"
    )
    cols = st.columns([1, 2, 1])
    if cols[0].button("Remove selected", disabled=not selection.selection.rows):
        drop = set(selection.selection.rows)
        st.session_state["labels"] = [e for i, e in enumerate(labels) if i not in drop]
        st.rerun()
    set_name = cols[1].text_input("Save as", placeholder="label set name", label_visibility="collapsed")
    if cols[2].button("Save label set"):
        if not re.fullmatch(r"[A-Za-z0-9_\-]+", set_name or ""):
            st.error("Use letters, digits, '-' and '_' for the name.")
        else:
            save_labels(workspace / "labels" / f"{set_name}.json", labels, description=f"Saved from the dashboard ({len(labels)} labels)")
            st.success(f"Saved {workspace / 'labels' / f'{set_name}.json'}")

    st.markdown("**Train**")
    cols = st.columns([2, 1, 1])
    chosen_metrics = cols[0].multiselect("Metrics", list(METRICS), default=list(METRICS), format_func=METRIC_LABELS.get)
    regularization = cols[1].number_input("C (inverse L2 strength)", 0.001, 1000.0, 1.0, format="%.3f")
    clf_name = cols[2].text_input("Classifier name", placeholder="my_classifier")
    if st.button("Train classifier", type="primary"):
        if not chosen_metrics:
            st.error("Choose at least one metric.")
        elif not re.fullmatch(r"[A-Za-z0-9_\-]+", clf_name or ""):
            st.error("Give the classifier a name made of letters, digits, '-' and '_'.")
        else:
            data = labeled_metric_table(labels)
            try:
                new_clf = train_classifier(
                    data, data["y"].to_numpy(), metrics=chosen_metrics, C=regularization,
                    description=f"{clf_name}: {len(labels)} labels",
                )
            except ValueError as error:
                st.error(str(error))
            else:
                path = workspace / "classifiers" / f"{clf_name}.json"
                new_clf.save(path)
                loo = f"{new_clf.loo_accuracy:.1%}" if new_clf.loo_accuracy is not None else "n/a"
                st.success(f"Saved {path}. Leave-one-out accuracy {loo}; select it in the classifier menu above.")
                coef = pd.DataFrame({"metric": new_clf.metrics, "coefficient": [new_clf.coefficients[m] for m in new_clf.metrics]})
                bar = px.bar(coef, x="metric", y="coefficient", title=f"Coefficients (intercept {new_clf.intercept:+.3f})")
                st.plotly_chart(bar, width="stretch")
else:
    st.info("No labels yet. Load a label set or label features above.")
