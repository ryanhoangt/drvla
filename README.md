# DR.VLA

**Sparse Autoencoders Reveal Interpretable and Steerable Features in VLA Models**

Aiden Swann, Lachlain McGranahan, Hugo Buurmeijer, Monroe Kennedy III, Mac Schwager

Conference on Robot Learning (CoRL), 2026

[Project Page](https://drvla.github.io) | [Paper](https://drvla.github.io/drvla.pdf) | [arXiv](https://arxiv.org/abs/2603.19183)

---

## Code coming October 1, 2026

The full implementation is being prepared for release and will be published in
this repository on **October 1, 2026**. Watch or star the repo to be notified.

---

## Overview

We train Sparse Autoencoders (SAEs) on the hidden-layer activations of
Vision-Language-Action (VLA) models, surfacing interpretable features for motion
primitives and semantic concepts, plus a metric that distinguishes general
transferable features from episode-specific memorizations. Steering experiments
on the LIBERO simulation benchmark and on real-world DROID hardware show these
features causally drive behavior: amplifying general and semantic features
induces predictable actions, while ablating them destroys performance.

## What will be released

- [ ] SAE training code and architectures for VLA activations
- [ ] Activation collection pipeline for Pi0.5 (PaliGemma backbone and action expert)
- [ ] Feature generality metric and the general/memorized classifier

## Citation

```bibtex
@inproceedings{swann2026sparse,
  title     = {Sparse Autoencoders Reveal Interpretable and Steerable
               Features in VLA Models},
  author    = {Swann, Aiden and McGranahan, Lachlain and Buurmeijer, Hugo
               and Kennedy III, Monroe and Schwager, Mac},
  booktitle = {Conference on Robot Learning (CoRL)},
  year      = {2026}
}
```
