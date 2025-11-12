# Poster Figure Captions

**F1 — Pipeline schematic.** From raw GW-like time series to Takens embedding, VR persistence diagrams, topological feature maps (PI, PL, BC), and machine-learning classifiers, followed by robustness and interpretability analysis.

**F2 — Example persistence diagrams by SNR.** H0 and H1 persistence diagrams for low, medium, and high SNR signals, illustrating the degradation of topological signal structure as noise increases.

**F3 — Topological feature representations.** Example persistence image (PI), persistence landscapes (PL), and Betti curves (BC) for a single GW-like signal, showing how PD information is embedded into fixed-length feature vectors.

**F4 — ROC and PR curves.** Receiver Operating Characteristic (ROC) and Precision–Recall (PR) curves for the best logistic-regression models using PI, PL, BC, and baseline features. AUC and AP values are indicated in the legend.

**F5 — Feature comparison summary.** Radar and bar plots summarizing AUC, AP, F1, and Brier score across PI, PL, BC, and baseline features, averaged over classifiers.

**F6 — Robustness analysis.** Metric degradation as a function of SNR and sensitivity of classifiers to ±10% perturbations in embedding parameters (m, τ), summarized via heatmaps of ΔAUC.

**F7 — Computational profile.** Per-component runtime (Takens embedding, VR PD, PI/PL/BC vectorization) with 95% confidence intervals, highlighting the computational bottlenecks of the TDA pipeline.

**F8 — PI weight maps.** Logistic-regression weight maps over the persistence-image birth–persistence grid for H0 and H1. Red regions indicate PD regions that increase the probability of a GW signal; blue regions decrease it.
