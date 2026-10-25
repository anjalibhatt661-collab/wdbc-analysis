#!/usr/bin/env python3
"""Which nuclear features differ between malignant and benign breast masses?

Biological question
-------------------
Do nuclear size, shape and texture measurements from fine-needle-aspirate
images of breast masses differ between malignant and benign diagnoses, and
which of the 30 measured features still show a reliable difference after
controlling the false discovery rate across all 30 tests?

Run everything with one command:

    python analysis.py

Outputs go to results/ (tables) and results/figures/ (plots).
Requires: numpy, pandas, scipy, matplotlib (see requirements.txt).
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# PARAMETERS - all settings live here.
# ---------------------------------------------------------------------------
DATA_CSV = "data/wdbc.csv"      # input table: one row per sample
OUT_DIR = "results"             # all outputs are written here
GROUP_COL = "diagnosis"         # column that holds the group label
CASE_LABEL = "malignant"        # group 1 (effect sizes are oriented case vs control)
CONTROL_LABEL = "benign"        # group 2
ID_COL = "sample_index"         # non-feature column to ignore
FDR_Q = 0.05                    # false discovery rate threshold (Benjamini-Hochberg)
SEED = 42                       # seed for every random step
N_BOOT = 2000                   # bootstrap resamples for AUC confidence intervals
N_PERM = 200                    # label permutations for the null check
DISCOVERY_FRACTION = 0.5        # share of samples used for the discovery half
# ---------------------------------------------------------------------------


# ----------------------------- statistics helpers --------------------------
def bh_adjust(p):
    """Benjamini-Hochberg adjusted p-values (q-values)."""
    p = np.asarray(p, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = p[order] * m / np.arange(1, m + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.clip(adj, 0, 1)
    return out


def bonferroni_adjust(p):
    return np.clip(np.asarray(p, dtype=float) * len(p), 0, 1)


def mwu_all(X, is_case):
    """Mann-Whitney U per column. Returns (p-values, AUC) with AUC = P(case > control)."""
    pvals, aucs = [], []
    for j in range(X.shape[1]):
        a, b = X[is_case, j], X[~is_case, j]
        res = stats.mannwhitneyu(a, b, alternative="two-sided")
        pvals.append(res.pvalue)
        aucs.append(res.statistic / (len(a) * len(b)))
    return np.array(pvals), np.array(aucs)


def auc_matrix(X, is_case):
    """Fast AUC for every column using ranks (ties get average ranks)."""
    ranks = stats.rankdata(X, axis=0)
    n1, n0 = is_case.sum(), (~is_case).sum()
    return (ranks[is_case].sum(axis=0) - n1 * (n1 + 1) / 2) / (n1 * n0)


def bootstrap_auc_ci(X, is_case, rng):
    """Percentile 95% CI for the AUC of each feature, resampling within each group."""
    idx_case, idx_ctrl = np.where(is_case)[0], np.where(~is_case)[0]
    labels = np.r_[np.ones(len(idx_case), bool), np.zeros(len(idx_ctrl), bool)]
    boots = np.empty((N_BOOT, X.shape[1]))
    for b in range(N_BOOT):
        rows = np.r_[rng.choice(idx_case, len(idx_case)), rng.choice(idx_ctrl, len(idx_ctrl))]
        boots[b] = auc_matrix(X[rows], labels)
    return np.percentile(boots, 2.5, axis=0), np.percentile(boots, 97.5, axis=0)


# ----------------------------- analysis steps ------------------------------
def load_data():
    df = pd.read_csv(DATA_CSV)
    features = [c for c in df.columns if c not in (GROUP_COL, ID_COL)]
    assert df[features].isna().sum().sum() == 0, "missing values found"
    assert set(df[GROUP_COL]) == {CASE_LABEL, CONTROL_LABEL}, "unexpected group labels"
    is_case = (df[GROUP_COL] == CASE_LABEL).to_numpy()
    return df, features, df[features].to_numpy(float), is_case


def main_tests(df, features, X, is_case, rng):
    p, auc = mwu_all(X, is_case)
    lo, hi = bootstrap_auc_ci(X, is_case, rng)
    res = pd.DataFrame({
        "feature": features,
        f"median_{CASE_LABEL}": np.median(X[is_case], axis=0),
        f"median_{CONTROL_LABEL}": np.median(X[~is_case], axis=0),
        "auc": auc, "auc_ci_low": lo, "auc_ci_high": hi,
        "cliffs_delta": 2 * auc - 1,
        "p_raw": p,
        "q_bh": bh_adjust(p),
        "p_bonferroni": bonferroni_adjust(p),
    })
    res["significant_bh"] = res["q_bh"] < FDR_Q
    res["significant_bonferroni"] = res["p_bonferroni"] < FDR_Q
    res["significant_uncorrected"] = res["p_raw"] < FDR_Q
    return res.sort_values("auc", key=lambda s: (s - 0.5).abs(), ascending=False).reset_index(drop=True)


def null_check(X, is_case, rng):
    """Shuffle the diagnosis labels; a sound procedure should find ~nothing."""
    rows = []
    for k in range(N_PERM):
        perm = rng.permutation(is_case)
        p, _ = mwu_all(X, perm)
        rows.append({"permutation": k + 1,
                     "uncorrected_p_below_alpha": int((p < FDR_Q).sum()),
                     "bh_discoveries": int((bh_adjust(p) < FDR_Q).sum())})
    return pd.DataFrame(rows)


def replication(features, X, is_case, rng):
    """Pick features in a discovery half, then test only those in the held-out half."""
    disc = np.zeros(len(is_case), bool)
    for flag in (True, False):  # stratified split
        idx = np.where(is_case == flag)[0]
        disc[rng.choice(idx, int(round(len(idx) * DISCOVERY_FRACTION)), replace=False)] = True
    p_d, auc_d = mwu_all(X[disc], is_case[disc])
    q_d = bh_adjust(p_d)
    chosen = np.where(q_d < FDR_Q)[0]
    out = pd.DataFrame({"feature": np.array(features)[chosen], "auc_discovery": auc_d[chosen],
                        "q_discovery": q_d[chosen]})
    if len(chosen):
        p_v, auc_v = mwu_all(X[~disc][:, chosen], is_case[~disc])
        q_v = bh_adjust(p_v)  # corrected over the features chosen in discovery
        out["auc_validation"] = auc_v
        out["q_validation"] = q_v
        out["replicated"] = (q_v < FDR_Q) & ((auc_v > 0.5) == (auc_d[chosen] > 0.5))
    return out, int(disc.sum()), int((~disc).sum())


def dimensionality(X):
    corr = pd.DataFrame(X).corr(method="spearman").to_numpy()
    eig = np.sort(np.linalg.eigvalsh(corr))[::-1]
    cum = np.cumsum(eig) / eig.sum()
    n_pc90 = int(np.searchsorted(cum, 0.90) + 1)
    iu = np.triu_indices_from(corr, k=1)
    n_pairs_high = int((np.abs(corr[iu]) > 0.9).sum())
    return corr, n_pc90, n_pairs_high, len(iu[0])


# ----------------------------- figures -------------------------------------
def figure_forest(res, path):
    d = res.sort_values("auc")
    fig, ax = plt.subplots(figsize=(7, 8))
    colors = np.where(d["significant_bh"], "#c0392b", "#7f8c8d")
    ax.hlines(range(len(d)), d["auc_ci_low"], d["auc_ci_high"], color=colors, lw=2)
    ax.scatter(d["auc"], range(len(d)), color=colors, zorder=3, s=22)
    ax.axvline(0.5, color="black", lw=1, ls="--")
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels(d["feature"], fontsize=8)
    ax.set_xlabel(f"AUC = P({CASE_LABEL} value > {CONTROL_LABEL} value), 95% bootstrap CI")
    ax.set_title(f"Effect size per feature (red = significant at BH FDR {FDR_Q})")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def figure_boxplots(df, res, path, k=6):
    top = res.head(k)["feature"].tolist()
    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    for ax, feat in zip(axes.ravel(), top):
        data = [df.loc[df[GROUP_COL] == g, feat] for g in (CONTROL_LABEL, CASE_LABEL)]
        ax.boxplot(data, tick_labels=[CONTROL_LABEL, CASE_LABEL], showfliers=True)
        ax.set_title(feat, fontsize=10)
    fig.suptitle(f"Top {k} features by effect size")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def figure_corr(corr, features, path):
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(features)))
    ax.set_yticks(range(len(features)))
    ax.set_xticklabels(features, rotation=90, fontsize=6)
    ax.set_yticklabels(features, fontsize=6)
    fig.colorbar(im, label="Spearman correlation")
    ax.set_title("Feature correlation: the 30 tests are not independent")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ----------------------------- main ----------------------------------------
def main():
    rng = np.random.default_rng(SEED)
    os.makedirs(os.path.join(OUT_DIR, "figures"), exist_ok=True)

    print(f"[1/6] Loading {DATA_CSV}")
    df, features, X, is_case = load_data()
    n1, n0 = int(is_case.sum()), int((~is_case).sum())
    print(f"      {len(df)} samples ({n1} {CASE_LABEL}, {n0} {CONTROL_LABEL}), {len(features)} features")

    print("[2/6] Mann-Whitney test per feature + multiple-testing correction")
    res = main_tests(df, features, X, is_case, rng)
    res.to_csv(os.path.join(OUT_DIR, "feature_tests.csv"), index=False, float_format="%.6g")

    print(f"[3/6] Null check: {N_PERM} label permutations")
    null = null_check(X, is_case, rng)
    null.to_csv(os.path.join(OUT_DIR, "null_check.csv"), index=False)

    print("[4/6] Discovery / validation split")
    rep, n_disc, n_val = replication(features, X, is_case, rng)
    rep.to_csv(os.path.join(OUT_DIR, "replication.csv"), index=False, float_format="%.6g")

    print("[5/6] Correlation structure")
    corr, n_pc90, n_high, n_pairs = dimensionality(X)

    print("[6/6] Figures and summary")
    figure_forest(res, os.path.join(OUT_DIR, "figures", "effect_sizes.png"))
    figure_boxplots(df, res, os.path.join(OUT_DIR, "figures", "top_features_boxplots.png"))
    figure_corr(corr, features, os.path.join(OUT_DIR, "figures", "feature_correlation.png"))

    sig = lambda col: int(res[col].sum())
    lines = [
        "SUMMARY",
        f"samples: {len(df)} ({n1} {CASE_LABEL}, {n0} {CONTROL_LABEL}); features tested: {len(features)}",
        f"significant, uncorrected p < {FDR_Q}: {sig('significant_uncorrected')}",
        f"significant, Benjamini-Hochberg FDR < {FDR_Q}: {sig('significant_bh')}",
        f"significant, Bonferroni < {FDR_Q}: {sig('significant_bonferroni')}",
        "not significant after BH: " + (", ".join(res.loc[~res['significant_bh'], 'feature']) or "none"),
        f"strongest effect: {res.loc[0, 'feature']} (AUC {res.loc[0, 'auc']:.3f})",
        f"weakest effect: {res.loc[len(res) - 1, 'feature']} (AUC {res.loc[len(res) - 1, 'auc']:.3f})",
        f"null check ({N_PERM} permutations): mean uncorrected hits {null['uncorrected_p_below_alpha'].mean():.2f}, "
        f"mean BH discoveries {null['bh_discoveries'].mean():.3f}, "
        f"permutations with >=1 BH discovery {int((null['bh_discoveries'] > 0).sum())}",
        f"split-half: discovery n={n_disc}, validation n={n_val}; "
        f"selected {len(rep)} features; replicated "
        f"{int(rep['replicated'].sum()) if len(rep) else 0}",
        f"correlation: {n_high} of {n_pairs} feature pairs have |Spearman rho| > 0.9; "
        f"{n_pc90} principal components explain 90% of the variance",
    ]
    with open(os.path.join(OUT_DIR, "summary.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
