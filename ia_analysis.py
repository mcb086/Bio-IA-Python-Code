"""Reproduce the Biology IA reanalysis from the supplied, merged CSV.

Run: python ia_analysis.py "asd_microbiome_combined_genus_wide(1).csv" results
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import re

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import scipy
from scipy import linalg, special, stats
from scipy.spatial.distance import pdist, squareform


N_PERM = 9_999
SEED = 20261111
MIN_PREVALENCE = 0.05
MIN_ABUNDANCE = 0.0001
PSEUDOCOUNTS = (1e-7, 1e-6, 1e-5)
PRIMARY_C = 1e-6
ALPHA_NAMES = ("Shannon", "Simpson", "Retained richness")
DISTANCE_NAMES = ("Bray-Curtis", "Jaccard")
STUDY_LABEL = {
    "GSE113690_Kaggle": "Dan",
    "GSE244391": "Carmel",
    "ScientificReports_2018": "Zhang",
    "Vernocchi_2022": "Vernocchi",
}
UNRESOLVED = re.compile(
    r"unclass|norank|uncultur|unknown|unassign|unident|mitochondri|chloroplast"
    r"|^(?:bacteria|archaea|unmapped)$"
    r"|^(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)-\d+$",
    re.IGNORECASE,
)


def load_input(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        header = next(csv.reader(handle))
    if header[:3] != ["study", "sample_id", "group"] or len(header) != len(set(header)):
        raise ValueError("Expected unique columns starting with study, sample_id and group.")
    data = pd.read_csv(path, dtype={"study": str, "sample_id": str, "group": str})
    metadata = data.iloc[:, :3]
    if metadata.isna().any().any() or metadata.duplicated(["study", "sample_id"]).any():
        raise ValueError("Missing metadata or duplicate study/sample ID.")
    if set(data.group) != {"ASD", "Control"} or set(data.study) != set(STUDY_LABEL):
        raise ValueError("Unexpected diagnosis or study labels.")
    if (pd.crosstab(data.study, data.group) < 2).any().any():
        raise ValueError("Each study needs at least two rows per diagnosis.")
    values = data.iloc[:, 3:].astype(float)
    if not np.isfinite(values.to_numpy()).all() or (values < 0).any().any():
        raise ValueError("Abundances must be finite and nonnegative.")
    totals = values.sum(axis=1)
    if (totals <= 0).any() or not np.allclose(totals, 1, atol=1e-6, rtol=0):
        raise ValueError("Expected rows of proportions summing to approximately one.")
    return metadata, values, values.div(totals, axis=0)


def select_panels(proportions, study):
    detected = proportions.gt(0)
    prevalence = detected.groupby(study).mean()
    unresolved = proportions.columns.to_series().map(lambda name: bool(UNRESOLVED.search(name)))
    pooled = detected.mean() >= MIN_PREVALENCE
    pooled &= proportions.mean() >= MIN_ABUNDANCE
    broad = pooled & ~unresolved
    shared = broad & prevalence.ge(MIN_PREVALENCE).all(axis=0)
    coverage = pd.DataFrame(
        {
            "pooled_prevalence": detected.mean(),
            "pooled_mean_proportion": proportions.mean(),
            "excluded_label": unresolved,
            "pooled_original": pooled,
            "pooled_clean": broad,
            "shared": shared,
        }
    ).join(prevalence.T.add_prefix("prevalence_"))
    panels = {
        "Shared 36": proportions.loc[:, shared],
        "Pooled clean": proportions.loc[:, broad],
        "Pooled original": proportions.loc[:, pooled],
    }
    return panels, coverage


def normalize(proportions):
    totals = proportions.sum(axis=1)
    if (totals <= 0).any():
        raise ValueError("At least one row has no detected feature in this panel.")
    return proportions.div(totals, axis=0)


def alpha_metrics(p):
    return np.column_stack(
        (-special.xlogy(p, p).sum(axis=1), 1 - np.square(p).sum(axis=1), (p > 0).sum(axis=1))
    )


def distance_matrices(p):
    return {
        "Bray-Curtis": squareform(pdist(p, metric="braycurtis")),
        "Jaccard": squareform(pdist(p > 0, metric="jaccard")),
    }


def gower(squared_distances):
    row_mean = squared_distances.mean(axis=1, keepdims=True)
    return -0.5 * (squared_distances - row_mean - row_mean.T + row_mean.mean())


def clr(p, c):
    logs = np.log(p + c)
    return logs - logs.mean(axis=1, keepdims=True)


class BlockedModel:
    """Fixed study intercepts with a common ASD coefficient and blocked label permutations."""

    def __init__(self, study, group):
        self.blocks = [np.flatnonzero(study == name) for name in np.unique(study)]
        self.n = len(group)
        self.df = self.n - len(self.blocks) - 1
        self.group = group.astype(float)
        self.z = self.group.copy()
        self.base_leverage = np.empty(self.n)
        self.permuted = np.empty((N_PERM, self.n))
        rng = np.random.default_rng(SEED)
        for indices in self.blocks:
            self.z[indices] -= self.group[indices].mean()
            self.base_leverage[indices] = 1 / len(indices)
            rows = np.broadcast_to(self.z[indices], (N_PERM, len(indices)))
            self.permuted[:, indices] = rng.permuted(rows, axis=1)
        self.denominator = self.z @ self.z

    def regression(self, outcomes, names):
        """OLS slopes and HC3 intervals. Permute labels and recompute absolute HC3 t."""
        y = np.asarray(outcomes, dtype=float).reshape(self.n, -1).copy()
        for indices in self.blocks:
            y[indices] -= y[indices].mean(axis=0)
        beta = self.z @ y / self.denominator
        h = self.base_leverage + self.z**2 / self.denominator
        residual = y - self.z[:, None] * beta
        se = np.linalg.norm(self.z[:, None] * residual / (1 - h[:, None]), axis=0)
        se /= self.denominator
        abs_t = np.divide(np.abs(beta), se, out=np.zeros_like(beta), where=se > 0)
        exceedances = np.zeros(len(names), dtype=int)
        for z in np.array_split(self.permuted, 20):
            b = z @ y / self.denominator
            u = (z / (1 - self.base_leverage - z**2 / self.denominator)) ** 2
            ss = u @ (y**2) - 2 * b * ((u * z) @ y) + b**2 * np.sum(u * z**2, axis=1)[:, None]
            perm_se = np.sqrt(np.maximum(ss, 0)) / self.denominator
            t_perm = np.divide(np.abs(b), perm_se, out=np.zeros_like(b), where=perm_se > 0)
            exceedances += np.sum(t_perm >= abs_t - 1e-10, axis=0)
        t_critical = stats.t.ppf(0.975, self.df)
        return pd.DataFrame(
            {
                "outcome": names,
                "n": self.n,
                "n_asd": int(self.group.sum()),
                "df": self.df,
                "beta": beta,
                "hc3_se": se,
                "ci_low": beta - t_critical * se,
                "ci_high": beta + t_critical * se,
                "t": np.sign(beta) * abs_t,
                "p": (exceedances + 1) / (N_PERM + 1),
            }
        )

    def permanova(self, d):
        """Study is entered first. Test ASD using the change in explained sums of squares."""
        g = gower(d**2)
        total = float(np.trace(g))
        study_ss = sum(g[np.ix_(ix, ix)].sum() / len(ix) for ix in self.blocks)
        status_ss = float(self.z @ g @ self.z / self.denominator)
        residual = total - study_ss - status_ss
        f = status_ss / (residual / self.df)
        exceeded = 0
        for z in np.array_split(self.permuted, 20):
            ss = np.einsum("ij,ij->i", z @ g, z) / self.denominator
            perm_f = ss / ((total - study_ss - ss) / self.df)
            exceeded += np.count_nonzero(perm_f >= f - 1e-10)
        return {
            "n": self.n,
            "df_status": 1,
            "df_residual": self.df,
            "ss_total": total,
            "ss_study": study_ss,
            "ss_asd": status_ss,
            "ss_residual": residual,
            "study_r2": study_ss / total,
            "status_r2": status_ss / total,
            "status_partial_r2": status_ss / (total - study_ss),
            "pseudo_f": f,
            "p": (1 + exceeded) / (N_PERM + 1),
        }

    def dispersion(self, d):
        """Compare bias-corrected distances to centroids of square-root dissimilarities."""
        values = np.zeros(self.n)
        for group in (0, 1):
            indices = np.flatnonzero(self.group == group)
            block = d[np.ix_(indices, indices)]
            squared_centroid_distance = block.mean(axis=1) - block.mean() / 2
            values[indices] = np.sqrt(
                np.maximum(squared_centroid_distance, 0) * len(indices) / (len(indices) - 1)
            )
        estimate = self.regression(values, ["dispersion"]).iloc[0].to_dict()
        estimate["mean_asd"] = float(values[self.group == 1].mean())
        estimate["mean_control"] = float(values[self.group == 0].mean())
        return estimate


def adjust(frame, method, label):
    frame = frame.copy()
    frame[label] = stats.false_discovery_control(frame.p, method="bh") if method == "bh" else holm(frame.p)
    return frame


def holm(p):
    p = np.asarray(p)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    adjusted[order] = np.minimum(1, np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    return adjusted


def pcoa(d):
    eigenvalues, vectors = linalg.eigh(gower(d**2))
    positive_total = eigenvalues[eigenvalues > 0].sum()
    negative_total = -eigenvalues[eigenvalues < 0].sum()
    coordinates = vectors[:, -1:-3:-1] * np.sqrt(eigenvalues[-1:-3:-1])
    return coordinates, eigenvalues[-1:-3:-1] / positive_total, negative_total / positive_total


def make_pcoa_plot(distances, studies, group, path):
    colors = ("#007279", "#b7471a", "#6249a0", "#566a26")
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.35), layout="constrained")
    diagnostics = []
    for ax, (metric, d) in zip(axes, distances.items()):
        coordinates, shares, negative = pcoa(d)
        diagnostics.append({"distance": metric, "axis_1": shares[0], "axis_2": shares[1], "negative": negative})
        for (code, name), color in zip(STUDY_LABEL.items(), colors):
            for status, marker in ((0, "o"), (1, "^")):
                rows = (studies == code) & (group == status)
                ax.scatter(*coordinates[rows].T, marker=marker, color=color, alpha=0.61, s=15, linewidths=0)
        ax.set_title(metric, fontsize=11, fontweight="semibold")
        ax.set_xlabel(f"PCoA 1 ({shares[0]:.1%})", fontsize=9)
        ax.set_ylabel(f"PCoA 2 ({shares[1]:.1%})", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    study_key = [
        Line2D([0], [0], marker="s", ls="", color=color, label=name, ms=6)
        for name, color in zip(STUDY_LABEL.values(), colors)
    ]
    status_key = [
        Line2D([0], [0], marker=marker, ls="", color="#40454c", label=name, ms=6)
        for name, marker in (("Control", "o"), ("ASD", "^"))
    ]
    fig.legend(handles=study_key + status_key, loc="outside lower center", ncol=6, frameon=False, fontsize=8)
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)
    return diagnostics


def make_forest_plot(cohorts, alpha, taxa, path):
    codes = list(STUDY_LABEL) + ["All studies"]
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.05), sharey=True, layout="constrained")
    for ax, outcome, pooled, title, xlabel in (
        (axes[0], "Shannon", alpha, "Shannon diversity", "ASD − control (Shannon units)"),
        (axes[1], "Bacteroides", taxa, "Bacteroides", "ASD − control (CLR log units)"),
    ):
        data = cohorts.loc[cohorts.outcome == outcome].set_index("study").loc[list(STUDY_LABEL)]
        data = pd.concat([data, pooled.loc[pooled.outcome == outcome].assign(study="All studies").set_index("study")])
        for i, row in enumerate(data.itertuples()):
            color = "#0a6870" if i == len(data) - 1 else "#263747"
            ax.errorbar(
                row.beta, i, xerr=[[row.beta - row.ci_low], [row.ci_high - row.beta]],
                marker="D" if i == len(data) - 1 else "o", ms=5, color=color, capsize=3, lw=1.2,
            )
        ax.axvline(0, lw=0.9, linestyle="--", color="#9099a1", zorder=0)
        ax.set_yticks(range(len(codes)), [STUDY_LABEL.get(name, name) for name in codes], fontsize=9)
        ax.set_ylim(len(codes) - 0.4, -0.6)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_title(title, fontsize=11, fontweight="semibold", fontstyle="italic" if outcome == "Bacteroides" else "normal")
        ax.tick_params(axis="x", labelsize=8)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)


def main(input_path, output):
    output.mkdir(parents=True, exist_ok=True)
    meta, raw, fractions = load_input(input_path)
    studies = meta.study.to_numpy()
    group = meta.group.eq("ASD").to_numpy(dtype=int)
    panels, coverage = select_panels(fractions, meta.study)
    primary_name = next(iter(panels))
    main_model = BlockedModel(studies, group)
    unadjusted_model = BlockedModel(np.repeat("Pooled", len(group)), group)
    alpha_results, beta_results, unadjusted_results = [], [], []
    for panel, data in panels.items():
        normalized = normalize(data).to_numpy()
        alpha = alpha_metrics(normalized)
        alpha_results.append(adjust(main_model.regression(alpha, ALPHA_NAMES), "holm", "p_holm").assign(panel=panel))
        unadjusted_results.append(
            adjust(unadjusted_model.regression(alpha, ALPHA_NAMES), "holm", "p_holm").assign(panel=panel)
        )
        distance = distance_matrices(normalized)
        beta = pd.DataFrame([main_model.permanova(d) | {"distance": metric} for metric, d in distance.items()])
        beta_results.append(adjust(beta, "holm", "p_holm").assign(panel=panel))
        if panel == primary_name:
            primary_alpha, primary_distances = alpha, distance
            primary_p = normalized
    alpha_results = pd.concat(alpha_results, ignore_index=True)
    unadjusted_results = pd.concat(unadjusted_results, ignore_index=True)
    beta_results = pd.concat(beta_results, ignore_index=True)
    shared = panels[primary_name]
    taxa = pd.concat(
        [
            adjust(main_model.regression(clr(primary_p, c), shared.columns), "bh", "q_bh").assign(pseudocount=c)
            for c in PSEUDOCOUNTS
        ],
        ignore_index=True,
    )
    primary_taxa = taxa.loc[taxa.pseudocount == PRIMARY_C].copy()
    primary_clr = clr(primary_p, PRIMARY_C)
    alpha_and_clr = np.column_stack([primary_alpha, primary_clr])
    endpoints = (*ALPHA_NAMES, *shared.columns)
    cohort_effects, cohort_beta, dispersion, leave_out, leave_out_beta = [], [], [], [], []
    for study in STUDY_LABEL:
        keep = studies == study
        cohort = BlockedModel(studies[keep], group[keep])
        cohort_effects.append(cohort.regression(alpha_and_clr[keep], endpoints).assign(study=study))
        rest = ~keep
        rest_model = BlockedModel(studies[rest], group[rest])
        leave_out.append(rest_model.regression(alpha_and_clr[rest], endpoints).assign(omitted_study=study))
        for metric, d in primary_distances.items():
            this_distance = d[np.ix_(keep, keep)]
            cohort_beta.append(cohort.permanova(this_distance) | {"study": study, "distance": metric})
            dispersion.append(cohort.dispersion(this_distance) | {"study": study, "distance": metric})
            leave_out_beta.append(
                rest_model.permanova(d[np.ix_(rest, rest)]) | {"omitted_study": study, "distance": metric}
            )
    cohort_effects = pd.concat(cohort_effects, ignore_index=True)
    dispersion = adjust(pd.DataFrame(dispersion), "holm", "p_holm")
    descriptives = meta.join(pd.DataFrame(primary_alpha, columns=ALPHA_NAMES))
    raw_extract = meta.join(raw[["Bacteroides", "Prevotella", "Sutterella"]]).assign(
        shared_proportion=shared.sum(axis=1), renormalized_Bacteroides=primary_p[:, shared.columns.get_loc("Bacteroides")]
    )
    outputs = {
        "feature_coverage": coverage.rename_axis("feature").reset_index(),
        "sample_alpha": descriptives,
        "raw_extract": raw_extract,
        "alpha": alpha_results,
        "alpha_unadjusted": unadjusted_results,
        "permanova": beta_results,
        "dispersion": dispersion,
        "genus": primary_taxa.sort_values(["q_bh", "p", "outcome"]),
        "pseudocount_sensitivity": taxa,
        "cohort_effects": cohort_effects,
        "cohort_permanova": pd.DataFrame(cohort_beta),
        "leave_one_study_out": pd.concat(leave_out, ignore_index=True),
        "leave_one_study_out_permanova": pd.DataFrame(leave_out_beta),
    }
    for name, table in outputs.items():
        table.to_csv(output / f"{name}.csv", index=False)
    diagnostic = make_pcoa_plot(primary_distances, studies, group, output / "pcoa.png")
    make_forest_plot(
        cohort_effects, alpha_results.loc[alpha_results.panel == primary_name], primary_taxa, output / "cohort_effects.png"
    )
    summary = {
        "input_name": input_path.name,
        "sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "n": len(meta),
        "source_columns": raw.shape[1],
        "rows_sum_min": float(raw.sum(axis=1).min()),
        "rows_sum_max": float(raw.sum(axis=1).max()),
        "identical_profiles": int(raw.duplicated().sum()),
        "panel_sizes": {name: int(panel.shape[1]) for name, panel in panels.items()},
        "primary_min_positive": float(primary_p[primary_p > 0].min()),
        "pcoa": diagnostic,
        "settings": {"permutations": N_PERM, "seed": SEED, "pseudocount": PRIMARY_C},
        "versions": {
            "Python": platform.python_version(),
            "NumPy": np.__version__,
            "pandas": pd.__version__,
            "SciPy": scipy.__version__,
            "Matplotlib": mpl.__version__,
        },
    }
    (output / "run.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Supplied combined abundance CSV")
    parser.add_argument("output", type=Path, help="Directory for result tables and figures")
    args = parser.parse_args()
    main(args.input, args.output)
