"""Training-only exploratory analysis with deterministic, reviewable artifacts."""

from __future__ import annotations

import io
import json
import math
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib as mpl
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from predictive_maintenance.config import (
    CV_FOLDS,
    DEFAULT_EDA_REPORT_DIR,
    DEFAULT_PROCESSED_DATA_DIR,
    DEFAULT_SPLIT_MANIFEST_PATH,
    HOLDOUT_FRACTION,
    PRIMARY_METRIC,
    RANDOM_SEED,
    SECONDARY_METRIC,
    THRESHOLD_STRATEGY,
    TIE_TOLERANCE,
)
from predictive_maintenance.splitting import load_training_partition
from predictive_maintenance.validation import FEATURE_COLUMNS, TARGET_COLUMN

NUMERIC_FEATURES = FEATURE_COLUMNS[1:]
TYPE_ORDER = ("L", "M", "H")
NEGATIVE_COLOR = "#4C78A8"
POSITIVE_COLOR = "#D1495B"
ACCENT_COLOR = "#2A9D8F"
TEXT_COLOR = "#263238"
GRID_COLOR = "#DCE3E8"
FIGURE_FILENAMES = (
    "01_target_prevalence.png",
    "02_type_support_and_rate.png",
    "03_numeric_ecdf_by_target.png",
    "04_spearman_correlation.png",
    "05_joint_relationships.png",
    "06_positive_rate_by_quintile.png",
)


@dataclass(frozen=True)
class EDAArtifacts:
    """Versionable outputs generated exclusively from the training partition."""

    report: Path
    summary: Path
    figures: tuple[Path, ...]
    training_rows: int
    positive_rows: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible command result."""
        return {
            "report": str(self.report),
            "summary": str(self.summary),
            "figures": [str(path) for path in self.figures],
            "training_rows": self.training_rows,
            "positive_rows": self.positive_rows,
        }


def _new_figure(width: float, height: float) -> Figure:
    figure = Figure(figsize=(width, height), layout="constrained", facecolor="white")
    FigureCanvasAgg(figure)
    return figure


def _style_axis(axis: Axes, *, grid_axis: str = "both") -> None:
    axis.set_facecolor("#FAFBFC")
    axis.tick_params(colors=TEXT_COLOR, labelsize=9)
    axis.xaxis.label.set_color(TEXT_COLOR)
    axis.yaxis.label.set_color(TEXT_COLOR)
    axis.title.set_color(TEXT_COLOR)
    axis.grid(True, axis=grid_axis, color=GRID_COLOR, linewidth=0.7, alpha=0.8)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color(GRID_COLOR)
    axis.spines["bottom"].set_color(GRID_COLOR)


def _write_bytes_if_changed(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.read_bytes() == content:
        return
    if path.exists() and not path.is_file():
        raise OSError(f"Expected a regular output file: {path}")
    with tempfile.NamedTemporaryFile(
        mode="wb",
        prefix=".eda-",
        suffix=path.suffix,
        dir=path.parent,
        delete=False,
    ) as temporary_file:
        temporary_file.write(content)
        temporary_path = Path(temporary_file.name)
    try:
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _write_text_if_changed(path: Path, content: str) -> None:
    _write_bytes_if_changed(path, content.encode("utf-8"))


def _save_figure(figure: Figure, path: Path) -> None:
    image = io.BytesIO()
    figure.savefig(
        image,
        format="png",
        dpi=160,
        facecolor="white",
        metadata={"Software": "Machine Failure Risk Classifier"},
    )
    _write_bytes_if_changed(path, image.getvalue())


def _wilson_interval(positive: int, total: int, z_value: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    proportion = positive / total
    denominator = 1.0 + z_value**2 / total
    center = (proportion + z_value**2 / (2.0 * total)) / denominator
    margin = (
        z_value
        * math.sqrt(proportion * (1.0 - proportion) / total + z_value**2 / (4.0 * total**2))
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def _type_summary(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for product_type in TYPE_ORDER:
        subset = frame.loc[frame["Type"] == product_type, TARGET_COLUMN]
        total = int(len(subset))
        positive = int(subset.sum())
        rate = positive / total
        interval_low, interval_high = _wilson_interval(positive, total)
        rows.append(
            {
                "type": product_type,
                "rows": total,
                "positive": positive,
                "positive_rate": rate,
                "wilson_95_low": interval_low,
                "wilson_95_high": interval_high,
            }
        )
    return rows


def _numeric_summary(frame: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for column in NUMERIC_FEATURES:
        overall = frame[column]
        by_target: dict[str, Any] = {}
        for target_value, label in ((0, "no_failure"), (1, "failure")):
            values = frame.loc[frame[TARGET_COLUMN] == target_value, column]
            by_target[label] = {
                "rows": int(len(values)),
                "min": float(values.min()),
                "q1": float(values.quantile(0.25)),
                "median": float(values.median()),
                "q3": float(values.quantile(0.75)),
                "max": float(values.max()),
            }
        summary[column] = {
            "overall": {
                "min": float(overall.min()),
                "q1": float(overall.quantile(0.25)),
                "median": float(overall.median()),
                "q3": float(overall.quantile(0.75)),
                "max": float(overall.max()),
            },
            "by_target": by_target,
        }
    return summary


def _correlation_summary(frame: pd.DataFrame) -> dict[str, dict[str, float]]:
    correlations = frame.loc[:, NUMERIC_FEATURES].corr(method="spearman")
    return {
        row: {column: float(correlations.loc[row, column]) for column in NUMERIC_FEATURES}
        for row in NUMERIC_FEATURES
    }


def _quintile_summary(frame: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {}
    for column in NUMERIC_FEATURES:
        bins, bin_edges = pd.qcut(frame[column], q=5, duplicates="drop", retbins=True)
        grouped = frame.groupby(bins, observed=True)[TARGET_COLUMN].agg(["count", "sum", "mean"])
        rows: list[dict[str, Any]] = []
        for quintile_index, (_, values) in enumerate(grouped.iterrows()):
            total = int(values["count"])
            positive = int(values["sum"])
            interval_low, interval_high = _wilson_interval(positive, total)
            rows.append(
                {
                    "quintile": quintile_index + 1,
                    "lower_bound": float(bin_edges[quintile_index]),
                    "upper_bound": float(bin_edges[quintile_index + 1]),
                    "lower_inclusive": quintile_index == 0,
                    "upper_inclusive": True,
                    "rows": total,
                    "positive": positive,
                    "positive_rate": float(values["mean"]),
                    "wilson_95_low": interval_low,
                    "wilson_95_high": interval_high,
                }
            )
        output[column] = rows
    return output


def _plot_target_prevalence(frame: pd.DataFrame, path: Path) -> None:
    counts = frame[TARGET_COLUMN].value_counts().reindex([0, 1], fill_value=0)
    total = int(counts.sum())
    figure = _new_figure(7.2, 4.6)
    axis = figure.subplots()
    bars = axis.bar(
        ["No failure", "Failure"],
        counts.to_list(),
        color=[NEGATIVE_COLOR, POSITIVE_COLOR],
        width=0.62,
    )
    for bar, count in zip(bars, counts, strict=True):
        axis.annotate(
            f"{int(count):,}\n({count / total:.2%})",
            (bar.get_x() + bar.get_width() / 2.0, bar.get_height()),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            va="bottom",
            color=TEXT_COLOR,
            fontsize=10,
        )
    axis.set_title("Target distribution — training only", fontsize=13, weight="bold")
    axis.set_ylabel("Observations")
    axis.set_ylim(0, max(counts) * 1.14)
    _style_axis(axis, grid_axis="y")
    _save_figure(figure, path)


def _plot_type_support_and_rate(frame: pd.DataFrame, path: Path) -> None:
    summary = _type_summary(frame)
    figure = _new_figure(11.5, 4.8)
    axes = figure.subplots(1, 2)
    types = [row["type"] for row in summary]
    supports = [row["rows"] for row in summary]
    rates = [row["positive_rate"] for row in summary]

    support_bars = axes[0].bar(types, supports, color=NEGATIVE_COLOR, width=0.62)
    for bar, support in zip(support_bars, supports, strict=True):
        axes[0].annotate(
            f"{support:,}",
            (bar.get_x() + bar.get_width() / 2.0, bar.get_height()),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            color=TEXT_COLOR,
        )
    axes[0].set_title("Support by Type")
    axes[0].set_xlabel("Type")
    axes[0].set_ylabel("Observations")
    axes[0].set_ylim(0, max(supports) * 1.15)
    _style_axis(axes[0], grid_axis="y")

    lower_errors = [rate - row["wilson_95_low"] for rate, row in zip(rates, summary, strict=True)]
    upper_errors = [row["wilson_95_high"] - rate for rate, row in zip(rates, summary, strict=True)]
    rate_bars = axes[1].bar(
        types,
        rates,
        yerr=[lower_errors, upper_errors],
        capsize=5,
        color=ACCENT_COLOR,
        width=0.62,
    )
    for bar, row in zip(rate_bars, summary, strict=True):
        axes[1].annotate(
            f"{row['positive']}/{row['rows']}\n{row['positive_rate']:.2%}",
            (bar.get_x() + bar.get_width() / 2.0, row["wilson_95_high"]),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            color=TEXT_COLOR,
            fontsize=9,
        )
    upper_limit = max(row["wilson_95_high"] for row in summary) * 1.35
    axes[1].set_title("Positive rate by Type (95% Wilson CI)")
    axes[1].set_xlabel("Type")
    axes[1].set_ylabel("Positive proportion")
    axes[1].set_ylim(0, upper_limit)
    axes[1].yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
    _style_axis(axes[1], grid_axis="y")
    figure.suptitle("Composition by Type — training only", fontsize=13, weight="bold")
    _save_figure(figure, path)


def _plot_numeric_ecdf(frame: pd.DataFrame, path: Path) -> None:
    figure = _new_figure(13.0, 8.0)
    axes = figure.subplots(2, 3).ravel()
    for axis, column in zip(axes, NUMERIC_FEATURES, strict=False):
        for target_value, label, color in (
            (0, "No failure", NEGATIVE_COLOR),
            (1, "Failure", POSITIVE_COLOR),
        ):
            values = sorted(frame.loc[frame[TARGET_COLUMN] == target_value, column].to_list())
            cumulative = [(index + 1) / len(values) for index in range(len(values))]
            axis.step(
                values, cumulative, where="post", label=f"{label} (n={len(values)})", color=color
            )
        axis.set_title(column)
        axis.set_xlabel(column)
        axis.set_ylabel("ECDF")
        axis.set_ylim(0, 1.02)
        axis.legend(frameon=False, fontsize=8)
        _style_axis(axis)
    axes[-1].axis("off")
    figure.suptitle(
        "Normalized numeric distributions by class — training only",
        fontsize=13,
        weight="bold",
    )
    _save_figure(figure, path)


def _plot_correlation(frame: pd.DataFrame, path: Path) -> None:
    correlations = frame.loc[:, NUMERIC_FEATURES].corr(method="spearman")
    figure = _new_figure(10.5, 7.2)
    axis = figure.subplots()
    image = axis.imshow(correlations.to_numpy(), cmap="RdBu_r", vmin=-1.0, vmax=1.0)
    axis.set_xticks(range(len(NUMERIC_FEATURES)), labels=NUMERIC_FEATURES, rotation=35, ha="right")
    axis.set_yticks(range(len(NUMERIC_FEATURES)), labels=NUMERIC_FEATURES)
    for row_index, row in enumerate(NUMERIC_FEATURES):
        for column_index, column in enumerate(NUMERIC_FEATURES):
            value = float(correlations.loc[row, column])
            axis.text(
                column_index,
                row_index,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if abs(value) > 0.55 else TEXT_COLOR,
                fontsize=9,
            )
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="Spearman ρ")
    axis.set_title(
        "Correlation among numeric features — training only",
        fontsize=13,
        weight="bold",
        color=TEXT_COLOR,
    )
    _save_figure(figure, path)


def _plot_joint_relationships(frame: pd.DataFrame, path: Path) -> None:
    figure = _new_figure(12.0, 5.2)
    axes = figure.subplots(1, 2)
    relationships = (
        ("Air temperature [K]", "Process temperature [K]", "Temperatures"),
        ("Rotational speed [rpm]", "Torque [Nm]", "Speed and torque"),
    )
    negative = frame[TARGET_COLUMN] == 0
    positive = frame[TARGET_COLUMN] == 1
    for axis, (x_column, y_column, title) in zip(axes, relationships, strict=True):
        density = axis.hexbin(
            frame.loc[negative, x_column],
            frame.loc[negative, y_column],
            gridsize=34,
            mincnt=1,
            cmap="Blues",
            linewidths=0,
            alpha=0.8,
        )
        axis.scatter(
            frame.loc[positive, x_column],
            frame.loc[positive, y_column],
            s=18,
            color=POSITIVE_COLOR,
            alpha=0.75,
            edgecolors="white",
            linewidths=0.25,
            label=f"Failure (n={int(positive.sum())})",
        )
        figure.colorbar(
            density,
            ax=axis,
            fraction=0.046,
            pad=0.04,
            label="No-failure count per hexagon",
        )
        axis.set_title(title)
        axis.set_xlabel(x_column)
        axis.set_ylabel(y_column)
        axis.legend(frameon=False, fontsize=8)
        _style_axis(axis)
    figure.suptitle(
        "Descriptive joint relationships — training only",
        fontsize=13,
        weight="bold",
    )
    _save_figure(figure, path)


def _plot_quintile_rates(frame: pd.DataFrame, path: Path) -> None:
    summaries = _quintile_summary(frame)
    figure = _new_figure(13.0, 8.2)
    axes = figure.subplots(2, 3, sharey=True).ravel()
    highest_interval = max(row["wilson_95_high"] for rows in summaries.values() for row in rows)
    shared_ceiling = max(0.10, math.ceil(highest_interval * 1.08 * 50.0) / 50.0)
    for axis, column in zip(axes, NUMERIC_FEATURES, strict=False):
        rows = summaries[column]
        positions = [row["quintile"] for row in rows]
        rates = [row["positive_rate"] for row in rows]
        lower_errors = [rate - row["wilson_95_low"] for rate, row in zip(rates, rows, strict=True)]
        upper_errors = [row["wilson_95_high"] - rate for rate, row in zip(rates, rows, strict=True)]
        axis.errorbar(
            positions,
            rates,
            yerr=[lower_errors, upper_errors],
            color=ACCENT_COLOR,
            marker="o",
            linewidth=1.8,
            capsize=4,
        )
        axis.set_xticks(positions, labels=[f"Q{position}" for position in positions])
        axis.set_title(column)
        axis.set_xlabel("Training quintile")
        axis.set_ylabel("Positive rate")
        axis.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
        axis.set_ylim(0.0, shared_ceiling)
        _style_axis(axis)
    axes[-1].axis("off")
    figure.suptitle(
        "Positive rate by quintile (95% Wilson CI) — training only",
        fontsize=13,
        weight="bold",
    )
    _save_figure(figure, path)


def _strongest_correlation(correlations: dict[str, dict[str, float]]) -> tuple[str, str, float]:
    strongest = (NUMERIC_FEATURES[0], NUMERIC_FEATURES[1], 0.0)
    for row_index, row in enumerate(NUMERIC_FEATURES):
        for column in NUMERIC_FEATURES[row_index + 1 :]:
            value = correlations[row][column]
            if abs(value) > abs(strongest[2]):
                strongest = (row, column, value)
    return strongest


def _build_report(summary: dict[str, Any]) -> str:
    target = summary["target"]
    type_rows = summary["type_summary"]
    strongest_left, strongest_right, strongest_value = _strongest_correlation(
        summary["spearman_correlations"]
    )
    highest_type = max(type_rows, key=lambda row: row["positive_rate"])
    correlations = summary["spearman_correlations"]
    temperature_correlation = correlations["Air temperature [K]"]["Process temperature [K]"]
    numeric_summary = summary["numeric_summary"]
    failure_air_median = numeric_summary["Air temperature [K]"]["by_target"]["failure"]["median"]
    no_failure_air_median = numeric_summary["Air temperature [K]"]["by_target"]["no_failure"][
        "median"
    ]
    failure_speed_median = numeric_summary["Rotational speed [rpm]"]["by_target"]["failure"][
        "median"
    ]
    no_failure_speed_median = numeric_summary["Rotational speed [rpm]"]["by_target"]["no_failure"][
        "median"
    ]
    failure_torque_median = numeric_summary["Torque [Nm]"]["by_target"]["failure"]["median"]
    no_failure_torque_median = numeric_summary["Torque [Nm]"]["by_target"]["no_failure"]["median"]
    failure_wear_median = numeric_summary["Tool wear [min]"]["by_target"]["failure"]["median"]
    no_failure_wear_median = numeric_summary["Tool wear [min]"]["by_target"]["no_failure"]["median"]
    numeric_rows = []
    for column, values in summary["numeric_summary"].items():
        overall = values["overall"]
        numeric_rows.append(
            f"| `{column}` | {overall['min']:.2f} | {overall['q1']:.2f} | "
            f"{overall['median']:.2f} | {overall['q3']:.2f} | {overall['max']:.2f} |"
        )
    type_table = []
    for row in type_rows:
        type_table.append(
            f"| {row['type']} | {row['rows']:,} | {row['positive']:,} | "
            f"{row['positive_rate']:.2%} | "
            f"{row['wilson_95_low']:.2%}–{row['wilson_95_high']:.2%} |"
        )

    figure_lines = [
        f"![{filename.removesuffix('.png')}](figures/{filename})" for filename in FIGURE_FILENAMES
    ]
    return f"""# Training EDA and frozen protocol

> Scope: this report uses exclusively the **training** partition after separating the holdout.
> AI4I 2020 is synthetic; the observed associations do not validate industrial use.

## Protocol frozen before modeling

- Source fixed by SHA-256: `{summary["source"]["sha256"]}`.
- Split: {1.0 - HOLDOUT_FRACTION:.0%} training / {HOLDOUT_FRACTION:.0%} holdout, stratified,
  with the predefined arbitrary random seed `{RANDOM_SEED}` and original snapshot order preserved
  within each partition. The seed was neither optimized nor compared with alternatives.
- Training rows analyzed: {summary["training_rows"]:,}. The holdout is neither loaded nor profiled.
- M3 cross-validation: `StratifiedKFold(n_splits={CV_FOLDS}, shuffle=True,
  random_state={RANDOM_SEED})`, shared by every candidate.
- Baseline: `DummyClassifier(strategy="prior")`.
- Selection: highest mean AP across the {CV_FOLDS} folds (`{PRIMARY_METRIC}`); ROC-AUC
  (`{SECONDARY_METRIC}`) is secondary. Pooled OOF AP is diagnostic only and does not replace the
  CV mean. A tie within `{TIE_TOLERANCE:.0e}` between logistic regression and random forest favors
  logistic regression for simplicity. If no candidate beats the dummy in mean AP, the holdout
  remains sealed.
- Threshold: `{THRESHOLD_STRATEGY}` on `predict_proba[:, 1]`. F1 is maximized using one OOF
  prediction per training row and the `score >= threshold` rule; ties within
  `{TIE_TOLERANCE:.0e}` are resolved by the smallest absolute difference between precision and
  recall, then by the lower threshold. The value is frozen before the one-time holdout evaluation.
- All preprocessing is fitted within each fold through a `Pipeline`.

M1 necessarily checked global file counts and ranges to validate the data contract. Since this
partition was created in M2, no holdout-specific statistics, examples, or results have been used
to make decisions.

## Training data quality and class balance

- Rows: {summary["training_rows"]:,}.
- Positive: {target["positive"]:,}; negative: {target["negative"]:,}.
- Positive prevalence: **{target["prevalence"]:.2%}**.
- Missing cells: {summary["quality"]["missing_cells"]}.
- Rows duplicated across the six features: {summary["quality"]["duplicate_feature_rows"]}.
- Columns analyzed: only the six allowed features and `{TARGET_COLUMN}`.

## Type

| Type | Rows | Positive | Positive rate | 95% Wilson CI |
|---|---:|---:|---:|---:|
{chr(10).join(type_table)}

The highest descriptive rate occurs for `Type={highest_type["type"]}`
({highest_type["positive_rate"]:.2%}); intervals and support must accompany any interpretation.
This is not interpreted as a causal effect.

## Training numeric profile

| Feature | Min | Q1 | Median | Q3 | Max |
|---|---:|---:|---:|---:|---:|
{chr(10).join(numeric_rows)}

The strongest monotonic association among numeric features is between `{strongest_left}` and
`{strongest_right}` (Spearman ρ = {strongest_value:.3f}). This may matter for coefficient
stability, but it does not justify removing variables before comparing the preregistered
pipelines. Air and process temperature are also strongly associated
(ρ = {temperature_correlation:.3f}).

In the training data, failure observations have higher median torque
({failure_torque_median:.1f} versus {no_failure_torque_median:.1f} Nm), tool wear
({failure_wear_median:.0f} versus {no_failure_wear_median:.0f} min), and air temperature
({failure_air_median:.1f} versus {no_failure_air_median:.1f} K), as well as lower rotational speed
({failure_speed_median:.0f} versus {no_failure_speed_median:.0f} rpm). The ECDFs show substantial
overlap, and the joint and quintile panels do not suggest a single uniform linear relationship.
These descriptive observations support comparing the preregistered logistic regression and
random forest candidates; they establish neither causality nor predictive performance.

## Visualizations

{chr(10).join(figure_lines)}

The quintiles are only a visual aid calculated on training data; they are not included as a
transformation and do not create new features. The ECDFs are normalized within each class, so they
must be read alongside the prevalence chart.

## Risks and limitations

- The random split estimates IID generalization within the same synthetic generator; it does not
  measure temporal, cross-machine, or real-world industrial generalization.
- Associations with the target are descriptive, not causal. The EDA does not authorize changing
  the target, features, or candidates without a new explicit decision.
- Positive cases are scarce; subgroup and quintile rates have visible uncertainty.
- The absence of nulls or duplicates in AI4I does not imply that real data would have this quality.
- The `TWF`, `HDF`, `PWF`, `OSF`, and `RNF` indicators, along with identifiers, remain excluded
  from derived data and all figures to prevent leakage.
- The 27 global discrepancies between failure modes and the target detected in M1 are not
  corrected; the contractual target remains `Machine failure`.

## Review gate

This report must be reviewed before M3 runs. There are no trained models, validation metrics, or
holdout results yet.
"""


def generate_eda(
    processed_dir: Path = DEFAULT_PROCESSED_DATA_DIR,
    split_manifest_path: Path = DEFAULT_SPLIT_MANIFEST_PATH,
    report_dir: Path = DEFAULT_EDA_REPORT_DIR,
) -> EDAArtifacts:
    """Generate six figures and summaries from training without resolving holdout."""
    frame, split_manifest = load_training_partition(processed_dir, split_manifest_path)
    target_counts = frame[TARGET_COLUMN].value_counts().reindex([0, 1], fill_value=0)
    positive_rows = int(target_counts.loc[1])
    training_rows = len(frame)
    figure_dir = report_dir / "figures"
    figure_paths = tuple(figure_dir / filename for filename in FIGURE_FILENAMES)

    plotters = (
        _plot_target_prevalence,
        _plot_type_support_and_rate,
        _plot_numeric_ecdf,
        _plot_correlation,
        _plot_joint_relationships,
        _plot_quintile_rates,
    )
    for plotter, figure_path in zip(plotters, figure_paths, strict=True):
        plotter(frame, figure_path)

    summary: dict[str, Any] = {
        "schema_version": 2,
        "scope": "training_only",
        "source": split_manifest["source"],
        "split": split_manifest["split"],
        "training_rows": training_rows,
        "feature_columns": list(FEATURE_COLUMNS),
        "target_column": TARGET_COLUMN,
        "target": {
            "negative": int(target_counts.loc[0]),
            "positive": positive_rows,
            "prevalence": positive_rows / training_rows,
        },
        "quality": {
            "missing_cells": int(frame.isna().sum().sum()),
            "duplicate_feature_rows": int(frame.duplicated(subset=FEATURE_COLUMNS).sum()),
        },
        "type_summary": _type_summary(frame),
        "numeric_summary": _numeric_summary(frame),
        "spearman_correlations": _correlation_summary(frame),
        "positive_rate_by_quintile": _quintile_summary(frame),
        "figures": [f"figures/{filename}" for filename in FIGURE_FILENAMES],
        "protocol": {
            "random_seed": RANDOM_SEED,
            "cv_folds": CV_FOLDS,
            "primary_metric": PRIMARY_METRIC,
            "secondary_metric": SECONDARY_METRIC,
            "threshold_strategy": THRESHOLD_STRATEGY,
            "tie_tolerance": TIE_TOLERANCE,
            "threshold_score": "predict_proba[:, 1]",
            "threshold_decision_rule": "score >= threshold",
            "pooled_oof_average_precision_role": "diagnostic_only",
            "holdout_profiled": False,
        },
    }
    summary_path = report_dir / "summary.json"
    report_path = report_dir / "EDA_REPORT.md"
    _write_text_if_changed(
        summary_path,
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
    )
    _write_text_if_changed(report_path, _build_report(summary))
    return EDAArtifacts(
        report=report_path,
        summary=summary_path,
        figures=figure_paths,
        training_rows=training_rows,
        positive_rows=positive_rows,
    )
