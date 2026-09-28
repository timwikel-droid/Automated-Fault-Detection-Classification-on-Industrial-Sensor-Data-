
"""
main.py

Runs the full pipeline end to end and writes everything a reviewer would
want to see into results/:

  results/plots/detection_example.png     - one experiment's sensors +
                                             true vs. detected fault region
  results/plots/learning_curves.png       - the core comparison: accuracy
                                             vs. labeling effort for
                                             Active Learning (uncertainty),
                                             a random-sampling baseline,
                                             and Template Matching
  results/summary.md                      - all numbers in one place

Run with:  python3 main.py   (from the project/ directory)
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.data_loader import load_all_experiments
from src.features import make_windows
from src.anomaly_detection import IsolationForestDetector, MahalanobisDetector, PerRunBaselineDetector
from src.evaluate import point_adjusted_scores
from src.classification_data import build_fault_window_pool, split_by_experiment, FAULT_CLASSES
from src.active_learning import run_active_learning_curve
from src.template_matching import run_template_matching_curve

RESULTS_DIR = Path(__file__).resolve().parent / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
SEED = 0


def stage1_detection(experiments):
    print("\n=== Stage 1: Fault Detection (unsupervised) ===")

    normal_exp = next(e for e in experiments if e.fault_type == "normal")
    normal_windows = make_windows(normal_exp.df)
    print(f"Trained on {len(normal_windows)} normal-operation windows "
          f"({normal_exp.experiment_id}).")

    iforest = IsolationForestDetector(contamination=0.02).fit(normal_windows)
    maha = MahalanobisDetector(n_sigma=3.0).fit(normal_windows)
    per_run = PerRunBaselineDetector(n_sigma=4.0)

    fault_exps = [e for e in experiments if e.fault_type != "normal"]

    agg = {
        "IsolationForest (global baseline)": {"y_true": [], "y_pred": []},
        "Mahalanobis (global baseline)": {"y_true": [], "y_pred": []},
        "Per-run adaptive baseline": {"y_true": [], "y_pred": []},
    }
    for exp in fault_exps:
        w = make_windows(exp.df)
        agg["IsolationForest (global baseline)"]["y_true"].append(w["label"].to_numpy())
        agg["IsolationForest (global baseline)"]["y_pred"].append(iforest.predict(w))
        agg["Mahalanobis (global baseline)"]["y_true"].append(w["label"].to_numpy())
        agg["Mahalanobis (global baseline)"]["y_pred"].append(maha.predict(w))
        agg["Per-run adaptive baseline"]["y_true"].append(w["label"].to_numpy())
        agg["Per-run adaptive baseline"]["y_pred"].append(per_run.predict_for_experiment(w))

    results = {}
    for name, d in agg.items():
        y_true = np.concatenate(d["y_true"])
        y_pred = np.concatenate(d["y_pred"])
        scores = point_adjusted_scores(y_true, y_pred)
        results[name] = scores
        print(f"  {name:32s} precision={scores['precision']:.3f}  "
              f"recall={scores['recall']:.3f}  f1={scores['f1']:.3f}  "
              f"(segments detected {scores['segments_detected']}/{scores['segments_total']})")

    # Also report each global detector's flagged-window rate: this is what
    # actually reveals the domain-shift problem (they flag nearly every
    # window in every other run, not just the faulty ones).
    all_true = np.concatenate(agg["IsolationForest (global baseline)"]["y_true"])
    if_pred = np.concatenate(agg["IsolationForest (global baseline)"]["y_pred"])
    print(f"  -> IsolationForest flags {if_pred.mean():.1%} of ALL windows "
          f"across other runs (true fault rate is {all_true.mean():.1%}) "
          f"-- see README for why.")

    example = next(e for e in fault_exps if e.experiment_id == "valve1/0")
    _plot_detection_example(example, iforest, per_run)

    return results


def _plot_detection_example(exp, global_detector, per_run_detector):
    windows = make_windows(exp.df)
    pred_global = global_detector.predict(windows)
    pred_local = per_run_detector.predict_for_experiment(windows)
    win_times = exp.df.index[windows["t_start"].to_numpy()]

    fig, axes = plt.subplots(
        4, 1, figsize=(11, 8), sharex=True,
        gridspec_kw={"height_ratios": [3, 3, 1, 1]},
    )
    channels = ["Pressure", "Accelerometer1RMS"]
    time = exp.df.index
    true_anom = exp.df["anomaly"].to_numpy().astype(bool)

    for ax, ch in zip(axes[:2], channels):
        ax.plot(time, exp.df[ch], linewidth=0.7, color="#2b6cb0")
        ax.set_ylabel(ch, fontsize=9)
        ax.fill_between(time, exp.df[ch].min(), exp.df[ch].max(),
                         where=true_anom, color="red", alpha=0.10,
                         label="true fault region" if ch == channels[0] else None)

    def _strip(ax, pred, title):
        detected_times = win_times[pred == 1]
        true_at_detected = true_anom[windows["t_start"].to_numpy()][pred == 1]
        ax.scatter(detected_times[true_at_detected], np.ones(true_at_detected.sum()),
                    marker="|", color="#2f855a", s=150)
        ax.scatter(detected_times[~true_at_detected], np.ones((~true_at_detected).sum()),
                    marker="|", color="#1a202c", s=150)
        ax.set_yticks([0.5], [title], fontsize=8)
        ax.set_ylim(0.5, 1.5)

    _strip(axes[2], pred_global, "IsolationForest\n(global)")
    _strip(axes[3], pred_local, "Per-run\nadaptive")

    handles = [
        plt.Line2D([0], [0], marker="|", color="#2f855a", linestyle="", markersize=12,
                   label="flagged, inside true fault"),
        plt.Line2D([0], [0], marker="|", color="#1a202c", linestyle="", markersize=12,
                   label="flagged, false alarm"),
    ]
    axes[0].legend(loc="upper right", fontsize=8)
    axes[2].legend(handles=handles, loc="upper right", fontsize=7, ncol=2)

    axes[0].set_title(
        f"Detection example: {exp.experiment_id} (fault type: {exp.fault_type})\n"
        "global baseline (trained once, on a different run) vs. per-run adaptive baseline"
    )
    axes[-1].set_xlabel("time")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "detection_example.png", dpi=130)
    plt.close(fig)


def stage2_classification(experiments):
    print("\n=== Stage 2: Fault Classification (Active Learning vs. Template Matching) ===")

    X, y, groups = build_fault_window_pool(experiments, seed=SEED)
    train_mask, test_mask = split_by_experiment(groups, test_fraction=0.3, seed=SEED)
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    print(f"Fault windows: {len(X)} total | train {len(X_train)} | test {len(X_test)}")
    print(f"Classes: {FAULT_CLASSES}")
    print(f"Train experiments: {len(set(groups[train_mask]))}, "
          f"test experiments: {len(set(groups[test_mask]))}")

    n_seed = 6
    n_queries = min(50, len(X_train) - n_seed - 1)

    al_curve = run_active_learning_curve(
        X_train, y_train, X_test, y_test,
        n_seed=n_seed, n_queries=n_queries, strategy="uncertainty", seed=SEED,
    )
    random_curve = run_active_learning_curve(
        X_train, y_train, X_test, y_test,
        n_seed=n_seed, n_queries=n_queries, strategy="random", seed=SEED,
    )
    schedule = [pt["n_labels"] for pt in al_curve]
    tm_curve = run_template_matching_curve(
        X_train, y_train, X_test, y_test,
        n_seed=n_seed, n_labels_schedule=schedule, seed=SEED,
    )

    _plot_learning_curves(al_curve, random_curve, tm_curve)

    def _at_budget(curve, budget):
        pt = min(curve, key=lambda p: abs(p["n_labels"] - budget))
        return pt["accuracy"]

    # Pick the smallest label budget at which every method has produced a
    # real (non-NaN) prediction -- template matching can't classify a class
    # it has no template for yet, which itself is a useful, honest data
    # point about the low-label regime, but shouldn't be the number we
    # headline as "small-budget accuracy."
    small_budget = next(
        (pt["n_labels"] for pt in tm_curve if not np.isnan(pt["accuracy"])),
        schedule[min(3, len(schedule) - 1)],
    )
    summary = {
        "n_train_windows": len(X_train),
        "n_test_windows": len(X_test),
        "label_budget_max": schedule[-1],
        "final_accuracy": {
            "active_learning": al_curve[-1]["accuracy"],
            "random_sampling": random_curve[-1]["accuracy"],
            "template_matching": tm_curve[-1]["accuracy"],
        },
        "small_budget": small_budget,
        "small_budget_accuracy": {
            "active_learning": _at_budget(al_curve, small_budget),
            "random_sampling": _at_budget(random_curve, small_budget),
            "template_matching": _at_budget(tm_curve, small_budget),
        },
    }
    print(f"  At {schedule[-1]} labels   -> AL {summary['final_accuracy']['active_learning']:.3f} | "
          f"random {summary['final_accuracy']['random_sampling']:.3f} | "
          f"template {summary['final_accuracy']['template_matching']:.3f}")
    print(f"  At {small_budget} labels   -> AL {summary['small_budget_accuracy']['active_learning']:.3f} | "
          f"random {summary['small_budget_accuracy']['random_sampling']:.3f} | "
          f"template {summary['small_budget_accuracy']['template_matching']:.3f}")

    return summary


def _plot_learning_curves(al_curve, random_curve, tm_curve):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot([p["n_labels"] for p in al_curve], [p["accuracy"] for p in al_curve],
            marker="o", markersize=3, label="Active Learning (uncertainty sampling)", color="#c53030")
    ax.plot([p["n_labels"] for p in random_curve], [p["accuracy"] for p in random_curve],
            marker="o", markersize=3, label="Random sampling baseline", color="#718096")
    ax.plot([p["n_labels"] for p in tm_curve], [p["accuracy"] for p in tm_curve],
            marker="s", markersize=3, label="Template Matching (nearest centroid)", color="#2b6cb0")
    ax.set_xlabel("Number of expert-labeled examples")
    ax.set_ylabel("Test accuracy (fault type classification)")
    ax.set_title("Classification accuracy vs. labeling effort")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "learning_curves.png", dpi=130)
    plt.close(fig)


def write_summary(detection_results, classification_summary):
    lines = ["# Results Summary\n"]

    lines.append("## Stage 1 — Fault Detection\n")
    lines.append("| Detector | Precision | Recall | F1 | Segments detected |")
    lines.append("|---|---|---|---|---|")
    for name, s in detection_results.items():
        lines.append(
            f"| {name} | {s['precision']:.3f} | {s['recall']:.3f} | {s['f1']:.3f} "
            f"| {s['segments_detected']}/{s['segments_total']} |"
        )
    lines.append("")
    lines.append("![detection example](plots/detection_example.png)\n")

    lines.append("## Stage 2 — Fault Classification (Active Learning vs. Template Matching)\n")
    c = classification_summary
    lines.append(f"- Train/test split: {c['n_train_windows']} / {c['n_test_windows']} windows "
                 f"(split by experiment, not by window, to avoid leakage)")
    lines.append(f"- At **{c['small_budget']} labeled examples**: "
                 f"Active Learning {c['small_budget_accuracy']['active_learning']:.3f}, "
                 f"Random {c['small_budget_accuracy']['random_sampling']:.3f}, "
                 f"Template Matching {c['small_budget_accuracy']['template_matching']:.3f}")
    lines.append(f"- At **{c['label_budget_max']} labeled examples**: "
                 f"Active Learning {c['final_accuracy']['active_learning']:.3f}, "
                 f"Random {c['final_accuracy']['random_sampling']:.3f}, "
                 f"Template Matching {c['final_accuracy']['template_matching']:.3f}")
    lines.append("")
    lines.append("![learning curves](plots/learning_curves.png)\n")

    (RESULTS_DIR / "summary.md").write_text("\n".join(lines))

    with open(RESULTS_DIR / "metrics.json", "w") as f:
        json.dump(
            {"detection": detection_results, "classification": classification_summary},
            f, indent=2, default=float,
        )


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    experiments = load_all_experiments()
    print(f"Loaded {len(experiments)} experiments.")

    detection_results = stage1_detection(experiments)
    classification_summary = stage2_classification(experiments)
    write_summary(detection_results, classification_summary)
    print(f"\nWrote results to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
