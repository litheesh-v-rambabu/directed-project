"""Aggregate per-run metrics into ablation tables and figures for the docs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from directed_speech_detector.evaluation import plots

ABLATION_ORDER = ["audio_only", "text_only", "fused", "fused_speaker"]


def _mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.array(values, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=0))


def _far5(m: dict[str, Any]) -> dict[str, float]:
    key = next(k for k in m["at_target_far"] if k.endswith("0.05"))
    out: dict[str, float] = m["at_target_far"][key]
    return out


def collect_runs(runs_dir: Path) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for path in sorted(runs_dir.glob("*_s[0-9]*/metrics.json")):
        metrics = json.loads(path.read_text())
        base = path.parent.name.rsplit("_s", 1)[0]
        metrics["run_dir"] = str(path.parent)
        grouped.setdefault(base, []).append(metrics)
    return grouped


def summarize(grouped: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for name in [n for n in ABLATION_ORDER if n in grouped]:
        runs = grouped[name]
        row: dict[str, Any] = {"n_seeds": len(runs), "n_parameters": runs[0]["n_parameters"]}
        for key, getter in {
            "auc": lambda m: m["auc"],
            "eer": lambda m: m["eer"],
            "accuracy_at_far5": lambda m: _far5(m)["accuracy"],
            "f1_at_far5": lambda m: _far5(m)["f1"],
            "precision_at_far5": lambda m: _far5(m)["precision"],
            "recall_at_far5": lambda m: _far5(m)["recall"],
            "far_at_far5": lambda m: _far5(m)["far"],
            "frr_at_far5": lambda m: _far5(m)["frr"],
            "ece_calibrated": lambda m: m["calibration"]["ece_calibrated"],
            "ece_uncalibrated": lambda m: m["calibration"]["ece_uncalibrated"],
        }.items():
            row[key] = _mean_std([getter(m) for m in runs])
        if all("without_enrollment_at_inference" in m for m in runs):
            row["no_enrollment_eer"] = _mean_std([m["without_enrollment_at_inference"]["eer"] for m in runs])
            row["no_enrollment_frr_at_far5"] = _mean_std(
                [m["without_enrollment_at_inference"]["frr"] for m in runs]
            )
        summary[name] = row
    return summary


def _fmt(pair: tuple[float, float], pct: bool = True, digits: int = 1) -> str:
    mean, std = pair
    return f"{mean * 100:.{digits}f} ± {std * 100:.{digits}f}" if pct else f"{mean:.3f} ± {std:.3f}"


def ablation_markdown(summary: dict[str, dict[str, Any]]) -> str:
    n_seeds = max(r["n_seeds"] for r in summary.values())
    lines = [
        f"Test split (synthetic proxy data), mean ± std over {n_seeds} training seeds. "
        "Operating point: threshold calibrated on the validation split for 5% false-accept rate (FAR).",
        "",
        "| Model | Params | Accuracy % | F1 % | Recall % | FRR % @5% FAR | EER % | AUC | ECE % |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, r in summary.items():
        lines.append(
            f"| {plots.DISPLAY.get(name, name)} | {r['n_parameters']:,} | {_fmt(r['accuracy_at_far5'])} | "
            f"{_fmt(r['f1_at_far5'])} | {_fmt(r['recall_at_far5'])} | {_fmt(r['frr_at_far5'])} | {_fmt(r['eer'])} | "
            f"{_fmt(r['auc'], pct=False)} | {_fmt(r['ece_calibrated'])} |"
        )
    if "fused_speaker" in summary and "no_enrollment_eer" in summary["fused_speaker"]:
        r = summary["fused_speaker"]
        lines += [
            "",
            "Audio + text + speaker model evaluated with the enrollment profile withheld at inference: "
            f"EER {_fmt(r['no_enrollment_eer'])} %, FRR @5% FAR {_fmt(r['no_enrollment_frr_at_far5'])} %.",
        ]
    return "\n".join(lines) + "\n"


def conditions_markdown(metrics: dict[str, Any]) -> str:
    lines = [f"Per-condition error analysis for `{metrics['run']}` (test split, 5% FAR threshold).", ""]
    for cond, rows in metrics["per_condition"].items():
        lines += [
            f"**{cond}**",
            "",
            "| Bucket | n | Accuracy % | FAR % | FRR % |",
            "|---|---:|---:|---:|---:|",
        ]
        lines += [
            f"| {r['bucket']} | {int(r['n'])} | {r['accuracy'] * 100:.1f} | {r['far'] * 100:.1f} | {r['frr'] * 100:.1f} |"
            for r in rows
        ]
        lines.append("")
    for kind, rows in metrics["confusion_examples"].items():
        lines += [
            f"**Top {kind.replace('_', ' ')}**",
            "",
            "| Transcript (ASR-style) | P(directed) | SNR dB | Enrolled speaker |",
            "|---|---:|---:|---|",
        ]
        lines += [
            f"| {r['text']} | {r['p_directed']} | {r['snr_db']} | {r['enrolled_speaker']} |" for r in rows
        ]
        lines.append("")
    return "\n".join(lines)


def build_report(runs_dir: Path, docs_dir: Path) -> dict[str, dict[str, Any]]:
    grouped = collect_runs(runs_dir)
    summary = summarize(grouped)
    if not summary:
        raise FileNotFoundError(f"No evaluated runs (*_s<seed>/metrics.json) found in {runs_dir}")
    results_dir, fig_dir = docs_dir / "results", docs_dir / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "ablation.json").write_text(json.dumps(summary, indent=2))
    (results_dir / "ablation.md").write_text(ablation_markdown(summary))

    first = {name: runs[0] for name, runs in grouped.items() if name in summary}
    preds = {}
    for name, m in first.items():
        z = np.load(Path(m["run_dir"]) / "predictions.npz")
        preds[name] = (z["logits"], z["labels"])
    plots.plot_error_tradeoff(
        preds, fig_dir / "error_tradeoff.png", "Error trade-off (test, synthetic proxy)"
    )
    plots.plot_ablation(summary, fig_dir / "ablation.png")
    per_model = {n: m["per_condition"] for n, m in first.items()}
    plots.plot_conditions(per_model, "snr", fig_dir / "accuracy_by_snr.png", "Accuracy by SNR bucket")
    plots.plot_conditions(
        per_model, "duration", fig_dir / "accuracy_by_duration.png", "Accuracy by utterance length"
    )
    best = first.get("fused_speaker") or next(iter(first.values()))
    plots.plot_reliability(
        best["reliability_bins"], fig_dir / "reliability.png", f"Calibration: {best['run']}"
    )
    (results_dir / "error_analysis.md").write_text(conditions_markdown(best))
    return summary
