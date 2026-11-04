"""Command-line interface: ``dsd <command>``."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from directed_speech_detector.config import ExperimentConfig, load_config
from directed_speech_detector.utils import get_logger

LOGGER = get_logger("dsd")
ABLATION_CONFIGS = ["audio_only", "text_only", "fused", "fused_speaker"]


def _add_config_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", required=True, help="Path to a YAML experiment config")
    p.add_argument(
        "--set", nargs="*", default=[], metavar="KEY=VALUE", help="Config overrides, e.g. train.epochs=3"
    )


def _train_and_eval(cfg: ExperimentConfig) -> Path:
    from directed_speech_detector.evaluation.analysis import evaluate_run
    from directed_speech_detector.training.trainer import train_model

    result = train_model(cfg)
    evaluate_run(result.run_dir)
    return result.run_dir


def cmd_generate_data(args: argparse.Namespace) -> None:
    from directed_speech_detector.data.dataset import get_split

    cfg = load_config(args.config, args.set)
    for split in ("train", "val", "test"):
        LOGGER.info("%s: %d utterances", split, len(get_split(cfg, split)))


def cmd_train(args: argparse.Namespace) -> None:
    cfg = load_config(args.config, args.set)
    LOGGER.info("run dir: %s", _train_and_eval(cfg))


def cmd_evaluate(args: argparse.Namespace) -> None:
    from directed_speech_detector.evaluation.analysis import evaluate_run

    res = evaluate_run(args.run_dir)
    far = next(iter(res["at_target_far"].values()))
    print(json.dumps({"auc": res["auc"], "eer": res["eer"], "at_target_far": far}, indent=2))


def cmd_ablate(args: argparse.Namespace) -> None:
    for name in args.configs:
        for seed in args.seeds:
            cfg = load_config(
                Path(args.config_dir) / f"{name}.yaml",
                [*args.set, f"train.seed={seed}", f"name={name}_s{seed}"],
            )
            LOGGER.info("ablation %s seed=%d -> %s", name, seed, _train_and_eval(cfg))


def cmd_report(args: argparse.Namespace) -> None:
    from directed_speech_detector.evaluation.report import build_report

    summary = build_report(Path(args.runs_dir), Path(args.docs_dir))
    LOGGER.info("wrote report for %s", list(summary))


def cmd_export(args: argparse.Namespace) -> None:
    from directed_speech_detector.export import export_all

    print(json.dumps(export_all(Path(args.run_dir)), indent=2))


def cmd_benchmark(args: argparse.Namespace) -> None:
    from directed_speech_detector.benchmark import run_benchmark

    run_benchmark(Path(args.run_dir), Path(args.docs_dir), n_latency=args.iterations)


def cmd_serve(args: argparse.Namespace) -> None:
    import uvicorn

    from directed_speech_detector.serving.app import create_app
    from directed_speech_detector.serving.inference import Detector

    detector = Detector.from_run_dir(args.run_dir, backend=args.backend)
    uvicorn.run(create_app(detector), host=args.host, port=args.port, log_level="info")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dsd", description="Directed speech detector toolkit")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("generate-data", help="Generate and cache the synthetic proxy dataset")
    _add_config_args(p)
    p.set_defaults(func=cmd_generate_data)

    p = sub.add_parser("train", help="Train one configuration and evaluate it")
    _add_config_args(p)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("evaluate", help="Evaluate a trained run directory")
    p.add_argument("--run-dir", required=True)
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("ablate", help="Train + evaluate the four ablations over seeds")
    p.add_argument("--config-dir", default="configs")
    p.add_argument("--configs", nargs="*", default=ABLATION_CONFIGS)
    p.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    p.set_defaults(func=cmd_ablate)

    p = sub.add_parser("report", help="Aggregate evaluated runs into docs tables and figures")
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--docs-dir", default="docs")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("export", help="INT8-quantize and export to ONNX with a parity check")
    p.add_argument("--run-dir", required=True)
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("benchmark", help="Benchmark FP32 vs INT8 vs ONNX (latency, size, accuracy)")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--docs-dir", default="docs")
    p.add_argument("--iterations", type=int, default=300)
    p.set_defaults(func=cmd_benchmark)

    p = sub.add_parser("serve", help="Serve the FastAPI app")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--backend", choices=["fp32", "int8", "onnx"], default="fp32")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(func=cmd_serve)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
