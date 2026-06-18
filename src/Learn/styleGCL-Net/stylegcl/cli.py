from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .data import infer_worker_truth, load_label_records, load_text_records
from .baselines import nkcd_baseline, stylometric_baseline, text_cosine_baseline
from .experiments import run_ablation, run_attack_sweep
from .label_correlation import LabelCorrelationConfig, LabelCorrelationDetector
from .metrics import (
    adjusted_rand_index,
    binary_classification_metrics,
    normalized_mutual_info,
)
from .model_numpy import StyleGCLConfig, StyleGCLDetector
from .synthetic import ATTACK_SCENARIOS, DATASET_PRESETS, generate_label_dataset, generate_text_dataset
from .utils import ensure_dir, save_json, write_csv_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stylegcl",
        description="StyleGCL-Net reproduction toolkit for LLM-assisted collusion detection.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    gen_text = sub.add_parser("generate-text", help="Generate synthetic text crowdsourcing data.")
    gen_text.add_argument("--output", default="data/synthetic_text.csv")
    gen_text.add_argument("--dataset", choices=sorted(DATASET_PRESETS), default="trec")
    gen_text.add_argument("--num-tasks", type=int)
    gen_text.add_argument("--num-workers", type=int)
    gen_text.add_argument("--answers-per-task", type=int)
    gen_text.add_argument("--colluder-ratio", type=float, default=0.3)
    gen_text.add_argument("--attack-scenario", choices=sorted(ATTACK_SCENARIOS), default="S1")
    gen_text.add_argument("--seed", type=int, default=13)

    run_text = sub.add_parser("run-text", help="Run the text style-fingerprint path.")
    run_text.add_argument("--input", required=True)
    run_text.add_argument("--output-dir", default="outputs/text_run")
    run_text.add_argument("--latent-dim", type=int, default=64)
    run_text.add_argument("--gat-layers", type=int, default=2)
    run_text.add_argument("--n-clusters", type=int)
    run_text.add_argument("--min-clusters", type=int, default=2)
    run_text.add_argument("--max-clusters", type=int, default=8)
    run_text.add_argument("--seed", type=int, default=13)

    gen_label = sub.add_parser("generate-label", help="Generate synthetic categorical-label data.")
    gen_label.add_argument("--output", default="data/synthetic_label.csv")
    gen_label.add_argument("--num-tasks", type=int, default=300)
    gen_label.add_argument("--num-workers", type=int, default=80)
    gen_label.add_argument("--answers-per-task", type=int, default=8)
    gen_label.add_argument("--num-classes", type=int, default=4)
    gen_label.add_argument("--colluder-ratio", type=float, default=0.3)
    gen_label.add_argument("--seed", type=int, default=13)

    run_label = sub.add_parser("run-label", help="Run the categorical-label correlation path.")
    run_label.add_argument("--input", required=True)
    run_label.add_argument("--output-dir", default="outputs/label_run")
    run_label.add_argument("--min-overlap", type=int, default=3)
    run_label.add_argument("--n-clusters", type=int)
    run_label.add_argument("--min-clusters", type=int, default=2)
    run_label.add_argument("--max-clusters", type=int, default=8)
    run_label.add_argument("--seed", type=int, default=13)

    baselines = sub.add_parser("run-baselines", help="Run lightweight text baselines on one CSV.")
    baselines.add_argument("--input", required=True)
    baselines.add_argument("--output-dir", default="outputs/baselines")

    sweep = sub.add_parser("attack-sweep", help="Generate and evaluate all six attack scenarios.")
    sweep.add_argument("--output-dir", default="outputs/attack_sweep")
    sweep.add_argument("--dataset", choices=sorted(DATASET_PRESETS), default="trec")
    sweep.add_argument("--num-tasks", type=int, default=160)
    sweep.add_argument("--num-workers", type=int, default=80)
    sweep.add_argument("--answers-per-task", type=int, default=6)
    sweep.add_argument("--colluder-ratio", type=float, default=0.3)
    sweep.add_argument("--seed", type=int, default=13)

    ablation = sub.add_parser("ablation", help="Run lightweight StyleGCL ablation variants.")
    ablation.add_argument("--input", required=True)
    ablation.add_argument("--output-dir", default="outputs/ablation")
    ablation.add_argument("--seed", type=int, default=13)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "generate-text":
        rows = generate_text_dataset(
            output=args.output,
            dataset=args.dataset,
            num_tasks=args.num_tasks,
            num_workers=args.num_workers,
            answers_per_task=args.answers_per_task,
            colluder_ratio=args.colluder_ratio,
            attack_scenario=args.attack_scenario,
            seed=args.seed,
        )
        print(f"wrote {len(rows)} text records to {args.output}")
    elif args.command == "run-text":
        _run_text(args)
    elif args.command == "generate-label":
        rows = generate_label_dataset(
            output=args.output,
            num_tasks=args.num_tasks,
            num_workers=args.num_workers,
            answers_per_task=args.answers_per_task,
            num_classes=args.num_classes,
            colluder_ratio=args.colluder_ratio,
            seed=args.seed,
        )
        print(f"wrote {len(rows)} label records to {args.output}")
    elif args.command == "run-label":
        _run_label(args)
    elif args.command == "run-baselines":
        _run_baselines(args)
    elif args.command == "attack-sweep":
        report = run_attack_sweep(
            output_dir=args.output_dir,
            dataset=args.dataset,
            num_tasks=args.num_tasks,
            num_workers=args.num_workers,
            answers_per_task=args.answers_per_task,
            colluder_ratio=args.colluder_ratio,
            seed=args.seed,
        )
        print(f"wrote attack sweep report to {args.output_dir}")
    elif args.command == "ablation":
        run_ablation(args.input, args.output_dir, seed=args.seed)
        print(f"wrote ablation report to {args.output_dir}")
    else:
        parser.error(f"unknown command: {args.command}")


def _run_text(args: argparse.Namespace) -> None:
    records = load_text_records(args.input)
    out_dir = ensure_dir(args.output_dir)
    detector = StyleGCLDetector(
        StyleGCLConfig(
            latent_dim=args.latent_dim,
            gat_layers=args.gat_layers,
            n_clusters=args.n_clusters,
            min_clusters=args.min_clusters,
            max_clusters=args.max_clusters,
            random_seed=args.seed,
        )
    )
    result = detector.fit_predict(records)
    truth = infer_worker_truth(records)
    metrics = _write_common_outputs(
        out_dir=out_dir,
        worker_ids=result.worker_ids,
        clusters=result.clusters,
        scores=result.scores,
        truth=truth,
        extra_info={
            "path": "text_style_fingerprint",
            "losses": result.losses,
            "cluster_info": result.cluster_info,
            "num_records": len(records),
        },
    )
    _write_task_contributions(out_dir / "task_contributions.csv", result.task_contributions)
    np.save(out_dir / "worker_embeddings.npy", result.embeddings)
    print(f"wrote text detection outputs to {out_dir}")
    if metrics.get("f1") is not None:
        print(f"F1={metrics['f1']:.3f} AUC={metrics['auc_roc']:.3f}")


def _run_label(args: argparse.Namespace) -> None:
    records = load_label_records(args.input)
    out_dir = ensure_dir(args.output_dir)
    detector = LabelCorrelationDetector(
        LabelCorrelationConfig(
            min_overlap=args.min_overlap,
            n_clusters=args.n_clusters,
            min_clusters=args.min_clusters,
            max_clusters=args.max_clusters,
            random_seed=args.seed,
        )
    )
    result = detector.fit_predict(records)
    truth = infer_worker_truth(records)
    metrics = _write_common_outputs(
        out_dir=out_dir,
        worker_ids=result.worker_ids,
        clusters=result.clusters,
        scores=result.scores,
        truth=truth,
        extra_info={
            "path": "label_correlation",
            "cluster_info": result.cluster_info,
            "num_records": len(records),
        },
    )
    np.save(out_dir / "worker_embeddings.npy", result.embeddings)
    np.save(out_dir / "worker_correlation.npy", result.correlation)
    print(f"wrote label detection outputs to {out_dir}")
    if metrics.get("f1") is not None:
        print(f"F1={metrics['f1']:.3f} AUC={metrics['auc_roc']:.3f}")


def _run_baselines(args: argparse.Namespace) -> None:
    records = load_text_records(args.input)
    out_dir = ensure_dir(args.output_dir)
    truth = infer_worker_truth(records)
    report = {"path": "text_baselines", "methods": {}}
    for method in [text_cosine_baseline(records), stylometric_baseline(records), nkcd_baseline(records)]:
        method_dir = ensure_dir(out_dir / method.name)
        metrics = _write_common_outputs(
            out_dir=method_dir,
            worker_ids=method.worker_ids,
            clusters=np.zeros(len(method.worker_ids), dtype=int),
            scores=method.scores,
            truth=truth,
            extra_info={"path": "baseline", "method": method.name},
        )
        np.save(method_dir / "worker_embeddings.npy", method.embeddings)
        report["methods"][method.name] = metrics
    save_json(out_dir / "baseline_metrics.json", report)
    print(f"wrote baseline outputs to {out_dir}")


def _write_common_outputs(
    out_dir: Path,
    worker_ids: list[str],
    clusters: np.ndarray,
    scores: np.ndarray,
    truth: dict[str, int],
    extra_info: dict[str, object],
) -> dict[str, float]:
    rows = []
    y_true = []
    has_truth = all(worker in truth for worker in worker_ids)
    for worker, cluster, score in zip(worker_ids, clusters, scores):
        row = {
            "worker_id": worker,
            "cluster": int(cluster),
            "suspicion_score": float(score),
        }
        if worker in truth:
            row["is_colluder"] = int(truth[worker])
            y_true.append(int(truth[worker]))
        rows.append(row)
    write_csv_rows(out_dir / "worker_predictions.csv", rows)

    metrics: dict[str, float] = {}
    if has_truth:
        metrics.update(binary_classification_metrics(np.asarray(y_true), scores))
        true_cluster = [truth[w] for w in worker_ids]
        metrics["nmi_binary"] = normalized_mutual_info(true_cluster, clusters)
        metrics["ari_binary"] = adjusted_rand_index(true_cluster, clusters)
    report = {
        **extra_info,
        "metrics": metrics,
        "num_workers": len(worker_ids),
    }
    save_json(out_dir / "metrics.json", report)
    return metrics


def _write_task_contributions(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["cluster", "task_id", "contribution", "num_answers"])
        return
    write_csv_rows(path, rows, fieldnames=["cluster", "task_id", "contribution", "num_answers"])
