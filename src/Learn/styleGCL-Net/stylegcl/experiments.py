from __future__ import annotations

from pathlib import Path

import numpy as np

from .baselines import nkcd_baseline, stylometric_baseline, text_cosine_baseline
from .data import infer_worker_truth, load_text_records
from .metrics import binary_classification_metrics
from .model_numpy import StyleGCLConfig, StyleGCLDetector
from .synthetic import ATTACK_SCENARIOS, generate_text_dataset
from .utils import ensure_dir, save_json


def run_attack_sweep(
    output_dir: str,
    dataset: str = "trec",
    num_tasks: int = 160,
    num_workers: int = 80,
    answers_per_task: int = 6,
    colluder_ratio: float = 0.3,
    seed: int = 13,
) -> dict[str, object]:
    out = ensure_dir(output_dir)
    summary: dict[str, object] = {
        "dataset": dataset,
        "colluder_ratio": colluder_ratio,
        "scenarios": {},
    }
    for scenario in sorted(ATTACK_SCENARIOS):
        csv_path = out / f"{scenario}_synthetic.csv"
        generate_text_dataset(
            output=str(csv_path),
            dataset=dataset,
            num_tasks=num_tasks,
            num_workers=num_workers,
            answers_per_task=answers_per_task,
            colluder_ratio=colluder_ratio,
            attack_scenario=scenario,
            seed=seed,
        )
        records = load_text_records(csv_path)
        truth = infer_worker_truth(records)
        y = np.asarray([truth[w] for w in sorted(truth)], dtype=int)
        methods = [
            _stylegcl_result(records, seed),
            text_cosine_baseline(records),
            stylometric_baseline(records),
            nkcd_baseline(records),
        ]
        scenario_metrics: dict[str, object] = {}
        for method in methods:
            aligned_y = np.asarray([truth[w] for w in method.worker_ids], dtype=int)
            scenario_metrics[method.name] = binary_classification_metrics(aligned_y, method.scores)
        summary["scenarios"][scenario] = scenario_metrics
    save_json(out / "attack_sweep_metrics.json", summary)
    return summary


def run_ablation(
    input_csv: str,
    output_dir: str,
    seed: int = 13,
) -> dict[str, object]:
    records = load_text_records(input_csv)
    truth = infer_worker_truth(records)
    variants = {
        "full": StyleGCLConfig(random_seed=seed),
        "w_o_graph": StyleGCLConfig(random_seed=seed, gat_layers=0),
        "weaker_graph": StyleGCLConfig(random_seed=seed, graph_weight=0.25),
        "high_mask": StyleGCLConfig(random_seed=seed, mask_prob=0.35),
    }
    report: dict[str, object] = {}
    for name, config in variants.items():
        result = StyleGCLDetector(config).fit_predict(records)
        if truth:
            y = np.asarray([truth[w] for w in result.worker_ids], dtype=int)
            report[name] = binary_classification_metrics(y, result.scores)
        else:
            report[name] = {"losses": result.losses}
    out = ensure_dir(output_dir)
    save_json(Path(out) / "ablation_metrics.json", report)
    return report


def _stylegcl_result(records, seed):
    result = StyleGCLDetector(StyleGCLConfig(random_seed=seed)).fit_predict(records)

    class _Result:
        worker_ids = result.worker_ids
        scores = result.scores
        embeddings = result.embeddings
        name = "stylegcl"

    return _Result()
