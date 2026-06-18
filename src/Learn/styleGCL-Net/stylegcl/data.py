from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .utils import read_csv_rows


TEXT_ALIASES = {
    "task_id": ("task_id", "task", "qid", "question_id", "item_id", "image_id"),
    "worker_id": ("worker_id", "worker", "annotator_id", "user_id", "uid"),
    "task_text": ("task_text", "task_desc", "question", "prompt", "description", "review"),
    "answer_text": ("answer_text", "answer", "response", "rationale", "reason", "text"),
    "label": ("label", "answer_label", "class", "category"),
    "is_colluder": ("is_colluder", "colluder", "malicious", "is_malicious", "target"),
    "gang_id": ("gang_id", "group_id", "collusion_group", "source_group"),
    "source_model": ("source_model", "llm", "model", "generator"),
    "attack_scenario": ("attack_scenario", "attack", "scenario"),
}

LABEL_ALIASES = {
    "task_id": TEXT_ALIASES["task_id"],
    "worker_id": TEXT_ALIASES["worker_id"],
    "label": TEXT_ALIASES["label"] + ("worker_label", "value"),
    "true_label": ("true_label", "gold", "gold_label", "truth"),
    "is_colluder": TEXT_ALIASES["is_colluder"],
    "gang_id": TEXT_ALIASES["gang_id"],
}


@dataclass(frozen=True)
class CrowdTextRecord:
    task_id: str
    worker_id: str
    task_text: str
    answer_text: str
    label: str | None = None
    is_colluder: int | None = None
    gang_id: str | None = None
    source_model: str | None = None
    attack_scenario: str | None = None


@dataclass(frozen=True)
class LabelRecord:
    task_id: str
    worker_id: str
    label: str
    true_label: str | None = None
    is_colluder: int | None = None
    gang_id: str | None = None


def _pick(row: dict[str, str], aliases: Iterable[str], default: str = "") -> str:
    lowered = {k.lower().strip(): v for k, v in row.items()}
    for key in aliases:
        if key.lower() in lowered:
            value = lowered[key.lower()]
            return "" if value is None else str(value).strip()
    return default


def _parse_bool_int(value: str) -> int | None:
    value = str(value).strip().lower()
    if value == "":
        return None
    if value in {"1", "true", "yes", "y", "colluder", "malicious"}:
        return 1
    if value in {"0", "false", "no", "n", "normal", "human"}:
        return 0
    try:
        return int(float(value))
    except ValueError:
        return None


def load_text_records(path: str | Path) -> list[CrowdTextRecord]:
    rows = read_csv_rows(path)
    records: list[CrowdTextRecord] = []
    for idx, row in enumerate(rows, start=2):
        task_id = _pick(row, TEXT_ALIASES["task_id"])
        worker_id = _pick(row, TEXT_ALIASES["worker_id"])
        answer_text = _pick(row, TEXT_ALIASES["answer_text"])
        if not task_id or not worker_id or not answer_text:
            raise ValueError(
                f"{path!s}:{idx} must contain task_id, worker_id, and answer_text"
            )
        task_text = _pick(row, TEXT_ALIASES["task_text"], default="")
        records.append(
            CrowdTextRecord(
                task_id=task_id,
                worker_id=worker_id,
                task_text=task_text,
                answer_text=answer_text,
                label=_pick(row, TEXT_ALIASES["label"], default="") or None,
                is_colluder=_parse_bool_int(_pick(row, TEXT_ALIASES["is_colluder"])),
                gang_id=_pick(row, TEXT_ALIASES["gang_id"], default="") or None,
                source_model=_pick(row, TEXT_ALIASES["source_model"], default="") or None,
                attack_scenario=_pick(row, TEXT_ALIASES["attack_scenario"], default="")
                or None,
            )
        )
    return records


def load_label_records(path: str | Path) -> list[LabelRecord]:
    rows = read_csv_rows(path)
    records: list[LabelRecord] = []
    for idx, row in enumerate(rows, start=2):
        task_id = _pick(row, LABEL_ALIASES["task_id"])
        worker_id = _pick(row, LABEL_ALIASES["worker_id"])
        label = _pick(row, LABEL_ALIASES["label"])
        if not task_id or not worker_id or label == "":
            raise ValueError(f"{path!s}:{idx} must contain task_id, worker_id, and label")
        records.append(
            LabelRecord(
                task_id=task_id,
                worker_id=worker_id,
                label=label,
                true_label=_pick(row, LABEL_ALIASES["true_label"], default="") or None,
                is_colluder=_parse_bool_int(_pick(row, LABEL_ALIASES["is_colluder"])),
                gang_id=_pick(row, LABEL_ALIASES["gang_id"], default="") or None,
            )
        )
    return records


def infer_worker_truth(records: Iterable[CrowdTextRecord | LabelRecord]) -> dict[str, int]:
    labels: dict[str, list[int]] = {}
    for rec in records:
        value = rec.is_colluder
        if value is not None:
            labels.setdefault(rec.worker_id, []).append(int(value))
    return {
        worker: int(sum(values) >= max(1, len(values) / 2))
        for worker, values in labels.items()
    }
