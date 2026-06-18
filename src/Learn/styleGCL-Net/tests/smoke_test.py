from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stylegcl.cli import main


def test_text_and_label_smoke(tmp_path: Path) -> None:
    text_csv = tmp_path / "text.csv"
    text_out = tmp_path / "text_out"
    label_csv = tmp_path / "label.csv"
    label_out = tmp_path / "label_out"

    main(
        [
            "generate-text",
            "--output",
            str(text_csv),
            "--dataset",
            "trec",
            "--num-tasks",
            "40",
            "--num-workers",
            "24",
            "--answers-per-task",
            "6",
        ]
    )
    main(
        [
            "run-text",
            "--input",
            str(text_csv),
            "--output-dir",
            str(text_out),
            "--n-clusters",
            "3",
        ]
    )
    assert (text_out / "worker_predictions.csv").exists()
    assert (text_out / "metrics.json").exists()
    assert (text_out / "task_contributions.csv").exists()

    main(
        [
            "generate-label",
            "--output",
            str(label_csv),
            "--num-tasks",
            "60",
            "--num-workers",
            "24",
            "--answers-per-task",
            "6",
        ]
    )
    main(
        [
            "run-label",
            "--input",
            str(label_csv),
            "--output-dir",
            str(label_out),
            "--n-clusters",
            "3",
        ]
    )
    assert (label_out / "worker_predictions.csv").exists()
    assert (label_out / "metrics.json").exists()


if __name__ == "__main__":
    test_text_and_label_smoke(Path("outputs/smoke_tmp"))
    print("smoke test passed")
