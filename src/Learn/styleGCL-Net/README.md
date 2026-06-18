# StyleGCL-Net Reproduction

This repository reconstructs the paper project from `paper.docx`: **StyleGCL-Net**, a framework for LLM-assisted worker collusion detection in crowdsourcing.

The implementation follows the paper's two detection paths:

- **Text answer path**: semantic/content-style disentanglement, worker style aggregation graph, contrastive diagnostics, spectral clustering, and task-level interpretability.
- **Categorical label path**: a complementary worker-label correlation detector for pure class-label tasks where text style fingerprints are not meaningful.

The real S4-Dog, TREC, and MULTITuDE data files were not provided, so the project includes synthetic benchmark generators that mimic the paper's dataset sizes, LLM sources, and six attack scenarios. The same CLI can read real CSV files later.

## Project Layout

```text
stylegcl/
  cli.py                  # command line entry
  data.py                 # CSV schema and loaders
  features.py             # frozen hashing sentence encoder fallback
  model_numpy.py          # runnable StyleGCL-Net reproduction without torch
  torch_model.py          # optional trainable PyTorch modules
  clustering.py           # spectral clustering and k-means
  label_correlation.py    # categorical-label complementary path
  synthetic.py            # synthetic benchmark generation
  metrics.py              # F1, AUC, NMI, ARI
configs/
  stylegcl_default.json
tests/
  smoke_test.py
```

## Text CSV Schema

Required columns:

- `task_id`
- `worker_id`
- `answer_text`

Recommended columns:

- `task_text`
- `label`
- `is_colluder`
- `gang_id`
- `source_model`
- `attack_scenario`

Aliases such as `question`, `prompt`, `response`, `rationale`, `user_id`, and `target` are accepted by the loader.

## Label CSV Schema

Required columns:

- `task_id`
- `worker_id`
- `label`

Recommended columns:

- `true_label`
- `is_colluder`
- `gang_id`

## Quick Start

Use the bundled Python in this Codex environment:

```powershell
& 'C:\Users\32082\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m stylegcl generate-text --output data/synthetic_text.csv --dataset trec --num-tasks 120 --num-workers 60 --answers-per-task 6 --attack-scenario S1
& 'C:\Users\32082\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m stylegcl run-text --input data/synthetic_text.csv --output-dir outputs/text_run --n-clusters 4
```

Run the categorical-label path:

```powershell
& 'C:\Users\32082\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m stylegcl generate-label --output data/synthetic_label.csv --num-tasks 200 --num-workers 60
& 'C:\Users\32082\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m stylegcl run-label --input data/synthetic_label.csv --output-dir outputs/label_run --n-clusters 4
```

Run lightweight baselines and paper-style experiment sweeps:

```powershell
& 'C:\Users\32082\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m stylegcl run-baselines --input data/synthetic_text.csv --output-dir outputs/baselines
& 'C:\Users\32082\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m stylegcl attack-sweep --output-dir outputs/attack_sweep --num-tasks 160 --num-workers 80
& 'C:\Users\32082\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m stylegcl ablation --input data/synthetic_text.csv --output-dir outputs/ablation
```

Outputs:

- `worker_predictions.csv`: worker-level cluster and suspicion score
- `metrics.json`: losses, clustering metadata, and evaluation metrics when `is_colluder` exists
- `worker_embeddings.npy`: learned worker style embeddings
- `task_contributions.csv`: text path only, tasks contributing most to cluster cohesion
- `baseline_metrics.json`, `attack_sweep_metrics.json`, `ablation_metrics.json`: lightweight reproduction reports

## Paper Mapping

- Section 3.2 semantic disentanglement: `StyleGCLDetector._disentangle`
- Content reconstruction, orthogonal, style consistency losses: `StyleGCLDetector._diagnostic_losses`
- Section 3.3 worker graph and GAT-style aggregation: `_worker_adjacency`, `_graph_attention`
- Section 3.4 InfoNCE and spectral clustering: `_info_nce`, `spectral_cluster`
- Section 3.5 categorical label complement: `LabelCorrelationDetector`
- Section 4.1 dataset construction and attack injection: `synthetic.py`

## Notes

The default implementation is intentionally dependency-light and runnable with NumPy only. It uses a deterministic hashing sentence encoder as a local fallback for the paper's frozen Sentence-BERT `all-mpnet-base-v2`. For full neural training, install PyTorch, PyTorch Geometric, and sentence-transformers, then extend the optional modules in `stylegcl/torch_model.py`.
