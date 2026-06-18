"""StyleGCL-Net reproduction toolkit.

The package contains a dependency-light implementation of the paper pipeline:
text style fingerprint detection, a categorical-label correlation path, data
loading utilities, synthetic benchmark generation, clustering, and metrics.
"""

from .data import CrowdTextRecord, LabelRecord, load_label_records, load_text_records
from .model_numpy import StyleGCLDetector
from .label_correlation import LabelCorrelationDetector

__all__ = [
    "CrowdTextRecord",
    "LabelRecord",
    "StyleGCLDetector",
    "LabelCorrelationDetector",
    "load_label_records",
    "load_text_records",
]

__version__ = "0.1.0"
