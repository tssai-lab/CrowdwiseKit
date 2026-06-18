from __future__ import annotations

import hashlib
import math
import re
import string
from dataclasses import dataclass

import numpy as np

from .utils import l2_normalize


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]|[^\s]", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def word_count(text: str) -> int:
    return len([tok for tok in tokenize(text) if any(ch.isalnum() for ch in tok)])


def _hash_index(feature: str, dim: int) -> int:
    digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % dim


@dataclass
class HashingSentenceEncoder:
    """Frozen encoder used when Sentence-BERT is unavailable.

    It produces a 768-dimensional vector by combining lexical n-grams, character
    n-grams, punctuation habits, length statistics, and shallow style markers.
    The interface mirrors a sentence encoder so a real SBERT backend can replace
    it without changing downstream code.
    """

    dim: int = 768
    style_bins: int = 48

    def encode_one(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float64)
        tokens = tokenize(text)
        words = [tok for tok in tokens if any(ch.isalnum() for ch in tok)]
        punct = [tok for tok in tokens if tok in string.punctuation]

        for n in (1, 2, 3):
            for i in range(0, max(0, len(words) - n + 1)):
                feat = "w{}:{}".format(n, " ".join(words[i : i + n]))
                vec[_hash_index(feat, self.dim - self.style_bins)] += 1.0 / n

        clean = re.sub(r"\s+", " ", text.lower())
        for n in (3, 4, 5):
            for i in range(0, max(0, len(clean) - n + 1)):
                feat = f"c{n}:{clean[i:i+n]}"
                vec[_hash_index(feat, self.dim - self.style_bins)] += 0.35 / n

        base = self.dim - self.style_bins
        chars = max(len(text), 1)
        word_total = max(len(words), 1)
        unique_words = len(set(words))
        avg_word_len = sum(len(w) for w in words) / word_total
        sentence_count = max(1, len(re.findall(r"[.!?。！？]", text)))
        comma_count = len(re.findall(r"[,;:，；：]", text))
        digit_count = sum(ch.isdigit() for ch in text)
        upper_count = sum(ch.isupper() for ch in text)
        space_count = sum(ch.isspace() for ch in text)

        stats = [
            math.log1p(chars),
            math.log1p(word_total),
            unique_words / word_total,
            avg_word_len / 12.0,
            len(punct) / max(len(tokens), 1),
            comma_count / sentence_count,
            digit_count / chars,
            upper_count / chars,
            space_count / chars,
            text.count("(") + text.count(")") + text.count("[") + text.count("]"),
            text.count("-") + text.count("_"),
            text.count('"') + text.count("'"),
        ]
        discourse = [
            "however",
            "therefore",
            "because",
            "overall",
            "first",
            "second",
            "notably",
            "clearly",
            "in conclusion",
            "for example",
            "moreover",
            "nevertheless",
        ]
        lowered = text.lower()
        stats.extend([1.0 if marker in lowered else 0.0 for marker in discourse])
        while len(stats) < self.style_bins:
            stats.append(0.0)
        vec[base:] = np.asarray(stats[: self.style_bins], dtype=np.float64)
        return l2_normalize(vec)

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float64)
        return np.vstack([self.encode_one(text) for text in texts])
