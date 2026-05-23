"""Configuration loading and small shared helpers.

The whole model is parameterised from config/config.yaml so a non-developer can
tune it without touching code. This module just loads that YAML and offers a
couple of helpers (age-band parsing) used across the population/demand modules.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load the YAML config into a plain nested dict."""
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with open(cfg_path, "r") as fh:
        return yaml.safe_load(fh)


def parse_age_band(label: str) -> tuple[int, int]:
    """Turn an age-band label into an inclusive (low, high) integer range.

    "0-4"  -> (0, 4);  "85+" -> (85, 120).
    """
    label = label.strip()
    if label.endswith("+"):
        low = int(label[:-1])
        return low, 120
    low_s, high_s = label.split("-")
    return int(low_s), int(high_s)


def make_rng(seed: int, name: str) -> np.random.Generator:
    """Return an independent, reproducible RNG stream for a named component.

    Lets each module (demand, simulation, ...) draw from its own stream so they
    are statistically independent yet fully determined by the global seed.
    """
    h = int(hashlib.sha256(name.encode()).hexdigest(), 16) % (2 ** 32)
    return np.random.default_rng(np.random.SeedSequence([int(seed), h]))
