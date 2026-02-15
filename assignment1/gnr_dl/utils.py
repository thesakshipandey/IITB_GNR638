from __future__ import annotations

import json
import random
from typing import Any, Dict


def G16SetSeed(seed: int) -> None:
    random.seed(seed)


def G16LoadJsonConfig(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def G16MergeWithDefaults(config: Dict[str, Any], defaults: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(defaults)
    merged.update(config)
    return merged


set_seed = G16SetSeed
load_json_config = G16LoadJsonConfig
merge_with_defaults = G16MergeWithDefaults
