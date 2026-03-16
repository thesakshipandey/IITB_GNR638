#!/usr/bin/env python3
"""Apply the selected tuned hyperparameters into config.py.

Reads:
  results/best_tuned_config.json

Updates keys in:
  config.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BEST_JSON = ROOT / "results" / "best_tuned_config.json"
CONFIG_PY = ROOT / "config.py"


def _py_literal(value):
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, str):
        return repr(value)
    return repr(value)


def _replace_assignment(text: str, key: str, value) -> tuple[str, bool]:
    pattern = re.compile(
        rf"^(?P<prefix>\s*{re.escape(key)}\s*=\s*)(?P<val>[^#\n]*)(?P<comment>\s*(#.*)?)$",
        re.MULTILINE,
    )
    repl = rf"\g<prefix>{_py_literal(value)}\g<comment>"
    new_text, n = pattern.subn(repl, text, count=1)
    return new_text, (n == 1)


def main() -> int:
    if not BEST_JSON.exists():
        raise FileNotFoundError(f"Missing tuned config file: {BEST_JSON}")
    if not CONFIG_PY.exists():
        raise FileNotFoundError(f"Missing config file: {CONFIG_PY}")

    best = json.loads(BEST_JSON.read_text(encoding="utf-8"))
    overrides = best.get("overrides", {})
    if not overrides:
        print("No overrides found in best_tuned_config.json; nothing to apply.")
        return 0

    original = CONFIG_PY.read_text(encoding="utf-8")
    updated = original
    changed = []

    for key, value in overrides.items():
        updated, ok = _replace_assignment(updated, key, value)
        if not ok:
            print(f"WARNING: Could not find assignment for {key} in config.py")
        else:
            changed.append((key, value))

    if updated == original:
        print("No config changes were applied.")
        return 0

    backup = CONFIG_PY.with_suffix(".py.bak_tuned")
    backup.write_text(original, encoding="utf-8")
    CONFIG_PY.write_text(updated, encoding="utf-8")

    print("Applied tuned config values to config.py")
    print(f"Selected candidate: {best.get('selected_candidate')}")
    print("Updated keys:")
    for key, value in changed:
        print(f"  - {key} = {value}")
    print(f"Backup: {backup}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
