"""
config.py

Simple YAML/JSON configuration loader for game automation.
"""

import json
import yaml
from typing import Any, Dict


def load_config(path: str) -> Dict[str, Any]:
    """
    Loads a configuration file (YAML or JSON) and returns it as a dictionary.

    Supports:
    - .yaml / .yml
    - .json

    Raises:
        ValueError: if the file extension is unsupported
        FileNotFoundError: if the file does not exist
        yaml.YAMLError / json.JSONDecodeError: if parsing fails
    """
    if path.endswith((".yaml", ".yml")):
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    elif path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    else:
        raise ValueError(
            f"Unsupported config format for '{path}'. "
            "Use .yaml, .yml, or .json"
        )
