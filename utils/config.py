from pathlib import Path

import yaml


def load_config(config_path):
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError("Config file not found: {}".format(path))
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)
