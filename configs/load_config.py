import yaml
from pathlib import Path

def load_config(name: str):
    config_path = Path(__file__).resolve().parent / f"{name}.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)