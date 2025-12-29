# qusa/utils/config.py

import os
import yaml

from pathlib import Path


def load_config(config_path="config.yaml"):
    """
    Load configuration from a YAML file.

    Parameters:
        1) config_path (str): Path to the YAML configuration file.
    """

    config_path = Path(config_path).expanduser()

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    # expand user paths
    for section in config.values():
        if isinstance(section, dict):
            for key, value in section.items():
                if isinstance(value, str) and value.startswith("~"):
                    section[key] = os.path.expanduser(value)
                if isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        if isinstance(subvalue, str) and subvalue.startswith("~"):
                            section[key][subkey] = os.path.expanduser(subvalue)
    return config
