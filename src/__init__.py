import yaml
import os
import sys
from logging import config
import logging
import coloredlogs

dirname = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

sys.path.append(dirname)
from src.data.logging import setup_logging


def setup_logging(
    default_path=dirname + "/src/data/logging.yaml",
    default_level=logging.INFO,
    env_key="LOG_CFG",
):
    """
    | **@author:** Prathyush SP
    | Logging Setup
    """
    path = default_path
    value = os.getenv(env_key, None)
    if value:
        path = value
    if os.path.exists(path):
        with open(path, "rt") as f:
            try:
                config = yaml.safe_load(f.read())
                logging.config.dictConfig(config)

            except Exception as e:
                print(e)
                print("Error in Logging Configuration. Using default configs")
                logging.basicConfig(level=default_level)
                coloredlogs.install(level=default_level)
    else:
        logging.basicConfig(level=default_level)
        coloredlogs.install(level=default_level)
        print("Failed to load configuration file. Using default configs")


# Should be the first statement in the module to avoid circular dependency issues.
setup_logging()
