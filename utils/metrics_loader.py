import json
import os
import logging

log = logging.getLogger(__name__)

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config")


def load_disease_config():
    path = os.path.join(CONFIG_DIR, "diseases.json")
    try:
        with open(path) as f:
            return json.load(f)["diseases"]
    except FileNotFoundError:
        log.error(f"Config file not found: {path}")
        return {}
    except (json.JSONDecodeError, KeyError) as e:
        log.error(f"Invalid config file {path}: {e}")
        return {}


def load_references():
    path = os.path.join(CONFIG_DIR, "references.json")
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        log.error(f"References file not found: {path}")
        return {}
    except json.JSONDecodeError as e:
        log.error(f"Invalid references file {path}: {e}")
        return {}
