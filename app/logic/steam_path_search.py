import json
import os
from pathlib import Path

config_path = "config/steam_folder.json"
CONFIG_PATH = Path(__file__).resolve().parent.parent / Path(config_path)
(Path(__file__).resolve().parent.parent / Path("config")).mkdir(exist_ok=True)


def get_steam_path():
    if not os.path.exists(CONFIG_PATH):
        return None
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return Path(data.get("steam_path"))


def save_steam_path(path):
    with open(CONFIG_PATH, "w") as f:
        json.dump({"steam_path": path}, f)