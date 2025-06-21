import json
import sys
import os
from pathlib import Path

from PySide6.QtWidgets import QApplication, QDialog
from ui.steam_path_dialog import SteamPathDialog
from ui.camera_select_dialog import CameraSelectDialog
from logic.steam_path_search import get_steam_path, save_steam_path

def is_valid_steam_path(path):
    return path and os.path.isdir(path) and os.path.exists(os.path.join(path, "steam.exe"))

def save_camera_micro_index(index, micro_index):
    cnfg_pth = str(Path(__file__).resolve().parent / Path("config"))
    with open(cnfg_pth + "/camera_micro_index.json", "w") as f:
        json.dump({"camera_index": index, "micro_index": micro_index}, f)

if __name__ == "__main__":
    app = QApplication(sys.argv)

    (Path(__file__).resolve().parent / Path("config")).mkdir(exist_ok=True)

    steam_path = get_steam_path()
    while not is_valid_steam_path(steam_path):
        on_selected = lambda path: save_steam_path(path)
        dialog = SteamPathDialog(on_path_selected=on_selected)
        if dialog.exec() != QDialog.Accepted:
            sys.exit()

        steam_path = get_steam_path()

    cam_dialog = CameraSelectDialog(on_camera_selected=save_camera_micro_index)
    if cam_dialog.exec() != QDialog.Accepted:
        sys.exit()

    from ui.main_menu_window import MainMenu
    main_menu = MainMenu()
    main_menu.show()
    sys.exit(app.exec())
