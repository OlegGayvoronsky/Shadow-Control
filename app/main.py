import sys
import os

from PySide6.QtWidgets import QApplication, QDialog
from ui.steam_path_dialog import SteamPathDialog
from logic.steam_path_search import get_steam_path, save_steam_path

def is_valid_steam_path(path):
    if not path:
        return False
    return os.path.isdir(path) and os.path.exists(os.path.join(path, "steam.exe"))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    steam_path = get_steam_path()
    while not steam_path or not is_valid_steam_path(steam_path):
        on_selected = lambda path: save_steam_path(path)


        dialog = SteamPathDialog(on_path_selected=on_selected)
        if dialog.exec() != QDialog.Accepted:
            break

        steam_path = get_steam_path()

    from ui.main_menu_window import MainMenu
    main_menu = MainMenu()
    main_menu.show()
    sys.exit(app.exec())
