from PySide6.QtWidgets import QApplication
import sys

# from ui.login_window import LoginWindow
from ui.main_menu_window import MainMenu

if __name__ == "__main__":
    app = QApplication(sys.argv)
    # login = LoginWindow()
    # login.show()
    main_menu = MainMenu()
    main_menu.show()
    sys.exit(app.exec())
