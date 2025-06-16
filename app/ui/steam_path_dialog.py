from PySide6.QtWidgets import QDialog, QVBoxLayout, QLabel, QPushButton, QFileDialog

class SteamPathDialog(QDialog):
    def __init__(self, on_path_selected=None):
        super().__init__()
        self.setWindowTitle("Указать путь до Steam папки")
        self.setFixedSize(400, 120)

        self.on_path_selected = on_path_selected

        layout = QVBoxLayout()
        layout.addWidget(QLabel("Укажите путь до папки steam на вашем устройстве."))

        btn = QPushButton("Выбрать путь...")
        btn.clicked.connect(self.select_path)
        layout.addWidget(btn)

        self.setLayout(layout)

    def select_path(self):
        path = QFileDialog.getExistingDirectory(self, "Выберите папку Steam")
        if path:
            if self.on_path_selected:
                self.on_path_selected(path)
            self.accept()