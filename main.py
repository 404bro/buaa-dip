import sys
from PySide6.QtWidgets import QApplication
from main_window import MainWindow

if __name__ == "__main__":
    app = QApplication(sys.argv)

    main_win = MainWindow()
    main_win.show()

    sys.exit(app.exec())
