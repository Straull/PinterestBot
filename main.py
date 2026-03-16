"""Point d'entrée de l'application Trading Bot V1."""

import sys
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

from trading_bot.gui.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Trading Bot V1")

    # Police par défaut
    font = QFont("Segoe UI", 10)
    font.setStyleStrategy(QFont.StyleStrategy.PreferAntialias)
    app.setFont(font)

    # Fenêtre principale
    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
