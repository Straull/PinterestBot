"""Point d'entree de l'application Trading Bot V1."""

import sys
import os
import traceback

# Supprimer les warnings de font pyqtgraph (cosmetic, pas critique)
os.environ["QT_LOGGING_RULES"] = "qt.qpa.fonts.warning=false"

from PyQt6.QtWidgets import QApplication, QMessageBox
from PyQt6.QtGui import QFont


def excepthook(exc_type, exc_value, exc_tb):
    """Attrape les exceptions non gerees pour eviter les crashs silencieux."""
    msg = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
    print(f"ERREUR NON GEREE:\n{msg}", file=sys.stderr)
    try:
        QMessageBox.critical(None, "Erreur", f"Une erreur est survenue:\n\n{exc_value}\n\nDetails dans la console.")
    except Exception:
        pass


def main():
    sys.excepthook = excepthook

    app = QApplication(sys.argv)
    app.setApplicationName("Trading Bot V1")

    # Police par defaut
    font = QFont("Segoe UI")
    font.setPointSize(10)
    font.setStyleStrategy(QFont.StyleStrategy.PreferAntialias)
    app.setFont(font)

    from trading_bot.gui.main_window import MainWindow

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
