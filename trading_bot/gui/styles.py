"""Thème sombre moderne pour l'interface de trading."""

# Palette de couleurs
COLORS = {
    "bg_dark": "#0d1117",
    "bg_panel": "#161b22",
    "bg_card": "#1c2333",
    "bg_input": "#0d1117",
    "border": "#30363d",
    "border_light": "#484f58",
    "text": "#e6edf3",
    "text_secondary": "#8b949e",
    "text_muted": "#6e7681",
    "accent_blue": "#58a6ff",
    "accent_green": "#3fb950",
    "accent_red": "#f85149",
    "accent_orange": "#d29922",
    "accent_purple": "#bc8cff",
    "chart_bg": "#0d1117",
    "chart_grid": "#21262d",
    "chart_up": "#3fb950",
    "chart_down": "#f85149",
}

DARK_THEME = f"""
QMainWindow {{
    background-color: {COLORS['bg_dark']};
}}

QWidget {{
    color: {COLORS['text']};
    font-family: 'Segoe UI', 'Inter', 'Roboto', sans-serif;
    font-size: 13px;
}}

/* Panneaux */
QFrame#panel {{
    background-color: {COLORS['bg_panel']};
    border: 1px solid {COLORS['border']};
    border-radius: 8px;
    padding: 12px;
}}

QFrame#card {{
    background-color: {COLORS['bg_card']};
    border: 1px solid {COLORS['border']};
    border-radius: 6px;
    padding: 8px;
}}

/* Labels */
QLabel {{
    color: {COLORS['text']};
    background: transparent;
    border: none;
}}

QLabel#title {{
    font-size: 18px;
    font-weight: bold;
    color: {COLORS['accent_blue']};
}}

QLabel#section {{
    font-size: 14px;
    font-weight: 600;
    color: {COLORS['text']};
    padding: 4px 0;
}}

QLabel#muted {{
    color: {COLORS['text_muted']};
    font-size: 11px;
}}

QLabel#value_up {{
    color: {COLORS['accent_green']};
    font-size: 20px;
    font-weight: bold;
}}

QLabel#value_down {{
    color: {COLORS['accent_red']};
    font-size: 20px;
    font-weight: bold;
}}

QLabel#value_neutral {{
    color: {COLORS['accent_blue']};
    font-size: 20px;
    font-weight: bold;
}}

/* Input fields */
QLineEdit {{
    background-color: {COLORS['bg_input']};
    border: 1px solid {COLORS['border']};
    border-radius: 6px;
    padding: 8px 12px;
    color: {COLORS['text']};
    font-size: 14px;
    selection-background-color: {COLORS['accent_blue']};
}}

QLineEdit:focus {{
    border: 1px solid {COLORS['accent_blue']};
}}

/* ComboBox */
QComboBox {{
    background-color: {COLORS['bg_input']};
    border: 1px solid {COLORS['border']};
    border-radius: 6px;
    padding: 8px 12px;
    color: {COLORS['text']};
    font-size: 13px;
    min-width: 120px;
}}

QComboBox:hover {{
    border: 1px solid {COLORS['border_light']};
}}

QComboBox::drop-down {{
    border: none;
    width: 30px;
}}

QComboBox::down-arrow {{
    image: none;
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-top: 6px solid {COLORS['text_secondary']};
    margin-right: 10px;
}}

QComboBox QAbstractItemView {{
    background-color: {COLORS['bg_panel']};
    border: 1px solid {COLORS['border']};
    color: {COLORS['text']};
    selection-background-color: {COLORS['accent_blue']};
    border-radius: 4px;
    padding: 4px;
}}

/* Buttons */
QPushButton {{
    background-color: {COLORS['bg_card']};
    border: 1px solid {COLORS['border']};
    border-radius: 6px;
    padding: 10px 16px;
    color: {COLORS['text']};
    font-size: 13px;
    font-weight: 500;
}}

QPushButton:hover {{
    background-color: {COLORS['border']};
    border: 1px solid {COLORS['border_light']};
}}

QPushButton:pressed {{
    background-color: {COLORS['bg_input']};
}}

QPushButton:disabled {{
    color: {COLORS['text_muted']};
    background-color: {COLORS['bg_dark']};
    border: 1px solid {COLORS['bg_card']};
}}

QPushButton#primary {{
    background-color: {COLORS['accent_blue']};
    color: #ffffff;
    border: none;
    font-weight: 600;
}}

QPushButton#primary:hover {{
    background-color: #4090e0;
}}

QPushButton#primary:disabled {{
    background-color: #2a4a6b;
    color: #6e7681;
}}

QPushButton#success {{
    background-color: {COLORS['accent_green']};
    color: #ffffff;
    border: none;
    font-weight: 600;
}}

QPushButton#success:hover {{
    background-color: #2ea043;
}}

QPushButton#danger {{
    background-color: {COLORS['accent_red']};
    color: #ffffff;
    border: none;
    font-weight: 600;
}}

QPushButton#danger:hover {{
    background-color: #da3633;
}}

/* Progress Bar */
QProgressBar {{
    background-color: {COLORS['bg_input']};
    border: 1px solid {COLORS['border']};
    border-radius: 4px;
    height: 8px;
    text-align: center;
    color: transparent;
}}

QProgressBar::chunk {{
    background-color: {COLORS['accent_blue']};
    border-radius: 3px;
}}

/* SpinBox */
QSpinBox {{
    background-color: {COLORS['bg_input']};
    border: 1px solid {COLORS['border']};
    border-radius: 6px;
    padding: 8px 12px;
    color: {COLORS['text']};
    font-size: 13px;
}}

QSpinBox:focus {{
    border: 1px solid {COLORS['accent_blue']};
}}

/* ScrollBar */
QScrollBar:vertical {{
    background-color: {COLORS['bg_dark']};
    width: 8px;
    border: none;
}}

QScrollBar::handle:vertical {{
    background-color: {COLORS['border']};
    border-radius: 4px;
    min-height: 30px;
}}

QScrollBar::handle:vertical:hover {{
    background-color: {COLORS['border_light']};
}}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}

/* TextEdit */
QTextEdit {{
    background-color: {COLORS['bg_input']};
    border: 1px solid {COLORS['border']};
    border-radius: 6px;
    padding: 8px;
    color: {COLORS['text']};
    font-family: 'Consolas', 'Fira Code', monospace;
    font-size: 12px;
}}

/* StatusBar */
QStatusBar {{
    background-color: {COLORS['bg_panel']};
    border-top: 1px solid {COLORS['border']};
    color: {COLORS['text_secondary']};
    font-size: 12px;
    padding: 4px;
}}

/* Splitter */
QSplitter::handle {{
    background-color: {COLORS['border']};
    width: 1px;
    height: 1px;
}}

/* TabWidget */
QTabWidget::pane {{
    border: 1px solid {COLORS['border']};
    border-radius: 6px;
    background-color: {COLORS['bg_panel']};
}}

QTabBar::tab {{
    background-color: {COLORS['bg_dark']};
    border: 1px solid {COLORS['border']};
    padding: 8px 16px;
    margin-right: 2px;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    color: {COLORS['text_secondary']};
}}

QTabBar::tab:selected {{
    background-color: {COLORS['bg_panel']};
    color: {COLORS['text']};
    border-bottom-color: {COLORS['bg_panel']};
}}

QTabBar::tab:hover {{
    color: {COLORS['text']};
}}

/* GroupBox */
QGroupBox {{
    border: 1px solid {COLORS['border']};
    border-radius: 6px;
    margin-top: 8px;
    padding-top: 16px;
    color: {COLORS['text']};
    font-weight: 500;
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 4px;
}}

/* TableWidget */
QTableWidget {{
    background-color: {COLORS['bg_dark']};
    border: 1px solid {COLORS['border']};
    border-radius: 4px;
    gridline-color: {COLORS['border']};
    color: {COLORS['text']};
    selection-background-color: {COLORS['accent_blue']}40;
    alternate-background-color: {COLORS['bg_card']};
}}

QTableWidget::item {{
    padding: 6px 8px;
    border: none;
}}

QTableWidget::item:selected {{
    background-color: {COLORS['accent_blue']}30;
    color: {COLORS['text']};
}}

QHeaderView::section {{
    background-color: {COLORS['bg_panel']};
    color: {COLORS['text_secondary']};
    border: none;
    border-bottom: 2px solid {COLORS['border']};
    border-right: 1px solid {COLORS['border']};
    padding: 8px 6px;
    font-weight: 600;
    font-size: 11px;
    text-transform: uppercase;
}}

/* DoubleSpinBox */
QDoubleSpinBox {{
    background-color: {COLORS['bg_input']};
    border: 1px solid {COLORS['border']};
    border-radius: 6px;
    padding: 8px 12px;
    color: {COLORS['text']};
    font-size: 13px;
}}

QDoubleSpinBox:focus {{
    border: 1px solid {COLORS['accent_blue']};
}}
"""
