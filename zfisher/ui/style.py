from napari.utils.theme import get_theme, register_theme
from qtpy.QtGui import QColor, QFont

# zFISHer color palette
COLORS = {
    'background': '#1a1421',        # Deep Amethyst Purple
    'canvas': '#000000',            # Black
    'primary': '#b2f2bb',           # Mint
    'primary_darker': '#4aa87c',    # Darker Mint for buttons
    'secondary': '#352c42',         # Muted Amethyst
    'text': 'white',
    'toolbox_tab_bg': '#251f2e',
    'toolbox_tab_selected_bg': '#352c42',
    'nested_toolbox_tab_bg': '#2c2538',
    'nested_toolbox_tab_selected_bg': '#4d4464',
    'separator_color': '#7a6b8a',
}

# Napari theme
def register_napari_theme():
    """Create and register the custom zFISHer napari theme."""
    try:
        custom_theme = get_theme('dark')
        custom_theme.background = COLORS['background']
        custom_theme.canvas = COLORS['canvas']
        custom_theme.primary = COLORS['primary_darker']
        custom_theme.highlight = COLORS['primary_darker']
        custom_theme.secondary = COLORS['secondary']
        
        register_theme('zfisher_theme', custom_theme, 'dark')
        return 'zfisher_theme'
    except Exception as e:
        print(f"Theme registration failed: {e}")
        return 'dark' # fallback theme

# PyQt Stylesheets
TOOLBOX_STYLESHEET = f"""
    QToolBox::tab {{
        color: {COLORS['primary']};
        background: {COLORS['toolbox_tab_bg']};
        font-weight: bold;
        border-radius: 4px;
    }}
    QToolBox::tab:selected {{
        background: {COLORS['toolbox_tab_selected_bg']};
        border: 1px solid {COLORS['primary']};
    }}
    QLabel#widgetHeader {{
        color: {COLORS['primary']};
        font-weight: bold;
    }}
    QLabel#widgetInfo {{
        color: {COLORS['primary']};
    }}
"""

NESTED_TOOLBOX_STYLESHEET = f"""
    QToolBox::tab {{
        color: {COLORS['primary']};
        background: {COLORS['nested_toolbox_tab_bg']};
        font-weight: bold;
        border-radius: 4px;
    }}
    QToolBox::tab:selected {{
        background: {COLORS['nested_toolbox_tab_selected_bg']};
        border: 1px solid {COLORS['primary']};
    }}
"""

# Start Session Widget Styles
LINE_STYLESHEET = f"margin: 15px 0; color: {COLORS['separator_color']};"
SEPARATOR_STYLESHEET = f"""
    margin: 10px 0px;
    border-bottom: 1px solid {COLORS['separator_color']};
    max-height: 1px;
    min-height: 1px;
"""

# Welcome Widget HTML styles
WELCOME_WIDGET_STYLE = {
    'h1': "style='font-size: 50px; color: white; margin-bottom: 0px;'",
    'p': f"style='font-size: 24px; color: {COLORS['primary']}; margin-top: 0px;'",
}

CREATE_WELCOME_WIDGET_STYLE = {
    'h1': f"style='color: {COLORS['primary']};'",
}

# Fonts
SCALE_BAR_FONT = QFont("Arial", 12)
SCALE_BAR_FONT.setBold(True)
TEXT_OVERLAY_FONT_SIZE = 10

# Colors as QColor objects
SCALE_BAR_PEN_COLOR = QColor(COLORS['text'])
SCALE_BAR_FONT_COLOR = QColor(COLORS['text'])
WELCOME_WIDGET_BG_COLOR = QColor(COLORS['canvas'])
