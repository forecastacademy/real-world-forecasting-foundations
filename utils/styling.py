"""
utils/styling.py

Global matplotlib/seaborn styling for Forecast Academy.
This ensures that all charts are clean, readable, and consistent
across every notebook, without duplicating code in tsforge.
"""

import matplotlib.pyplot as plt
import seaborn as sns

# Forecast Academy brand-friendly palette
FA_COLORS = {
    "primary": "#2C32D5",     # deep blue
    "secondary": "#620E96",   # purple
    "accent": "#F2A900",      # gold
    "light_gray": "#E5E5E5",
    "dark_gray": "#404040",
}

def apply_style():
    """
    Apply a clean, consistent matplotlib + seaborn style for all charts.
    Call this once at the top of each notebook:
    
        from utils.styling import apply_style
        apply_style()
    """
    # Base style
    sns.set_theme(
        context="notebook",
        style="whitegrid",
        palette=[FA_COLORS["primary"], FA_COLORS["secondary"], FA_COLORS["accent"]],
    )

    # Matplotlib rcParams
    plt.rcParams.update({
        "figure.figsize": (10, 5),
        "figure.dpi": 120,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 11,
        "axes.edgecolor": FA_COLORS["dark_gray"],
        "axes.titleweight": "bold",
        "grid.alpha": 0.25,
    })

    print("Forecast Academy style applied ✔")


# Optional convenience for students
def set_figsize(width=10, height=5, dpi=120):
    plt.rcParams["figure.figsize"] = (width, height)
    plt.rcParams["figure.dpi"] = dpi
