"""Plotext chart rendering utilities for the TUI."""

import plotext as plt
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import io


def render_histogram(series: pd.Series, title: str = "", width: int = 60, height: int = 15) -> str:
    """Render a histogram of a numeric series as a string."""
    plt.clear_figure()
    plt.clear_data()
    data = series.dropna().values.tolist()
    plt.hist(data, bins=20)
    plt.title(title or series.name or "Histogram")
    plt.plotsize(width, height)
    plt.theme("dark")
    return plt.build()


def render_scatter(x: pd.Series, y: pd.Series, title: str = "", width: int = 60, height: int = 15) -> str:
    """Render a scatter plot as a string."""
    plt.clear_figure()
    plt.clear_data()
    plt.scatter(x.values.tolist(), y.values.tolist())
    plt.title(title or f"{x.name} vs {y.name}")
    plt.xlabel(str(x.name))
    plt.ylabel(str(y.name))
    plt.plotsize(width, height)
    plt.theme("dark")
    return plt.build()


def render_bar(labels: List[str], values: List[float], title: str = "", width: int = 60, height: int = 15) -> str:
    """Render a horizontal bar chart as a string."""
    plt.clear_figure()
    plt.clear_data()
    plt.bar(labels, values)
    plt.title(title)
    plt.plotsize(width, height)
    plt.theme("dark")
    return plt.build()


def render_correlation_matrix(corr_dict: Dict[str, Dict[str, float]], width: int = 60) -> str:
    """Render a correlation matrix as colored text blocks."""
    if not corr_dict:
        return "  No numeric features for correlation matrix."

    cols = list(corr_dict.keys())
    if not cols:
        return "  No data."

    # Header
    max_label = max(len(c) for c in cols)
    header_labels = [c[:7].rjust(7) for c in cols]
    header = " " * (max_label + 2) + " ".join(header_labels)
    lines = [header]

    for row in cols:
        label = row[:max_label].ljust(max_label)
        cells = []
        for col in cols:
            val = corr_dict.get(col, {}).get(row, 0)
            # Color code
            if val > 0.7:
                color = "\033[92m"  # green
            elif val > 0.3:
                color = "\033[93m"  # yellow
            elif val < -0.3:
                color = "\033[91m"  # red
            else:
                color = "\033[90m"  # grey
            cells.append(f"{color}{val:+.2f}\033[0m")
        lines.append(f"  {label} " + "  ".join(cells))

    return "\n".join(lines)


def render_box_plot(df: pd.DataFrame, columns: List[str] = None, title: str = "", width: int = 60, height: int = 15) -> str:
    """Render box plots for numeric columns."""
    plt.clear_figure()
    plt.clear_data()
    cols = columns or df.select_dtypes(include=[np.number]).columns.tolist()[:5]
    data = [df[c].dropna().values.tolist() for c in cols]
    plt.box(data, labels=cols)
    plt.title(title or "Box Plots")
    plt.plotsize(width, height)
    plt.theme("dark")
    return plt.build()
