"""
Utility helpers for the Solar Power Forecasting project.

Provides:
  - Consistent matplotlib plot styling
  - Streamlit concept note rendering
"""

import streamlit as st


# ── Color Palette ──
C_RF = "#43A047"      # Green — RandomForest
C_ACTUAL = "#FFA726"  # Amber — Actual values
C_ACCENT = "#7E57C2"  # Purple — Accents
C_GOLD = "#FFD54F"    # Gold — Highlights


def style_plot(ax, title, xlabel="", ylabel=""):
    """Apply consistent dark-themed styling to a matplotlib Axes.

    Args:
        ax: Matplotlib Axes object.
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
    """
    ax.set_title(title, fontsize=14, fontweight="bold", pad=14, color="#E5E7EB")
    ax.set_xlabel(xlabel, fontsize=10, color="#9CA3AF")
    ax.set_ylabel(ylabel, fontsize=10, color="#9CA3AF")
    ax.grid(alpha=0.12, linestyle="--", color="#4B5563")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#374151")
    ax.spines["left"].set_color("#374151")
    ax.tick_params(colors="#9CA3AF")
    ax.set_facecolor("#111827")


def concept_note(text):
    """Render an academic concept note in the Streamlit UI.

    Displays a styled blue-bordered info box with educational content.

    Args:
        text: HTML string with concept explanation.
    """
    st.markdown(f'<div class="concept-box">{text}</div>', unsafe_allow_html=True)
