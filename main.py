"""
Launcher da Central Gráfica do Hockey de Robôs RoboCore (Classe 1lb / HockeyBot).
Execute:
    python main.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ui.hockey_gui import run_hockey_gui

if __name__ == "__main__":
    run_hockey_gui()
