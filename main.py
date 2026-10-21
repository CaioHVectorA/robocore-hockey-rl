"""
Launcher do Hockey Lab, com perfis de competição e mesa de treino.
Execute:
    python main.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ui.hockey_gui import main

if __name__ == "__main__":
    main()
