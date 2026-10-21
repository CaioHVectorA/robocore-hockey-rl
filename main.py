"""
Launcher do Hockey Lab, com perfis de competição e mesa de treino.
Execute:
    python main.py
    python main.py --headless --preset solo_goal
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if "--headless" in sys.argv[1:]:
    from rl.cli import main
else:
    from ui.hockey_gui import main

if __name__ == "__main__":
    main()
