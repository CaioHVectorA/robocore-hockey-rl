"""Interface gráfica interativa do simulador de Hockey de Robôs RoboCore."""


def run_hockey_gui(*args, **kwargs):
    from .hockey_gui import run_hockey_gui as run

    return run(*args, **kwargs)


__all__ = ["run_hockey_gui"]
