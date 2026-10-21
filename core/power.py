"""Keep a Windows training process awake without changing system power settings."""

from contextlib import contextmanager
import os


@contextmanager
def keep_system_awake(enabled=False):
    setter = None
    if enabled and os.name == "nt":
        import ctypes

        setter = ctypes.windll.kernel32.SetThreadExecutionState
        if not setter(0x80000001):  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED; display may turn off.
            raise OSError("Não foi possível inibir a suspensão durante o treino")
    try:
        yield
    finally:
        if setter:
            setter(0x80000000)
