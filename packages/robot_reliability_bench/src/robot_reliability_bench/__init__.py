"""Independent reliability bench. No perception-lab imports are required."""

__version__ = "0.2.0"

from .replay import replay_system


def run_benchmark(*args, **kwargs):
    """Import the measurement runner lazily, so replay has no runtime backend dependency."""
    from .runner import run_benchmark as run
    return run(*args, **kwargs)


def replay_model(*args, **kwargs):
    from .integration import replay_model as run
    return run(*args, **kwargs)


__all__ = ["run_benchmark", "replay_system", "replay_model", "__version__"]
