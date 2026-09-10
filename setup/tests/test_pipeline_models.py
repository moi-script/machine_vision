import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import pipeline  # noqa: E402


def test_every_model_weight_lives_under_models_dir():
    """Weights must be under models/ to be LFS-tracked and reach the Pi.

    runs/ is gitignored, so a weight referenced there exists only on the
    machine that trained it.
    """
    for key, cfg in pipeline.MODELS.items():
        if key == "none":
            continue
        assert cfg["weights"].startswith("models/"), \
            f"{key} weights {cfg['weights']!r} are outside models/ and will not reach the Pi"


def test_every_model_weight_file_exists():
    for key, cfg in pipeline.MODELS.items():
        if key == "none":
            continue
        path = pipeline._resolve(cfg["weights"])
        assert os.path.exists(path), f"{key} weights missing at {path}"
