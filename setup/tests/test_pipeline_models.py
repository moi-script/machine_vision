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


# ── catalog imgsz must match what a Worker actually runs at ────

def _worker_imgsz(model_key: str, backend: str) -> int:
    """The same override Worker.__init__ applies, without constructing a
    real Worker (which opens a capture and loads a model)."""
    from config import settings
    cfg = dict(pipeline.MODELS[model_key])
    if backend == "ncnn" and model_key in settings.PI_IMGSZ:
        cfg["imgsz"] = settings.PI_IMGSZ[model_key]
    return cfg["imgsz"]


def test_catalog_imgsz_matches_worker_on_ncnn():
    """On the Pi (ncnn), Worker.__init__ overrides shuttle/landed/pose to
    PI_IMGSZ (640) - the catalog must report that, not the x86 config value
    (1280), or the Cameras page misreports what is actually running."""
    catalog = {m["key"]: m for m in pipeline.model_catalog(backend="ncnn")}
    for key, cfg in pipeline.MODELS.items():
        if key == "none":
            continue
        assert catalog[key]["imgsz"] == _worker_imgsz(key, "ncnn")


def test_catalog_imgsz_matches_worker_on_openvino():
    """On x86 (openvino), no PI_IMGSZ override applies - the catalog must
    keep reporting each model's configured imgsz unchanged."""
    catalog = {m["key"]: m for m in pipeline.model_catalog(backend="openvino")}
    for key, cfg in pipeline.MODELS.items():
        if key == "none":
            continue
        assert catalog[key]["imgsz"] == cfg["imgsz"]
        assert catalog[key]["imgsz"] == _worker_imgsz(key, "openvino")
