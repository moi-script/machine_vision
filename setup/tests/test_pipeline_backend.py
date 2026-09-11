import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import pipeline  # noqa: E402


# ── backend selection ────────────────────────────────────────

def test_x86_windows_gets_openvino():
    assert pipeline.resolve_backend("AMD64") == "openvino"


def test_x86_linux_gets_openvino():
    assert pipeline.resolve_backend("x86_64") == "openvino"


def test_pi_gets_ncnn():
    assert pipeline.resolve_backend("aarch64") == "ncnn"


def test_unknown_arch_gets_ncnn():
    """Fail toward the portable backend, not the Intel-only one."""
    assert pipeline.resolve_backend("armv7l") == "ncnn"


# ── export cache directory ───────────────────────────────────

def test_export_dir_includes_backend_and_size():
    got = pipeline._export_dir("models/yolov8n-pose.pt", 640, "ncnn")
    assert got == "models/yolov8n-pose_640_ncnn_model"


def test_export_dir_separates_backends():
    """An OpenVINO export must never be mistaken for an NCNN one."""
    ov = pipeline._export_dir("models/pose.pt", 640, "openvino")
    nc = pipeline._export_dir("models/pose.pt", 640, "ncnn")
    assert ov != nc


def test_export_dir_separates_sizes():
    """Reusing a 640 export at 1280 silently runs at 640."""
    assert pipeline._export_dir("m.pt", 640, "ncnn") != \
           pipeline._export_dir("m.pt", 1280, "ncnn")
