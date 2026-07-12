# ============================================================
# roboflow_stream.py — real-time Roboflow Workflow streaming (WebRTC)
#
# Cloud-hosted GPU streaming of the shuttlecock workflow over WebRTC, for a
# live webcam, an RTSP stream, or a video file. This is Roboflow's real-time
# path; the serverless /workflows REST client in roboflow_client.py is for
# single stills.
#
# ⚠️  REQUIREMENTS (this will NOT run on the repo's default Python 3.13)
#   - inference-sdk supports Python 3.10–3.12 only. Create a compatible env:
#         py -3.12 -m venv .venv-stream
#         .venv-stream\Scripts\activate
#         pip install -U "inference-sdk[webrtc]"
#   - The Roboflow Workflow itself must compile (fix the `model_id` binding in
#     the Workflow editor first — otherwise streaming 500s like the REST call).
#
# API key is read from ROBOFLOW_API_KEY (env var or the gitignored .env).
# NEVER hard-code it (the Roboflow UI snippets inline it — we don't).
#
# Run:
#   python -m utils.roboflow_stream --source webcam
#   python -m utils.roboflow_stream --source rtsp  --url rtsp://host:8554/stream
#   python -m utils.roboflow_stream --source video --path clip.mp4 [--save out.mp4]
# ============================================================

from __future__ import annotations

import argparse
import os
from typing import Any

# Reuse the workflow coordinates + .env loader from the REST client so both
# paths agree on one source of truth.
from utils.roboflow_client import (
    API_URL,
    IMAGE_INPUT_NAME,
    WORKFLOW_ID,
    WORKSPACE_NAME,
    RoboflowConfigError,
    _load_dotenv,
)

# Defaults for the hosted WebRTC session (match Roboflow's UI snippet options).
DEFAULT_PLAN = "webrtc-gpu-medium"   # webrtc-gpu-small | -medium | -large
DEFAULT_REGION = "us"                # us | eu | ap
DEFAULT_TIMEOUT = 3600               # seconds the hosted session may run


def _get_api_key(api_key: str | None) -> str:
    _load_dotenv()
    api_key = api_key or os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise RoboflowConfigError(
            "No Roboflow API key found. Set ROBOFLOW_API_KEY (env or .env)."
        )
    return api_key


def _import_sdk():
    """Import inference-sdk lazily with a clear message if it's missing."""
    try:
        from inference_sdk import InferenceHTTPClient
        from inference_sdk.webrtc import (
            RTSPSource,
            StreamConfig,
            VideoFileSource,
            VideoMetadata,
            WebcamSource,
        )
    except ImportError as exc:  # pragma: no cover - environment guard
        raise RoboflowConfigError(
            "inference-sdk (with the webrtc extra) is not installed, or you are "
            "on Python 3.13 (unsupported). Use Python 3.10–3.12 and run:\n"
            '    pip install -U "inference-sdk[webrtc]"'
        ) from exc
    return (
        InferenceHTTPClient,
        WebcamSource,
        RTSPSource,
        VideoFileSource,
        StreamConfig,
        VideoMetadata,
    )


def _summarize_data(data: dict[str, Any]) -> dict[str, Any]:
    """Log-safe view of a per-frame data payload — never dump base64 blobs."""
    out: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, dict) and isinstance(value.get("value"), str) and len(value["value"]) > 256:
            out[key] = f"<image base64, {len(value['value'])} chars>"
        elif isinstance(value, list):
            out[key] = f"<list, {len(value)} items>"
        else:
            out[key] = value
    return out


def stream_workflow(
    source_kind: str,
    *,
    url: str | None = None,
    path: str | None = None,
    api_key: str | None = None,
    resolution: tuple[int, int] = (1280, 720),
    plan: str = DEFAULT_PLAN,
    region: str = DEFAULT_REGION,
    processing_timeout: int = DEFAULT_TIMEOUT,
    realtime: bool = True,
    list_outputs: bool = False,
) -> None:
    """
    Start a WebRTC streaming session that runs the shuttlecock workflow on a
    live source. Blocks until the window is closed (q) or the source ends.

    Args:
        source_kind: "webcam", "rtsp", or "video".
        url:         RTSP URL (required for source_kind="rtsp").
        path:        Video file path (required for source_kind="video").
        api_key:     Roboflow key; falls back to $ROBOFLOW_API_KEY / .env.
        resolution:  Webcam capture resolution.
        plan/region: Hosted WebRTC GPU plan and region.
        realtime:    For video files, False buffers and processes every frame.
        list_outputs: If True, print the workflow's real data-channel output
                     keys from the first frame, then close the session. Use
                     this to discover the exact keys to wire scoring into.
    """
    import cv2  # local import: only needed at run time

    (
        InferenceHTTPClient,
        WebcamSource,
        RTSPSource,
        VideoFileSource,
        StreamConfig,
        VideoMetadata,
    ) = _import_sdk()

    api_key = _get_api_key(api_key)
    client = InferenceHTTPClient.init(api_url=API_URL, api_key=api_key)

    if source_kind == "webcam":
        source = WebcamSource(resolution=resolution)
    elif source_kind == "rtsp":
        if not url:
            raise RoboflowConfigError("--url is required for source=rtsp")
        source = RTSPSource(url)
    elif source_kind == "video":
        if not path:
            raise RoboflowConfigError("--path is required for source=video")
        source = VideoFileSource(path, realtime_processing=realtime)
    else:
        raise RoboflowConfigError(f"Unknown source kind: {source_kind!r}")

    config = StreamConfig(
        processing_timeout=processing_timeout,
        requested_plan=plan,
        requested_region=region,
    )

    session = client.webrtc.stream(
        source=source,
        workflow=WORKFLOW_ID,
        workspace=WORKSPACE_NAME,
        image_input=IMAGE_INPUT_NAME,
        config=config,
    )

    @session.on_frame
    def _show_frame(frame, metadata):  # noqa: ANN001 - SDK-provided types
        cv2.imshow("Roboflow Workflow Output", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            session.close()

    printed_keys = {"done": False}

    @session.on_data()
    def _on_data(data: dict, metadata: "VideoMetadata"):  # noqa: F821
        if list_outputs:
            # Print the real output keys once, then stop — a discovery run.
            if not printed_keys["done"]:
                printed_keys["done"] = True
                print("[STREAM] workflow data-channel output keys:",
                      sorted(data.keys()))
                print("[STREAM] first-frame summary:", _summarize_data(data))
                session.close()
            return
        # Keep it small: log a summary, not raw blobs. Hook your shuttle
        # tracking / scoring here using the workflow's real output keys.
        print(f"[STREAM] frame {getattr(metadata, 'frame_id', '?')}:",
              _summarize_data(data))

    mode = "output-discovery" if list_outputs else "streaming"
    print(f"[STREAM] starting {source_kind} session ({mode}, "
          f"plan={plan}, region={region}). Press 'q' in the window to stop.")
    session.run()
    cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stream a live source through the Roboflow shuttlecock workflow (WebRTC)."
    )
    parser.add_argument("--source", required=True, choices=["webcam", "rtsp", "video"])
    parser.add_argument("--url", help="RTSP URL (for --source rtsp)")
    parser.add_argument("--path", help="Video file path (for --source video)")
    parser.add_argument("--plan", default=DEFAULT_PLAN)
    parser.add_argument("--region", default=DEFAULT_REGION)
    parser.add_argument("--buffer", action="store_true",
                        help="For video: buffer and process every frame (not real-time)")
    parser.add_argument("--list-outputs", action="store_true",
                        help="Print the workflow's real data-channel output keys "
                             "from the first frame, then exit")
    args = parser.parse_args()

    stream_workflow(
        args.source,
        url=args.url,
        path=args.path,
        plan=args.plan,
        region=args.region,
        realtime=not args.buffer,
        list_outputs=args.list_outputs,
    )


if __name__ == "__main__":
    main()
