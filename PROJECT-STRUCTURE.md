# Project Structure — `C:\thesis`

Complete hierarchy of the AeroSense / Badminton Feeder Trainer thesis project, from the
repository root down to the last child. Three parts:

1. [Full tree](#1-full-tree) — every folder and file in one block
2. [Folder reference](#2-folder-reference) — what each directory is for
3. [File reference](#3-file-reference) — what each file is for

Bulk image/label directories are shown with a file count instead of 37,000 individual
filenames; everything else is listed literally. `.git/` internals and the ~10,000 files
inside `setup/.venv-stream/` (a stock Python virtualenv) are summarised for the same reason.

---

## 1. Full tree

```text
C:\thesis\
├── .gitattributes
├── .gitignore
├── README.md
├── PROJECT-STRUCTURE.md                  (this file)
│
├── .git/                                 [git internals — not expanded]
│
├── .superpowers/
│   └── sdd/
│       ├── .gitignore
│       ├── progress.md
│       ├── review-51c8139..9cc58d5.diff
│       ├── review-78ba30e..9c3c126.diff
│       ├── review-78c9ebe..5ab2912.diff
│       ├── review-9cc58d5..a5aca6c.diff
│       ├── review-a1d8214..51c8139.diff
│       ├── review-a1d8214..c28f7da.diff
│       ├── review-a5aca6c..c28f7da.diff
│       ├── review-abe3b68..549fab9.diff
│       ├── task-1-brief.md
│       ├── task-1-report.md
│       ├── task-2-brief.md
│       ├── task-2-report.md
│       ├── task-3-brief.md
│       ├── task-3-report.md
│       ├── task-4-brief.md
│       ├── task-4-report.md
│       └── 2026-08-20-feeder-court/
│           ├── final-fix-report.md
│           ├── progress.md
│           ├── review-4e38fef..d964189.diff
│           ├── review-5ea85a3..77ec62a.diff
│           ├── review-64a0e5e..3d419c9.diff
│           ├── review-77ec62a..85c8bd7.diff
│           ├── review-85c8bd7..64a0e5e.diff
│           ├── review-c9d60d7..4e38fef.diff
│           ├── review-d964189..5ea85a3.diff
│           ├── review-e3e8f34..64a0e5e.diff
│           ├── task-1-brief.md
│           ├── task-1-report.md
│           ├── task-2-brief.md
│           ├── task-2-report.md
│           ├── task-3-brief.md
│           └── task-3-report.md
│
├── camera_OV9281/
│   ├── 0.3.py
│   ├── guide.txt
│   ├── lags_release.py
│   ├── live.py
│   ├── pixels.txt
│   ├── save_vid.py
│   └── singe_img.py
│
├── docs/
│   └── superpowers/
│       ├── plans/
│       │   ├── 2026-07-22-player-skill-ranking.md
│       │   └── 2026-08-20-feeder-court.md
│       └── specs/
│           ├── 2026-07-22-player-skill-ranking-design.md
│           └── 2026-08-20-feeder-court-design.md
│
├── notebooks/
│   └── feeder_court_train.ipynb
│
├── pipeline/
│   ├── guide.txt
│   ├── multi.py
│   ├── realtime_yolo_vision_pipeline.svg
│   └── single.py
│
├── roboflow/
│   └── main.py
│
├── yolo/
│   └── main.py
│
└── setup/                                ← the real application lives here
    ├── .env
    ├── .gitignore
    ├── calibrate.py
    ├── fetch_face_models.py
    ├── main.py
    ├── README.md
    ├── requirements.txt
    ├── run_server.py
    ├── runs_train.log
    ├── runs_train_resume.err
    ├── runs_train_resume.log
    ├── seed.py
    ├── yolov8n.pt
    ├── yolov8n-pose.pt
    │
    ├── .pytest_cache/
    │   ├── .gitignore
    │   ├── CACHEDIR.TAG
    │   ├── README.md
    │   └── v/
    │       └── cache/
    │           ├── lastfailed
    │           └── nodeids
    │
    ├── .venv-stream/                     [virtualenv — ~10,000 files]
    │   ├── pyvenv.cfg
    │   ├── Include/
    │   ├── Lib/                          [site-packages]
    │   ├── Scripts/
    │   └── share/
    │
    ├── __pycache__/
    │   ├── calibrate.cpython-313.pyc
    │   ├── main.cpython-313.pyc
    │   └── seed.cpython-313.pyc
    │
    ├── app/
    │   ├── __init__.py
    │   ├── db.py
    │   ├── engine.py
    │   ├── esp32_camera_client.py
    │   ├── events.py
    │   ├── face.py
    │   ├── models.py
    │   ├── server.py
    │   ├── streamer.py
    │   ├── __pycache__/                  [9 .pyc files]
    │   └── routers/
    │       ├── __init__.py
    │       ├── control.py
    │       ├── esp32_enroll.py
    │       ├── players.py
    │       ├── sessions.py
    │       ├── settings.py
    │       └── __pycache__/              [6 .pyc files]
    │
    ├── config/
    │   ├── settings.py
    │   └── __pycache__/
    │       └── settings.cpython-313.pyc
    │
    ├── docs/
    │   ├── feeder-court-distance-test.md
    │   ├── scope-and-limitations.md
    │   ├── shuttle-v2-dataset.md
    │   └── superpowers/
    │       ├── plans/
    │       │   └── 2026-07-13-court-space-homography-geometry.md
    │       └── specs/
    │           └── 2026-07-12-court-space-homography-geometry-design.md
    │
    ├── datasets/
    │   ├── feeder_court_yolo.zip
    │   ├── smashspeed-fetch.err
    │   ├── smashspeed-fetch.log
    │   │
    │   ├── feeder_court/
    │   │   ├── segments.json
    │   │   ├── calibration/              [60 sampled frames]
    │   │   └── proposals/
    │   │       ├── far.json
    │   │       ├── mid.json
    │   │       ├── near.json
    │   │       └── ...                   [9 files total]
    │   │
    │   ├── feeder_court_yolo/
    │   │   ├── data.yaml
    │   │   ├── train/
    │   │   │   ├── labels.cache
    │   │   │   ├── images/               [41 files]
    │   │   │   └── labels/               [41 files]
    │   │   └── val/
    │   │       ├── labels.cache
    │   │       ├── images/               [12 files]
    │   │       └── labels/               [12 files]
    │   │
    │   ├── feeder_court-calibration/
    │   │   ├── data.yaml
    │   │   └── train/
    │   │       ├── images/               [53 files]
    │   │       └── labels/               [53 files]
    │   │
    │   ├── scene-dummy/
    │   │   ├── raw/
    │   │   │   ├── images/               [442 files]
    │   │   │   └── labels/               [442 files]
    │   │   ├── raw2/
    │   │   │   ├── images/               [174 files]
    │   │   │   └── labels/               [174 files]
    │   │   ├── random/
    │   │   │   ├── data.yaml
    │   │   │   ├── train/{images,labels}/     [353 each]
    │   │   │   └── valid/{images,labels}/     [89 each]
    │   │   ├── temporal/
    │   │   │   ├── data.yaml
    │   │   │   ├── train/{images,labels}/     [309 each]
    │   │   │   └── valid/{images,labels}/     [133 each]
    │   │   ├── combined/
    │   │   │   ├── data.yaml
    │   │   │   ├── train/{images,labels}/     [654 each]
    │   │   │   └── valid/{images,labels}/     [92 each]
    │   │   └── roboflow_upload/
    │   │       ├── images/               [423 files]
    │   │       └── labels/               [424 files]
    │   │
    │   ├── shuttlecock-1/
    │   │   ├── data.yaml
    │   │   ├── train/{images,labels}/    [392 each] + labels.cache
    │   │   ├── valid/{images,labels}/    [20 each]  + labels.cache
    │   │   └── test/{images,labels}/     [10 each]
    │   │
    │   ├── shuttle-v2/
    │   │   ├── data.yaml
    │   │   ├── train/{images,labels}/    [466 each] + labels.cache
    │   │   ├── valid/{images,labels}/    [47 each]  + labels.cache
    │   │   └── test/{images,labels}/     [24 each]
    │   │
    │   ├── smashspeed-v8/
    │   │   ├── data.yaml
    │   │   ├── data_selected.yaml
    │   │   ├── train/{images,labels}/    [9,931 each] + labels.cache
    │   │   ├── valid/{images,labels}/    [2,165 each] + labels.cache
    │   │   └── test/{images,labels}/     [1,068 each]
    │   │
    │   ├── smashspeed-v8-fast/
    │   │   ├── data.yaml
    │   │   ├── train/{images,labels}/    [1,200 each] + labels.cache
    │   │   └── valid/{images,labels}/    [400 each]   + labels.cache
    │   │
    │   └── vid_source/
    │       ├── angle_1.mp4
    │       ├── angle_2.mp4
    │       ├── angle_3.mp4
    │       ├── Angle_4.mp4
    │       ├── shuttlecoc_testing_camera_live_2.mp4
    │       └── new_badminton_source/
    │           ├── far.mp4
    │           ├── line_1.mp4
    │           ├── line_2.mp4
    │           ├── mid.mp4
    │           └── near.mp4
    │
    ├── models/
    │   ├── face_detection_yunet_2023mar.onnx
    │   ├── face_recognition_sface_2021dec.onnx
    │   ├── shuttlecock.pt
    │   ├── shuttlecock_scene_dummy.pt
    │   ├── shuttlecock_scene_v2.pt
    │   ├── shuttlecock_v2.pt
    │   ├── shuttlecock_v3.pt
    │   ├── shuttlecock_v3_smoke15.pt
    │   ├── shuttlecock_v2_640_openvino_model/
    │   │   ├── metadata.yaml
    │   │   ├── shuttlecock_v2.bin
    │   │   └── shuttlecock_v2.xml
    │   ├── shuttlecock_v2_960_openvino_model/
    │   │   ├── metadata.yaml
    │   │   ├── shuttlecock_v2.bin
    │   │   └── shuttlecock_v2.xml
    │   └── shuttlecock_v3_640_openvino_model/
    │       ├── metadata.yaml
    │       ├── shuttlecock_v3.bin
    │       └── shuttlecock_v3.xml
    │
    ├── runs/
    │   ├── feeder_court_train.log
    │   ├── labelimg.log
    │   ├── scene_dummy_combined.log
    │   ├── scene_dummy_train.log
    │   ├── shuttle_motion.csv
    │   ├── shuttle_motion_out.mp4
    │   ├── shuttle_test.mp4
    │   ├── shuttlecock_testing_camera_live.mp4
    │   ├── train.pid
    │   ├── train_v3.pid
    │   ├── train_v3_full.err
    │   ├── train_v3_full.log
    │   ├── train_v3_full.pid
    │   ├── train_v3_smoke.err
    │   ├── train_v3_smoke.log
    │   ├── webcam_test.err
    │   ├── webcam_test.log
    │   ├── webcam_v3.pid
    │   ├── webcam_v3only.err
    │   ├── webcam_v3only.log
    │   │
    │   ├── feeder_court/
    │   │   ├── args.yaml
    │   │   ├── best_1280_openvino_model/
    │   │   │   ├── metadata.yaml
    │   │   │   ├── best.bin
    │   │   │   └── best.xml
    │   │   ├── detect/
    │   │   │   ├── far_boxed.mp4
    │   │   │   ├── mid_boxed.mp4
    │   │   │   └── near_boxed.mp4
    │   │   └── p2-native/
    │   │       ├── args.yaml
    │   │       ├── results.csv
    │   │       ├── results.png
    │   │       ├── labels.jpg
    │   │       ├── confusion_matrix.png
    │   │       ├── confusion_matrix_normalized.png
    │   │       ├── BoxF1_curve.png
    │   │       ├── BoxP_curve.png
    │   │       ├── BoxR_curve.png
    │   │       ├── BoxPR_curve.png
    │   │       ├── train_batch0.jpg … train_batch3837.jpg
    │   │       └── weights/
    │   │           ├── best.pt
    │   │           └── last.pt
    │   │
    │   ├── scene_dummy/
    │   │   ├── combined960/               (+ weights/{best.pt,last.pt})
    │   │   ├── random/                    (+ weights/{best.pt,last.pt})
    │   │   └── temporal/
    │   │       └── weights/
    │   │           ├── best.pt
    │   │           ├── last.pt
    │   │           ├── best_416_openvino_model/   [metadata.yaml, best.bin, best.xml]
    │   │           ├── best_640_openvino_model/   [metadata.yaml, best.bin, best.xml]
    │   │           └── best_1280_openvino_model/  [metadata.yaml, best.bin, best.xml]
    │   │
    │   ├── shuttle/
    │   │   └── yolov8n-640/
    │   │       ├── args.yaml, results.csv, results.png, curves, batch mosaics
    │   │       └── weights/{best.pt, last.pt}
    │   │
    │   └── shuttle_v2/
    │       ├── yolov8n-640-v2/            (+ weights/{best.pt,last.pt})
    │       ├── yolov8n-640-v3-full/       (+ weights/{best.pt,last.pt})
    │       └── yolov8n-640-v3-smashspeed/ (+ weights/{best.pt,last.pt})
    │
    ├── scripts/
    │   ├── audit_dataset.py
    │   ├── autolabel_scene_video.py
    │   ├── build_combined_scene.py
    │   ├── fetch_shuttle_v2.py
    │   ├── prepare_fast_subset.py
    │   ├── prepare_for_roboflow.py
    │   ├── prune_leaky_eval.py
    │   ├── resplit_by_clip.py
    │   ├── select_training_subset.py
    │   ├── shuttle_motion_prototype.py
    │   ├── split_scene_dummy.py
    │   ├── test_shuttle_v2_webcam.py
    │   ├── train_scene_dummy.py
    │   ├── train_shuttlecock.py
    │   ├── train_shuttlecock_v2.py
    │   ├── train_status.py
    │   ├── __pycache__/                   [4 .pyc files]
    │   └── feeder_court/
    │       ├── detect_video.py
    │       ├── live_detect.py
    │       ├── propose_shuttles.py
    │       ├── sample_calibration.py
    │       ├── segment_clips.py
    │       ├── train_local.py
    │       └── __pycache__/               [2 .pyc files]
    │
    ├── tests/
    │   ├── test_control_api.py
    │   ├── test_control_arm.py
    │   ├── test_db.py
    │   ├── test_engine.py
    │   ├── test_engine_persist.py
    │   ├── test_engine_reco.py
    │   ├── test_engine_skill.py
    │   ├── test_events.py
    │   ├── test_face.py
    │   ├── test_face_enroll.py
    │   ├── test_feeder_court_motion.py
    │   ├── test_feeder_court_roi.py
    │   ├── test_feeder_court_segments.py
    │   ├── test_feeder_court_sizes.py
    │   ├── test_feeder_court_tracks.py
    │   ├── test_health.py
    │   ├── test_models.py
    │   ├── test_players_api.py
    │   ├── test_pose_features.py
    │   ├── test_roboflow_client.py
    │   ├── test_seed.py
    │   ├── test_sessions_api.py
    │   ├── test_settings_api.py
    │   ├── test_shuttle_worker.py
    │   ├── test_skill_profile.py
    │   ├── test_streamer.py
    │   ├── test_ws.py
    │   ├── test_zones.py
    │   └── __pycache__/                   [30 .pyc files]
    │
    ├── utils/
    │   ├── display.py
    │   ├── pose_features.py
    │   ├── roboflow_client.py
    │   ├── roboflow_stream.py
    │   ├── scoring.py
    │   ├── shuttle_worker.py
    │   ├── skill_profile.py
    │   ├── zones.py
    │   ├── __pycache__/                   [10 .pyc files]
    │   └── feeder_court/
    │       ├── __init__.py
    │       ├── motion.py
    │       ├── roi.py
    │       ├── segments.py
    │       ├── sizes.py
    │       ├── tracks.py
    │       └── __pycache__/               [6 .pyc files]
    │
    └── yolov8n-pose_openvino_model/
        ├── metadata.yaml
        ├── yolov8n-pose.bin
        └── yolov8n-pose.xml
```

---

## 2. Folder reference

### Repository root

| Folder | What it is |
|---|---|
| `C:\thesis\` | The repository root; it holds the finished application in `setup/` plus a set of older standalone experiment folders kept as a record of how the camera and detection stack were arrived at. |
| `.git/` | Git's internal object and ref store, including the Git LFS pointers used to version the large `.pt` and `.onnx` model weights. |
| `.superpowers/` | Scratch workspace for the subagent-driven development workflow; it is gitignored and holds nothing the application needs at runtime. |
| `.superpowers/sdd/` | Per-task briefs, completion reports and review diffs from the first SDD run (player skill ranking), written so each implementation step could be reviewed independently. |
| `.superpowers/sdd/2026-08-20-feeder-court/` | The same brief/report/review bundle for the second SDD run, the `feeder_court` shuttlecock detector, including the final fix report that closed the branch out. |
| `camera_OV9281/` | Early throwaway scripts for bringing up the OV9281 global-shutter USB camera, kept because they document the exact resolution and FPS combinations that proved stable on this hardware. |
| `docs/` | Repository-level design documentation, as opposed to the implementation notes that live under `setup/docs/`. |
| `docs/superpowers/plans/` | Step-by-step implementation plans, one per major feature, each written before any code was touched so progress could be ticked off against it. |
| `docs/superpowers/specs/` | The design specs those plans were derived from, describing the problem, the chosen approach and the rejected alternatives. |
| `notebooks/` | Jupyter notebooks meant to be run on external GPU hardware rather than on the development laptop. |
| `pipeline/` | A pair of prototype YOLO capture-and-inference loops, single-threaded and multi-threaded, written to measure how much throughput threading actually buys on this CPU. |
| `roboflow/` | A minimal one-file demo of running a Roboflow-exported model against a webcam, predating the proper client in `setup/utils/`. |
| `yolo/` | The equivalent minimal demo for a locally trained Ultralytics model, kept as the simplest possible reference of the inference call. |
| `setup/` | The actual thesis system: the FastAPI backend, the drill engine, the vision utilities, the training scripts, the datasets and the trained weights. |

### Inside `setup/`

| Folder | What it is |
|---|---|
| `setup/.pytest_cache/` | Pytest's own cache of last-failed and collected test node IDs, regenerated on every run and safe to delete at any time. |
| `setup/.venv-stream/` | A dedicated Python virtual environment for the streaming and inference dependencies, isolated so a heavy install cannot break the main environment. |
| `setup/.venv-stream/Lib/` | The site-packages tree of that virtualenv, which is where the roughly 10,000 vendored dependency files live. |
| `setup/.venv-stream/Scripts/` | The virtualenv's Windows entry points, including its own `python.exe` and `pip.exe`. |
| `setup/.venv-stream/Include/`, `share/` | The virtualenv's C header and shared-data directories, created by `venv` and effectively unused here. |
| `setup/__pycache__/` | Compiled bytecode for the top-level modules, produced automatically by the interpreter and gitignored. |
| `setup/app/` | The FastAPI backend package: database access, the drill engine, face recognition, WebSocket events and the MJPEG stream buffer. |
| `setup/app/routers/` | One FastAPI router module per API surface, so player, session, settings, engine-control and ESP32 enrollment endpoints stay separately testable. |
| `setup/config/` | Central configuration for the vision system, kept apart from `app/` so training and prototype scripts can import thresholds without pulling in the web stack. |
| `setup/docs/` | Implementation-level notes: what the system does not claim to do, why each dataset generation exists, and how to reproduce the distance tests. |
| `setup/docs/superpowers/plans/` | The implementation plan for the court-space homography work, which is the geometry layer the zone scoring depends on. |
| `setup/docs/superpowers/specs/` | The design spec behind that homography plan, including why image-space zone boxes were abandoned. |
| `setup/datasets/` | Every YOLO dataset generation used in the thesis, plus the raw source video they were cut from; entirely gitignored because it is large and reproducible. |
| `setup/datasets/feeder_court/` | Intermediate artefacts of the feeder_court labelling pipeline — the usable-window segment index, sampled calibration frames and per-clip auto-label proposals. |
| `setup/datasets/feeder_court/calibration/` | The 60 frames sampled across the three distance clips so a human could hand-box them and establish real shuttle pixel sizes. |
| `setup/datasets/feeder_court/proposals/` | Per-clip JSON proposal files listing which frames the motion tracker believes contain a shuttle flight, split into positive, negative and uncertain buckets. |
| `setup/datasets/feeder_court_yolo/` | The train/val dataset assembled from the hand-corrected feeder_court labels, in the exact directory layout Ultralytics expects. |
| `setup/datasets/feeder_court-calibration/` | The 53 hand-boxed calibration frames packaged as a trainable dataset, which is what the small local CPU training run uses. |
| `setup/datasets/scene-dummy/` | The single-clip scene-specific experiment that measured how badly frame-level splitting inflates validation scores. |
| `setup/datasets/scene-dummy/raw/` | Recording session 1 — close range, grey wall, olive shirt — whose labels were verified by hand. |
| `setup/datasets/scene-dummy/raw2/` | Recording session 2 — varied distance, white wall, black shirt — held separately because its labels are mostly unverified model pre-labels. |
| `setup/datasets/scene-dummy/random/` | The deliberately wrong split, where frames were shuffled before splitting, included so its inflated metrics can be cited as a counter-example. |
| `setup/datasets/scene-dummy/temporal/` | The correct split of the same frames, holding out the tail of the clip's timeline so no validation frame has a near-duplicate in training. |
| `setup/datasets/scene-dummy/combined/` | Both recording sessions merged into one training set, which is the dataset the `shuttlecock_scene_v2` weights were actually trained on. |
| `setup/datasets/scene-dummy/roboflow_upload/` | Frames plus pre-labels staged for upload to Roboflow, so a human could correct the boxes in a proper annotation UI. |
| `setup/datasets/shuttlecock-1/` | The v1 shuttlecock dataset, 422 frames from a single clip, retained as evidence of the leakage problem rather than for training. |
| `setup/datasets/shuttle-v2/` | The v2 dataset that replaced it; it turned out to be close-up grayscale photos rather than court footage, which is why the v2 weights fire on blank walls. |
| `setup/datasets/smashspeed-v8/` | The large public smashspeed dataset, roughly 13,000 labelled frames of real court footage, and the first data that produced detections at match distance. |
| `setup/datasets/smashspeed-v8-fast/` | A pre-resized 1,600-image subset of it, selected by shuttle pixel size, so CPU training is not bottlenecked on decoding 1080p JPEGs. |
| `setup/datasets/vid_source/` | The raw camera recordings every dataset above was cut from, including the distance clips shot specifically for the feeder_court work. |
| `setup/datasets/vid_source/new_badminton_source/` | The five-clip distance series recorded on the OV9281 rig at fixed near, mid and far camera positions, plus two line-of-play angles. |
| `setup/models/` | The deployed weights the running application loads, tracked in Git LFS, as distinct from the raw training outputs under `runs/`. |
| `setup/models/shuttlecock_v2_640_openvino_model/` | The OpenVINO IR export of the v2 weights at input size 640, which is the fast default path on this laptop. |
| `setup/models/shuttlecock_v2_960_openvino_model/` | The same weights exported at 960, trading speed for the extra pixels that distant shuttles need. |
| `setup/models/shuttlecock_v3_640_openvino_model/` | The OpenVINO export of the v3 smashspeed-trained weights at 640, roughly 7x faster than PyTorch and verified numerically equivalent. |
| `setup/runs/` | Every Ultralytics training output plus the console logs, PID files and annotated test videos produced along the way. |
| `setup/runs/feeder_court/` | Outputs of the feeder_court detector: the Colab training run, its OpenVINO export and the annotated near/mid/far detection videos. |
| `setup/runs/feeder_court/best_1280_openvino_model/` | The IR export of that run's best checkpoint at input size 1280, the size the p2 head was trained at. |
| `setup/runs/feeder_court/detect/` | The three annotated output videos with detection boxes burned in, which are the artefact the distance test is judged from. |
| `setup/runs/feeder_court/p2-native/` | The actual yolov8-p2 training run directory, holding the metrics CSV, the curve plots, the batch mosaics and the checkpoints. |
| `setup/runs/scene_dummy/` | Training runs for the scene-specific experiment, one directory per split so the random-versus-temporal gap is directly comparable. |
| `setup/runs/scene_dummy/temporal/weights/` | The temporal-split checkpoints plus their three OpenVINO exports at 416, 640 and 1280, which is where the speed-versus-distance trade-off was measured. |
| `setup/runs/shuttle/` | The v1 shuttlecock training run, whose headline mAP50 of 0.898 is now known to be memorisation of a single clip. |
| `setup/runs/shuttle_v2/` | The v2 and v3 training runs, including the smashspeed subset run that first achieved detection at court distance. |
| `setup/runs/**/weights/` | The per-run checkpoint pair — best-epoch and last-epoch weights — that promotion into `models/` copies from. |
| `setup/scripts/` | One-shot command-line tools for dataset preparation, training, auditing and live smoke testing; nothing here is imported by the server. |
| `setup/scripts/feeder_court/` | The feeder_court pipeline end to end, from clip segmentation through label proposal to training and live detection. |
| `setup/tests/` | The pytest suite covering the API, the engine, the persistence layer and every pure vision helper. |
| `setup/utils/` | Importable vision and scoring helpers shared by `main.py`, the drill engine and the scripts. |
| `setup/utils/feeder_court/` | The pure, unit-tested core of the motion-based shuttle detector: segmentation, ROI cropping, candidate extraction, size gating and tracking. |
| `setup/yolov8n-pose_openvino_model/` | The OpenVINO export of the stock pose model, which is what the live drill loop actually runs for player keypoints. |

---

## 3. File reference

### Root files

| File | What it does |
|---|---|
| `.gitattributes` | Routes `setup/models/*.pt` and `*.onnx` through Git LFS so multi-megabyte weights are stored as pointers instead of bloating the repository. |
| `.gitignore` | Excludes the `.superpowers/` scratch workspace from version control. |
| `README.md` | A one-line placeholder at the repository root; the substantive documentation is `setup/README.md`. |
| `PROJECT-STRUCTURE.md` | This document — the full directory hierarchy plus a description of every folder and file. |

### `.superpowers/sdd/`

| File | What it does |
|---|---|
| `.gitignore` | Keeps the entire scratch directory out of git while still allowing it to exist on disk. |
| `progress.md` | The running checklist for an SDD run, recording which plan tasks are done and which are still open. |
| `task-N-brief.md` | The self-contained instruction handed to a fresh subagent for task N, written so it needs no prior conversation context. |
| `task-N-report.md` | That subagent's completion report — what it changed, what it verified and what it deliberately left alone. |
| `review-<sha>..<sha>.diff` | A captured diff between two commits, saved so a reviewing agent could examine exactly one task's changes in isolation. |
| `2026-08-20-feeder-court/final-fix-report.md` | The closing report for the feeder_court branch, summarising the fixes applied after the last review round. |

### `camera_OV9281/`

| File | What it does |
|---|---|
| `0.3.py` | Opens the camera at 640x480 and 60 FPS, the combination that proved most stable during bring-up. |
| `guide.txt` | Working notes on driver quirks and which settings did and did not work with this camera. |
| `lags_release.py` | A commented-out experiment in fixing capture lag via the DirectShow backend and the MJPG fourcc on Windows. |
| `live.py` | The minimal live-preview loop, used to confirm the camera enumerates and streams at all. |
| `pixels.txt` | Recorded pixel measurements taken off still frames, used to seed the hardcoded geometry in `config/settings.py`. |
| `save_vid.py` | Records 1280x720 / 60 FPS video to disk; this is the script the source clips were captured with. |
| `singe_img.py` | Grabs a single frame and writes it to `snapshot.png`, for measuring court coordinates by hand. |

### `docs/`

| File | What it does |
|---|---|
| `superpowers/plans/2026-07-22-player-skill-ranking.md` | The implementation plan for per-player skill accumulation and the rule-based rubric. |
| `superpowers/plans/2026-08-20-feeder-court.md` | The implementation plan for the feeder_court shuttlecock detector, broken into individually reviewable tasks. |
| `superpowers/specs/2026-07-22-player-skill-ranking-design.md` | The design spec behind that ranking work, defining the metrics and how session stats merge into a career profile. |
| `superpowers/specs/2026-08-20-feeder-court-design.md` | The design spec for the detector, including the decision to detect by motion rather than by appearance. |

### `notebooks/`, `pipeline/`, `roboflow/`, `yolo/`

| File | What it does |
|---|---|
| `notebooks/feeder_court_train.ipynb` | The Colab notebook that trains the yolov8n-p2 detector on a T4, since the same run takes roughly 30 hours on this laptop's CPU. |
| `pipeline/guide.txt` | Notes comparing the single- and multi-threaded prototypes and the FPS each achieved. |
| `pipeline/single.py` | A straight-line capture-detect-draw loop, serving as the throughput baseline. |
| `pipeline/multi.py` | The same loop with capture and inference split across threads and joined by a queue, so camera I/O overlaps with inference. |
| `pipeline/realtime_yolo_vision_pipeline.svg` | A diagram of that real-time pipeline, drawn for the thesis write-up. |
| `roboflow/main.py` | A minimal webcam loop against a Roboflow-exported `best.pt`, the earliest end-to-end detection demo. |
| `yolo/main.py` | The same minimal loop for a locally trained Ultralytics model, kept as the reference form of the inference call. |

### `setup/` top level

| File | What it does |
|---|---|
| `.env` | Local secrets, chiefly the Roboflow API key and the Mongo connection string; gitignored and never committed. |
| `.gitignore` | Excludes secrets, stray weights, OpenVINO exports, datasets, training runs, caches and virtualenvs, while whitelisting the tracked `models/` weights. |
| `calibrate.py` | The click-four-corners tool that must be run before `main.py`; it captures the half-court corners the homography and zone geometry are built from. |
| `fetch_face_models.py` | Downloads the OpenCV YuNet and SFace models into `models/` once after cloning, since those weights are not committed. |
| `main.py` | The original full standalone system — capture, pose detection, zone scoring and on-screen overlay in one OpenCV loop. |
| `README.md` | The main project documentation: what the system does, how to install it, and how to run each part. |
| `requirements.txt` | The dependency list, with inline comments explaining why each package is needed and which ones Python 3.13 forced a workaround for. |
| `run_server.py` | Launches the FastAPI app under uvicorn on port 8000, which is the normal way to start the backend. |
| `runs_train.log` | Captured stdout from a local training run, kept for the record. |
| `runs_train_resume.log` / `.err` | Stdout and stderr from a resumed training run, retained so an interrupted run's history is not lost. |
| `seed.py` | Loads demo players, sessions and attendance into MongoDB, mirroring the frontend's seed data exactly so both sides show the same ids. |
| `yolov8n.pt` | The stock YOLOv8 nano detection checkpoint, used as the pretrained starting point for every shuttlecock training run. |
| `yolov8n-pose.pt` | The stock YOLOv8 nano pose checkpoint, which supplies the 17 COCO keypoints the skill metrics are computed from. |

### `setup/app/`

| File | What it does |
|---|---|
| `__init__.py` | Marks the backend as a package and carries its one-line description. |
| `db.py` | Opens the MongoDB connection and exposes thin per-collection repository helpers, so no router talks to pymongo directly. |
| `engine.py` | The headless drill engine: it runs the capture-detect-score loop on a background thread, publishing annotated JPEG frames to the streamer and structured events to the hub. |
| `esp32_camera_client.py` | Pulls a single snapshot from the ESP32-CAM enrollment station over HTTP, kept standalone so it cannot disturb the main capture path. |
| `events.py` | A WebSocket broadcast hub that bridges the synchronous engine thread to asynchronous browser clients through a queue. |
| `face.py` | Face detection and recognition through YuNet and SFace, guarded throughout so a missing model file degrades to "recognition unavailable" instead of crashing. |
| `models.py` | The Pydantic request and response schemas for the API, with defaults sourced from `config/settings.py`. |
| `server.py` | The FastAPI application entrypoint: it mounts every router, sets up CORS and exposes the WebSocket endpoint. |
| `streamer.py` | A thread-safe shared frame buffer holding the latest JPEG, which the MJPEG video endpoint reads from. |
| `routers/__init__.py` | Package marker for the router modules. |
| `routers/control.py` | Engine control endpoints — start, stop, arm, set difficulty — plus the MJPEG video stream response. |
| `routers/esp32_enroll.py` | Face enrollment driven by the ESP32-CAM station, added as its own router so it changes nothing in the existing player endpoints. |
| `routers/players.py` | CRUD for players, including attaching a face embedding to a player record. |
| `routers/sessions.py` | CRUD for training sessions, their status transitions, and reading back attendance. |
| `routers/settings.py` | Reads and writes runtime settings and the stored court calibration corners. |

### `setup/config/` and `setup/docs/`

| File | What it does |
|---|---|
| `config/settings.py` | Every hardcoded value in one place — camera source, court corners, zone boundaries, drill intervals and detection thresholds — calibrated from measured still frames. |
| `docs/feeder-court-distance-test.md` | The procedure for testing the detector at near, mid and far camera distance, and the results it produced. |
| `docs/scope-and-limitations.md` | An explicit statement of what the system does and does not claim, written for the thesis defence. |
| `docs/shuttle-v2-dataset.md` | The record of why the v1 dataset was discarded and what replaced it. |
| `docs/superpowers/plans/2026-07-13-court-space-homography-geometry.md` | The implementation plan for converting image-space detections into real court coordinates. |
| `docs/superpowers/specs/2026-07-12-court-space-homography-geometry-design.md` | The design spec for that homography layer and the reasoning against image-space zone boxes. |

### `setup/datasets/` — non-image files

| File | What it does |
|---|---|
| `data.yaml` (one per dataset) | The Ultralytics dataset descriptor naming the train/val/test paths and the class list. |
| `smashspeed-v8/data_selected.yaml` | A variant descriptor pointing at the size-selected subset instead of the full 13,000-frame set. |
| `labels.cache` | Ultralytics' binary cache of parsed labels, regenerated automatically whenever the labels change. |
| `feeder_court/segments.json` | The usable time window of each source clip, excluding the camera-motion head and tail where the rig was being carried. |
| `feeder_court/proposals/{near,mid,far}.json` | Per-clip auto-label proposals bucketed as positive, negative or uncertain, so background frames become usable rather than discarded. |
| `feeder_court_yolo.zip` | The packaged dataset as uploaded to Colab for the GPU training run. |
| `smashspeed-fetch.log` / `.err` | Console output from the long download of the smashspeed dataset, kept to show the fetch completed cleanly. |
| `vid_source/angle_1..Angle_4.mp4` | The earlier multi-angle recordings, from which the first shuttlecock datasets were cut. |
| `vid_source/shuttlecoc_testing_camera_live_2.mp4` | A fixed-camera test recording used as a stable input for repeated detector comparisons. |
| `vid_source/new_badminton_source/{near,mid,far}.mp4` | The three fixed-distance clips the detector is calibrated and evaluated against. |
| `vid_source/new_badminton_source/{line_1,line_2}.mp4` | Two line-of-play angles from the same session, recorded as additional scene variety. |

### `setup/models/`

| File | What it does |
|---|---|
| `face_detection_yunet_2023mar.onnx` | The YuNet face detector used to locate faces before embedding them. |
| `face_recognition_sface_2021dec.onnx` | The SFace model that turns a cropped face into the embedding vector player identity is matched on. |
| `shuttlecock.pt` | The v1 shuttlecock weights, retained only as the historical baseline the thesis text cites. |
| `shuttlecock_v2.pt` | The v2 weights, trained on close-up photos, which detect confidently but also fire on blank walls. |
| `shuttlecock_v3.pt` | The v3 weights trained on smashspeed court footage — the first model to detect a shuttle at real playing distance. |
| `shuttlecock_v3_smoke15.pt` | A 15-epoch smoke-test checkpoint of that run, used to confirm the pipeline before committing to the full training. |
| `shuttlecock_scene_dummy.pt` | The single-clip scene-specific model, kept as the measured demonstration of the single-clip training ceiling. |
| `shuttlecock_scene_v2.pt` | The combined-session scene model that fixed distance detection while regressing near detection, at 3.3x the speed of the imgsz-1280 workaround. |
| `*_openvino_model/metadata.yaml` | The export metadata recording the input size and class names an IR export was built with. |
| `*_openvino_model/*.xml` | The OpenVINO network topology description. |
| `*_openvino_model/*.bin` | The corresponding weight blob loaded alongside that XML. |

### `setup/runs/`

| File | What it does |
|---|---|
| `feeder_court_train.log` | Console output of the feeder_court training run. |
| `scene_dummy_train.log` / `scene_dummy_combined.log` | Console output of the two scene-dummy training runs, one per split. |
| `train_v3_full.log` / `.err` | Stdout and stderr of the full v3 training run on the smashspeed data. |
| `train_v3_smoke.log` / `.err` | The same for the short smoke run that preceded it. |
| `webcam_test.log` / `.err`, `webcam_v3only.log` / `.err` | Output from the live webcam smoke tests, capturing which model reported what on real input. |
| `labelimg.log` | Output from the labelling tool session, kept alongside the dataset it produced. |
| `train.pid`, `train_v3.pid`, `train_v3_full.pid`, `webcam_v3.pid` | The process IDs of background training and webcam runs, so a long job can be checked on or stopped later. |
| `shuttle_motion.csv` | Per-frame output of the motion prototype — candidate counts and timings — which is where the 13.4 ms/frame figure comes from. |
| `shuttle_motion_out.mp4` | The annotated video from that prototype, showing what the motion detector actually selected. |
| `shuttle_test.mp4`, `shuttlecock_testing_camera_live.mp4` | Test recordings used as fixed inputs so detector changes are compared on identical footage. |
| `args.yaml` (per run) | The exact Ultralytics hyperparameters that run was launched with, which is what makes it reproducible. |
| `results.csv` (per run) | One clean row per epoch of losses and metrics; `train_status.py` reads this rather than the log. |
| `results.png` (per run) | The plotted version of that CSV across all epochs. |
| `BoxP_curve.png`, `BoxR_curve.png`, `BoxF1_curve.png`, `BoxPR_curve.png` | Precision, recall, F1 and precision-recall curves against confidence, used to choose the deployment threshold. |
| `confusion_matrix.png` / `confusion_matrix_normalized.png` | Confusion matrices in absolute and normalised form. |
| `labels.jpg` | A summary plot of the label distribution — box sizes and positions — which is where a bad dataset usually announces itself. |
| `train_batch*.jpg` | Mosaics of augmented training batches, the fastest way to spot broken labels or wrong augmentation. |
| `weights/best.pt` | The checkpoint from the best-scoring epoch, and the one promoted into `models/`. |
| `weights/last.pt` | The final-epoch checkpoint, needed to resume an interrupted run. |
| `feeder_court/detect/{near,mid,far}_boxed.mp4` | The three distance clips with detection boxes burned in, which are the deliverable of the distance test. |

### `setup/scripts/`

| File | What it does |
|---|---|
| `audit_dataset.py` | Inspects a YOLO dataset before training — duplicate frames, split leakage, box-size distribution — after two training runs were wasted on problems it now catches. |
| `autolabel_scene_video.py` | Auto-labels one fixed-camera clip by exploiting that the shuttle is the only large, bright, desaturated object in that specific scene. |
| `build_combined_scene.py` | Merges the two recording sessions into one training set, deliberately dropping session 2's empty label files rather than trusting them as negatives. |
| `fetch_shuttle_v2.py` | Downloads the v2 dataset from Roboflow, with a header documenting why v1 had to be replaced. |
| `prepare_fast_subset.py` | Pre-resizes a selected subset so CPU training is bottlenecked on compute rather than on decoding 1080p and 4K JPEGs every epoch. |
| `prepare_for_roboflow.py` | Extracts frames and writes low-confidence pre-labels for human correction, favouring recall because deleting a wrong box is faster than drawing a missing one. |
| `prune_leaky_eval.py` | Moves near-duplicate validation images back into train, catching the duplicates that filename-based splitting cannot see. |
| `resplit_by_clip.py` | Re-splits a Roboflow export by source clip rather than by frame, which is the fix for the v1 leakage that inflated mAP50 to 0.898. |
| `select_training_subset.py` | Selects training images by shuttle pixel size instead of at random, because a random subsample is dominated by shuttles under 12 px. |
| `shuttle_motion_prototype.py` | The motion-based detector prototype, written after every YOLO attempt failed because a single frame does not contain enough signal. |
| `split_scene_dummy.py` | Builds a random and a temporal split of the same clip so the leakage-induced score gap can be measured directly. |
| `test_shuttle_v2_webcam.py` | A standalone live webcam smoke test for the v2 weights, importing nothing from `app/` so running it cannot affect the drill system. |
| `train_scene_dummy.py` | Trains the two scene-dummy models with identical settings so only the split differs. |
| `train_shuttlecock.py` | Trains the v1 weights locally, written because Roboflow's hosted inference costs 1.2–3.9 s per round trip — far too slow to track a shuttle in flight. |
| `train_shuttlecock_v2.py` | Trains the v2 and v3 weights, kept as a separate file so the v1 script stays an unmodified record of numbers the thesis cites. |
| `train_status.py` | Prints readable training progress from `results.csv`, avoiding the carriage-return progress bars that make the raw log unreadable. |
| `feeder_court/segment_clips.py` | Emits each clip's usable window as JSON, excluding the head and tail where the rig is being carried into and out of position. |
| `feeder_court/sample_calibration.py` | Phase 0 of the pipeline: samples 20 frames per clip for hand-boxing, then reads the labels back and prints the size-gate decision. |
| `feeder_court/propose_shuttles.py` | Generates auto-label proposals in three buckets — positive, negative, uncertain — so background frames become usable training data. |
| `feeder_court/train_local.py` | Trains the detector on CPU against the calibration-sized dataset, a roughly 4 hour run versus the ~30 the full dataset would cost. |
| `feeder_court/detect_video.py` | Runs the trained detector over a video and burns the boxes in, defaulting to imgsz 1280 because dropping to 640 halves the shuttle. |
| `feeder_court/live_detect.py` | The live USB-camera equivalent, deliberately standalone so it reads no shared config and cannot disturb the running drill system. |

### `setup/tests/`

| File | What it does |
|---|---|
| `test_health.py` | Asserts the health endpoint reports MongoDB connectivity truthfully. |
| `test_db.py` | Verifies the Mongo connection pings and that the collection helpers target the right database. |
| `test_models.py` | Checks the Pydantic schemas populate their defaults from `config/settings.py`. |
| `test_engine.py` | Covers the engine's initial state and its basic lifecycle transitions. |
| `test_engine_persist.py` | Checks zone-key conversion and that completed sessions are written to Mongo. |
| `test_engine_reco.py` | Verifies that a face-recognised track wins over a positionally attributed player when a shot is assigned. |
| `test_engine_skill.py` | Confirms the engine flushes an accumulated skill profile into the player's record. |
| `test_events.py` | Confirms broadcasts made from the synchronous engine thread are correctly enqueued for async delivery. |
| `test_streamer.py` | Covers publishing to and reading from the shared JPEG frame buffer. |
| `test_ws.py` | Drives a real WebSocket client to confirm broadcast events actually reach the browser. |
| `test_control_api.py` | Exercises the engine-control endpoints, including setting difficulty. |
| `test_control_arm.py` | Covers unarmed start and arm-in-place, so identity acquisition can run recognition before the feeder fires. |
| `test_players_api.py` | Covers player CRUD through the HTTP layer. |
| `test_sessions_api.py` | Covers session CRUD and status transitions. |
| `test_settings_api.py` | Checks settings defaults are created on first read and persist on write. |
| `test_face.py` | Unit-tests the cosine similarity and the guards around missing face models. |
| `test_face_enroll.py` | Covers the end-to-end enrollment flow that attaches an embedding to a player. |
| `test_seed.py` | Confirms the seed script inserts the expected demo players and sessions. |
| `test_zones.py` | Covers court zone geometry and player-to-zone matching. |
| `test_pose_features.py` | Tests the pure pose-geometry helpers, especially that low-confidence keypoints yield `None` rather than a junk metric. |
| `test_skill_profile.py` | Verifies the running mean and standard deviation accumulate correctly without keeping raw samples. |
| `test_shuttle_worker.py` | Tests the background shuttle runner against a fake detect function, so no camera or network is required. |
| `test_roboflow_client.py` | Smoke-tests both Roboflow paths, documenting that the direct model works while the full workflow is blocked by a compile bug. |
| `test_feeder_court_motion.py` | Tests candidate extraction against synthetic scenes with a known shift. |
| `test_feeder_court_roi.py` | Tests the court-band crop geometry and its bounds handling. |
| `test_feeder_court_segments.py` | Confirms the head and tail camera-motion regions are excluded from the usable window. |
| `test_feeder_court_sizes.py` | Tests conversion from YOLO-normalised lines to pixel dimensions and the size gate built on it. |
| `test_feeder_court_tracks.py` | Tests that noisy per-frame candidates are linked into plausible flight tracks. |

### `setup/utils/`

| File | What it does |
|---|---|
| `display.py` | Draws the overlays — zones, scores, boxes and FPS — onto the camera frame. |
| `pose_features.py` | Pure geometry helpers over the 17 COCO keypoints, each confidence-gated so a jittery point never becomes a metric. |
| `roboflow_client.py` | Calls the hosted Roboflow serverless workflow over plain REST to detect a shuttlecock in a single frame. |
| `roboflow_stream.py` | The real-time WebRTC path to Roboflow's cloud GPU, for a live webcam, an RTSP stream or a file. |
| `scoring.py` | Accumulates per-player, per-zone scores and flags zones that fall below the weakness threshold. |
| `shuttle_worker.py` | Runs shuttle detection on a background thread, since a ~1 s inline call would drag the render loop below 1 FPS. |
| `skill_profile.py` | Accumulates per-player skill statistics and applies the rule-based rubric that turns them into a rating. |
| `zones.py` | Court zone detection and player-to-zone matching, built on the calibrated corner geometry. |
| `feeder_court/__init__.py` | Package marker for the detector's pure core. |
| `feeder_court/motion.py` | Isolates fast movers on a drifting camera, after median-background subtraction failed because the rig drifts tens of pixels even inside a stable window. |
| `feeder_court/roi.py` | Crops to the court band, which is 2.7x cheaper than letterboxing the full frame to 1280x1280 and costs zero shuttle pixels. |
| `feeder_court/segments.py` | Finds each clip's usable window, excluding the carried-rig footage at head and tail. |
| `feeder_court/sizes.py` | Measures real shuttlecock pixel size and gates candidates on it, after automated probing returned a suspiciously identical distribution for near and far. |
| `feeder_court/tracks.py` | Links noisy per-frame candidates into flights, using the fact that a shuttle flight lasts 15–40 frames and goes roughly where its velocity says it will. |

### Generated and cache files

| File | What it does |
|---|---|
| `**/__pycache__/*.pyc` | Compiled bytecode written automatically by the interpreter; regenerated on import and gitignored. |
| `.pytest_cache/v/cache/lastfailed` | The set of tests that failed on the last run, which is what enables `pytest --lf`. |
| `.pytest_cache/v/cache/nodeids` | The collected test node IDs from the last run, used to speed up collection. |
| `.pytest_cache/CACHEDIR.TAG` | The standard marker telling backup tools this directory is a regenerable cache. |
| `.pytest_cache/README.md` | Pytest's own note explaining that the directory is auto-generated and should not be committed. |
| `.venv-stream/pyvenv.cfg` | Records which base Python interpreter the streaming virtualenv was created from. |
