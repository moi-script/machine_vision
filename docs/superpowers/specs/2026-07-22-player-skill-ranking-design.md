# Player Skill Ranking — Design Spec

**Date:** 2026-07-22
**Repo:** `C:\thesis` (backend / `setup/`)
**Status:** Approved design — ready for implementation planning
**Scope:** Spec 1 of 2 (backend foundation). Spec 2 (frontend visualization in `C:\thesis_ui`) is a separate later spec.

## 1. Goal

Accumulate per-player analytics — **body posture**, **movement/trajectory**, and **stroke quality**, all derived from the pose-estimation stream we already produce — into a **cumulative career profile** per player, and map that profile to a five-tier skill ranking:

`Beginner · Novice · Intermediate · Advanced · Expert`

The ranking is **rule-based** (explainable thresholds, no trained classifier, no labeled dataset required) and every player accumulates data across **all** their drill sessions for life.

## 2. Key design decisions (locked)

| Decision | Choice | Why |
|---|---|---|
| Ranking method | Rule-based rubric | Explainable, defensible in a thesis, works immediately with no labeled data. |
| Accumulation model | Cumulative career profile | Matches "data accumulation for a specific player"; tier climbs as they train. |
| Stroke quality | Pose-based swing proxy | No racket/swing detector exists; arm kinematics from existing keypoints give a real, defensible approximation. |
| Metric families | Posture + Movement + Stroke + Accuracy | "Pose estimation" is the underlying tech (YOLOv8n-pose), not a separate scored family. |

## 3. Reality constraints this design is built for

The pipeline is **not** clean/high-FPS/close-range. It is:
- **~10 FPS** worst case (Raspberry Pi target).
- **Long distance** — fine joint angles (knee, torso, shoulder) get low keypoint confidence and jitter.
- **Laggy, intermittent shuttle detection** — serverless adds 0.1–0.4 s latency; the shuttle is a box that blinks on/off, not a smooth track.

Four principles remove the dependence on the unreliable signals:

1. **Time off software events, never the shuttle.** Reaction lag = `_fire_feeder` timestamp (frame-exact, our own event) → first significant **ankle** displacement toward the target zone. The ankle is the most reliable keypoint (large, grounded, consistently tracked). The shuttle is never in this measurement.
2. **Detect strokes from the pose, not the shuttle.** A swing = a wrist-speed spike relative to the shoulder, computed locally at pose FPS. The shuttle is used **only for outcome** (returned/scored — coarse, already confirmed over `RETURN_CONFIRM_FRAMES`).
3. **Confidence + distance gate every fine metric.** A posture/stroke sample is recorded only when (a) the relevant keypoints' confidence exceeds a threshold AND (b) the person's bbox height in pixels exceeds a threshold (proxy for "close enough to resolve this"). Otherwise the metric is **not sampled** rather than sampled wrongly.
4. **Accumulate distributions, not instants.** Every metric is a running aggregate (Welford mean + variance + count) over many gated samples. Low FPS stops mattering for spatial/statistical aggregates, and **consistency (low variance) becomes a first-class skill signal** — experts are repeatable.

## 4. Metrics (rubric inputs), tiered by trustworthiness

### Tier A · Robust — always sampled (feet & gross position; low-FPS-proof). Carries the ranking.
- **Court coverage** — how much of the 6-zone area the player reaches.
- **Distance traveled per shot** (court units).
- **Movement speed** — avg + peak, from ankle court-space positions.
- **Recovery-to-center** — does the player return toward base after a shot?
- **Reaction lag** — feeder-fire timestamp → first significant ankle displacement toward target.
- **Accuracy** — returns/shots overall and per zone (already tracked today).

### Tier B · Opportunistic — sampled only when bbox-size + keypoint-confidence gates pass.
- **Knee bend** (hip–knee–ankle angle) — target-band scored (athletic ready vs. stiff/collapsed).
- **Torso lean / balance** (shoulder-midline vs. hip-midline).
- **Stance width** (ankle separation, normalized to body height) — target-band scored (NOT "wider = better").
- **Posture consistency** (variance of the above) — accumulates whenever the player is near enough.

### Tier C · Stroke — pose-only, window-based, imprecision-tolerant.
- **Swing presence** — wrist-speed spike on the arm nearest the incoming shuttle, in a window around the return.
- **Preparation range** — shoulder/elbow extension before the peak.
- **Follow-through** — motion arc continuing after the peak.
- **Swing consistency** (variance of the kinematic signature) — the primary stroke-quality signal; gated on arm-keypoint confidence.

**Universal rule:** every metric stores its own **reliable-sample count**, and the rubric weights each metric/family by how much trustworthy data backs it (see §5).

## 5. Scoring → composite → tier

**Step 1 — Normalize each metric to 0–100** against tunable reference bands in settings. Three metric shapes:
- **Monotonic** (more/less better): linear map between `min`/`max` refs, clamped 0–100 (e.g. accuracy ↑, reaction lag ↓).
- **Consistency** (variance): inverted — lower variance → higher score.
- **Target-band** (stance width, knee bend): scored by closeness to a target band; middle is best.

**Step 2 — Family sub-scores.** Metrics roll into the four families, each a 0–100 sub-score.

**Step 3 — Data-weighted composite.** Default family weights:

| Family | Default weight |
|---|---|
| Movement / Trajectory | 30% |
| Accuracy | 30% |
| Stroke | 20% |
| Posture | 20% |

Each family's weight is **scaled by whether it has enough reliable samples**; a family below its sample floor has its weight **redistributed** to families that qualify (weights renormalize). A thin family never drags the tier toward a wrong answer — it steps aside.

**Step 4 — Tier bands** (tunable):

| Composite | Tier |
|---|---|
| 0–20 | Beginner |
| 20–40 | Novice |
| 40–60 | Intermediate |
| 60–80 | Advanced |
| 80–100 | Expert |

**Step 5 — Minimum-data gate.** Below a floor of accumulated reliable shots/samples → **"Unranked — insufficient data"** instead of a noisy tier.

**Trend:** append a lightweight per-session snapshot (composite + family scores) to `skillHistory` so a player can be shown climbing tiers over time.

All weights, band cutoffs, reference bands, gates, and the min-data floor are **tunable constants** (calibrate against real players without code changes).

## 6. Architecture & components (all in `setup/`)

### New files
- **`utils/pose_features.py`** — pure, stateless geometry from one keypoints frame (17×3) + optional previous frame + dt. Computes joint angles (knee, elbow, torso lean), stance width, wrist speed, ankle displacement, bbox height. **Confidence gating baked in**: returns an "unreliable"/`None` marker instead of a bad number when a keypoint's confidence is below threshold. Fully unit-testable.
- **`utils/skill_profile.py`** — two parts:
  - `SkillAccumulator` — stateful running aggregates (Welford mean/variance + counts) per attributed player during a drill. Consumes: `add_frame(...)`, `on_feeder_fired(ts, zone)`, `on_return(...)`. Tracks a reliable-sample count per metric. Method `snapshot()` returns the session aggregates. Includes a `merge_into(cumulative)` helper for correct cross-session accumulation.
  - `evaluate_rubric(profile) -> {composite, tier, families, breakdown}` — pure function: normalize → family → data-weighted composite → tier band. Config-driven.

### Modified files
- **`config/settings.py`** — add tunable constants: per-metric reference bands, family weights, tier cutoffs, min-data floor, keypoint-confidence gate, bbox-height gate.
- **`app/engine.py`** — in the `_run` loop, feed each frame's detection for the attributed/recognized player into the `SkillAccumulator`; hook `_fire_feeder` and return/score events into it; on drill stop/complete, **merge the session aggregates into the player's cumulative `skillProfile` in Mongo**, append a `skillHistory` snapshot, re-evaluate the rubric, store the tier.
- **`app/routers/players.py`** — `GET /api/players/{id}/skill` → profile + tier + per-family breakdown + history. Fold `skillProfile.tier` into `get_player`. Keep bulky raw aggregates out of the default list payload (same pattern as `faceEmbedding` stripping); expose full breakdown only on the dedicated endpoint.

### Storage — Mongo player doc (new fields)
```
skillProfile: {
  sampleCounts: { move, posture, stroke, accuracy },   # reliable-sample counts
  aggregates: {                                         # running: sum, sumsq, count, min, max per metric
    move:     { <metric>: {n, sum, sumsq, ...}, ... },
    posture:  { ... },
    stroke:   { ... },
    accuracy: { ... },
  },
  computed: { composite, tier, families: {move, posture, stroke, accuracy}, updatedAt },
}
skillHistory: [ { at, sessionId, composite, tier, families } ]
```
Running aggregates (sum + sum-of-squares + count), **not just means**, so cumulative career accumulation is mathematically correct across sessions.

### Data flow
```
camera frame → YOLO pose (existing) → attributed playerId (face / single-trainee, existing)
  → pose_features(frame, prev)             (gated)
  → SkillAccumulator.add_frame(...)        ← _fire_feeder(ts) / return events
  → on stop: merge into Mongo skillProfile → evaluate_rubric → tier stored
  → GET /api/players/{id}/skill → (Spec 2 UI)
```

## 7. Observability — `[SKILL]` console logging (first-class, in the backend terminal)

Matches the existing `[SHUTTLE]` / `[RECO]` style: `print(..., flush=True)`, throttled so it informs without spamming.

- **Per-sample (~1/s) — proves gating works:**
  `[SKILL] <pid> frameseen: bbox_h=412px posture=SAMPLED knee=138° stroke=gated(arm conf 0.31) move=SAMPLED speed=2.1`
- **Feeder-timed reaction captured:**
  `[SKILL] <pid> reaction: feeder→first-move = 0.42s (zone back-left)`
- **Pose-detected swing:**
  `[SKILL] <pid> swing: wrist-speed peak=6.8 prep-range=71° consistency-sample #14`
- **Drill end — merge + persisted result (the "is it saving?" proof):**
  ```
  [SKILL] <pid> session merged → cumulative samples: move=1830 posture=142 stroke=17 accuracy=63
  [SKILL] <pid> rubric: move=64 acc=58 stroke=41 posture=insufficient → composite=57.3 tier=INTERMEDIATE
  [SKILL] <pid> skillProfile written to Mongo (history len=4)
  ```

## 8. Testing

- **`pose_features`** — angle math correctness, confidence-gate behavior (weak keypoint → unreliable, not a wrong number), bbox-height computation.
- **`skill_profile`** — Welford accumulation + cross-session merge math; rubric band edges; data-weight redistribution when a family is thin; min-data floor → "Unranked".
- Run with the existing suite: `python -m pytest tests/ --ignore=tests/test_shuttle_worker.py`.

## 9. Phasing / decomposition

- **Spec 1 (this doc — backend foundation):** `pose_features` + `skill_profile` (accumulator + rubric) + settings constants + storage + engine integration + `[SKILL]` logging + skill API + tests. Delivers a real, queryable per-player tier. No new UI beyond the API.
- **Spec 2 (later, `C:\thesis_ui`):** frontend — tier badge, family breakdown (radar), trend chart on the player profile.

Tier C (stroke) rides along in Spec 1 but is isolated so it can be validated/tuned independently on the real camera.

## 10. Non-goals (YAGNI)

- No trained ML classifier (rule-based only).
- No racket detection / true swing biomechanics.
- No frontend work in this spec.
- No per-session-only or rolling-window modes (cumulative only).
