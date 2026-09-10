# Finger posing for pose_transfer_photo

## Context

`pose_transfer_photo.py` (`ai_vision/pose_transfer_photo/`) extracts a body
pose from a photo via MediaPipe PoseLandmarker and drives a Genesis 9
figure's wrists/ankles toward the photo-derived positions via multi-effector
IK. As of bd `daz-script-server-hewu`, the `--backend pinocchio` path solves
this locally (zero HTTP round-trips per iteration) with active-set
joint-limit handling verified both synthetically and live against DAZ
Studio.

Fingers are currently untouched — the figure's hands stay at whatever pose
they were zeroed to. This spec adds a `--fingers` flag that also poses each
detected hand's fingers from the same photo, once MediaPipe can see the
hand clearly enough.

## Goals / non-goals

- **Goal:** extend the existing pipeline, not build a separate tool.
  Fingers are posed from the same photo, in the same run, after the
  existing body IK solve.
- **Goal:** reuse `pinocchio_ik.py`'s `build_figure_model`/`solve_ik`
  as-is — no new solver code. Finger joints are, if anything, a better fit
  for the active-set machinery than the body chains: Genesis 9's own
  `axis_limits` show most distal phalanges are pure single-axis hinges
  (e.g. `l_index2`/`l_index3`: x and y both `min==max==0`), the exact
  degenerate case that machinery was built to handle correctly.
- **Non-goal:** `--backend stacked` support. Solving ~40 extra DOF (5
  fingers × ~3 joints × 2 hands × up to 3 axes) via HTTP finite-difference
  Jacobians would be far too slow — the whole reason `--backend pinocchio`
  exists. `--fingers` requires `--backend pinocchio`; combining it with
  `--backend stacked` is a clear CLI error, not a silent no-op.
- **Non-goal:** perfect anthropometric fingertip length modeling. See
  "Why the tip bone is handled differently" below — we sidestep needing a
  fingertip length entirely rather than approximate one.

## Genesis 9 finger rig (confirmed live against "Jason Cross")

Per hand: `l_thumb{1,2,3}`, and per non-thumb finger (index/mid/ring/pinky):
`l_<finger>metacarpal`, `l_<finger>{1,2,3}`. All parented up to `l_hand`
(mirrored for `r_*`).

`bone_metadata()`'s `axis_limits` for a representative finger (`l_index*`):

| bone | parent | free axes |
|---|---|---|
| `l_indexmetacarpal` | `l_hand` | y (±3–4°), z (±4°); x dead |
| `l_index1` | `l_indexmetacarpal` | x, y, z all free (z is the main knuckle flex, -95..50°) |
| `l_index2` | `l_index1` | z only (-105..12°); x, y dead |
| `l_index3` | `l_index2` | z only (-90..20°); x, y dead |

Thumb (`l_thumb*`) is looser (real x/y/z range on `thumb1` for
opposability) but the same shape: `thumb1`/`thumb2` have some freedom,
`thumb3` is a single-axis hinge (y only).

**Important:** `bone_metadata()` gives each bone's own joint *origin*, not
its length or endpoint. There is no bone or metadata field marking where a
fingertip physically is.

## Why the tip bone is handled differently

For a chain `A -> B -> C` (e.g. `l_index1 -> l_index2 -> l_index3`), bone
`C`'s own rotation affects the position of `C`'s *children* — but `l_index3`
has no child bone in the rig, so its own rotation affects nothing we can
see as a world position. IK cannot meaningfully solve for it: there is no
observable effector whose error depends on that DOF. This is separate from
(and would exist regardless of) the missing-fingertip-length problem above.

Fortunately Genesis 9's own limits show this bone is a pure single-axis
hinge anyway. So per digit:

- IK-solve every bone **except the last** against real, observable rig
  joint positions (each intermediate bone's *child's* joint = a landmark).
- Set the last bone's single free-axis angle **directly from landmark
  geometry** — the angle between the two landmark-derived vectors on either
  side of that joint, which is exactly what a hinge angle is.

This means no fingertip length is ever needed, and every DOF is driven by
the technique suited to it (multi-DOF proximal/middle joints → IK against
real targets; single-DOF terminal hinge → direct geometric angle).

## Digit → chain → landmark mapping

MediaPipe HandLandmarker's 21 landmarks per hand (standard indices):
`0=WRIST, 1-4=THUMB_{CMC,MCP,IP,TIP}, 5-8=INDEX_{MCP,PIP,DIP,TIP},
9-12=MIDDLE_*, 13-16=RING_*, 17-20=PINKY_*`.

For a 4-bone finger (index/mid/ring/pinky), e.g. index:

| rig bone | its own world position corresponds to | role |
|---|---|---|
| `l_indexmetacarpal` | (near-fixed; ±3–4° range) | rotatable, IK |
| `l_index1` | MCP landmark (5) | rotatable, IK |
| `l_index2` | PIP landmark (6) | rotatable + **effector** (target = PIP) |
| `l_index3` | DIP landmark (7) | trailing marker + **effector** (target = DIP); own hinge angle set from TIP (8) geometry |

So: `chain_bones = ["l_indexmetacarpal", "l_index1", "l_index2", "l_index3"]`,
`effector_bones = ["l_index2", "l_index3"]`, target points = transformed
PIP and DIP landmarks. This is the *exact same* multi-effector stacked-IK
shape `_CHAINS` already uses for hand/foot targets (a trailing bone
included purely as a position marker) — `build_figure_model`/`solve_ik`
need no changes.

Thumb is the same pattern with one fewer bone:
`chain_bones = ["l_thumb1", "l_thumb2", "l_thumb3"]`,
`effector_bones = ["l_thumb2", "l_thumb3"]`, targets = MCP (2), IP (3)
landmarks; `l_thumb3`'s hinge angle set from TIP (4) geometry.

This mapping is identical in shape for `r_*` and for mid/ring/pinky
(substitute their landmark index ranges). One generic function handles all
10 digit-chains (5 × 2 hands).

**Hinge angle from geometry:** for a bone whose only free axis is `z` (say),
compute vectors `v_in = joint_pos - parent_joint_pos` and
`v_out = tip_landmark - joint_pos` (both from the *transformed* landmark
positions, not rig positions), and the signed angle between them about that
bone's local hinge axis (using its `rest_orientation` to express both
vectors in the bone's local frame first, then `atan2` of the free axis's
in-plane components — same rest-orientation-correction pattern
`bone_local_rotation` already uses elsewhere in `pinocchio_ik.py`). Clamp to
`axis_limits` afterward (reusing `clamp_angles`).

## Coordinate alignment (the open risk, resolved by staged validation)

MediaPipe Pose and Hand Landmarker world landmarks may or may not share a
rotational axis convention (both are metric, subject-centered, but centered
on different points). Per discussion: **assume they share a convention,
verify live, add a correction only if wrong.**

Pipeline:

1. Run the existing body IK solve first (unchanged), producing final
   `l_hand`/`r_hand` world positions.
2. Run MediaPipe HandLandmarker on the same photo, per detected hand.
3. Apply the *same* scale+rotation transform already computed for body
   calibration (shoulder width / hip origin) to that hand's 21 landmarks.
4. Translate the transformed set so its own WRIST landmark (0) lands
   exactly on the already-solved `l_hand`/`r_hand` world position — a pure
   translation, no new rotation estimate.
5. Run the per-digit chains from the mapping above.
6. **Validate live**, the same way the body active-set fix was validated:
   render/inspect the resulting hand pose in DAZ Studio. If fingers curl or
   splay in a physically wrong direction (not just "off by some IK
   residual"), that specifically indicates the shared-convention assumption
   is false — the fix at that point is a per-hand Kabsch/Procrustes
   rotation alignment (using WRIST + INDEX_MCP + PINKY_MCP as a stable
   corresponding triangle against the rig's own rest-pose hand geometry),
   not a redesign of the chain-solving approach above.

## CLI

New flag `--fingers` (default off). Validation at startup:

- `--fingers` with `--backend stacked` → clear error, exit non-zero.
- `--fingers` with `--no-hands` → clear error (fingers depend on the wrist
  already being solved).

## Error handling

Per digit-chain group (i.e. per hand): if MediaPipe fails to detect that
hand, or its landmarks are below a confidence threshold, skip finger IK for
that hand entirely — leave its fingers at whatever pose they already had
(typically zeroed). Reported in the run summary alongside existing
per-limb `[OK]`/`[NOT CONVERGED]` lines, e.g. `l_hand fingers: SKIPPED (no
hand detected)`.

## Testing

- **Synthetic, no live DAZ Studio** (mirrors `test_pinocchio_ik.py`'s
  approach): the hinge-angle-from-geometry function checked against known
  vector pairs with hand-computed expected angles; the digit → chain →
  landmark-index mapping table checked for internal consistency (every
  referenced bone name is a plausible Genesis 9 name, every landmark index
  in `0..20`, thumb chains one bone shorter than finger chains).
- **Live validation** (same methodology as the body active-set fix):
  run against DAZ Studio with a real hand-visible photo, `zero_figure()`
  first, inspect per-digit convergence errors and the resulting hand shape
  by eye for the coordinate-alignment risk called out above.

## Out of scope for this spec

- The original epic's batch-processing scope (caching one `FigureModel` per
  unique figure across many photos, running many-photos × many-figures)
  remains separate, tracked work — this spec only adds finger posing to the
  existing single-figure/single-photo run.
