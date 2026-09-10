"""Local Pinocchio-based FK/IK backend for pose_transfer_photo.

WHY THIS EXISTS
----------------
`pose_transfer_photo.py`'s default `solve_stacked_ik()` does ~150 IK
iterations, each one an HTTP round-trip to DAZ Studio's main thread (it
perturbs every chain bone there and reads back effector positions to build a
finite-difference Jacobian). That's fine for one photo but doesn't scale to
many photos x many figures, since the round-trip cost dominates. This module
moves the entire iterative solve out of DAZ Studio: build a local kinematic
model once per figure from `DazSkeleton.bone_metadata()`, run every IK
iteration against that local model with Pinocchio's analytic Jacobians (zero
HTTP calls), and push the final joint angles to DAZ Studio with exactly one
`set_bone_rotations()` call.

REQUIRES REAL PINOCCHIO (NOT THE PyPI PACKAGE NAMED "pinocchio")
------------------------------------------------------------------
The actual C++/Eigen robotics library is distributed on PyPI as `pin`, not
`pinocchio` (that name on PyPI is unrelated software). `pin` has no Windows
wheel and fails to build from source there without a full VS/Eigen toolchain.
On Windows, install it from conda-forge instead:

    conda create -n pinocchio-ik -c conda-forge python=3.11 pinocchio numpy scipy
    conda activate pinocchio-ik
    pip install -e Y:/working/BlueMoonFoundry/daz-script-server   # dazpy itself

Run `pose_transfer_photo.py --backend pinocchio` with *that* environment's
python.exe -- `--backend pinocchio` still needs mediapipe/opencv in-process
(landmark extraction happens before the solve), so `pip install mediapipe
opencv-python` into `pinocchio-ik` too rather than keeping a second venv;
`--backend stacked` (the default) needs mediapipe/opencv but not Pinocchio,
so it's fine to keep using a lighter venv for that path if Pinocchio is
never needed there.

IMPORTANT (Windows): invoke this conda env's python.exe directly by absolute
path only after prepending `<env>/Library/bin` to PATH (conda's own
activation normally does this for you) -- numpy's BLAS DLL lives there, and
without it on PATH, plain matrix multiplication segfaults the process with
no Python traceback at all (looks like the interpreter vanished). Prefer
`conda run -n pinocchio-ik python pose_transfer_photo.py ...` or an activated
shell over calling the env's python.exe by raw path.

THE FK FORMULA (validated live against DAZ Studio, see bd daz-script-server-hewu)
-----------------------------------------------------------------------------------
For a bone with pose channel angles (X, Y, Z degrees) and a given
`rotation_order` string (e.g. "XYZ"):

    R_local(bone) = O^-1 @ compose(rotation_order[::-1], angles_xyz) @ O

where O is the bone's `rest_orientation` quaternion (from `bone_metadata()`)
as a rotation matrix, and `compose(order, angles)` is the standard sequential
intrinsic Euler product (R = R[order[0]] @ R[order[1]] @ R[order[2]]).
Propagating this down a 4-bone chain (translation offsets from each bone's
rest `world_position`) matched DAZ Studio's live `getWSPos()` exactly
(diff_norm=0.0000) for every bone, including simultaneous non-trivial
rotations on all three axes. This module is a from-scratch reimplementation
of that same math on top of Pinocchio's `JointModelSpherical` (one per bone,
quaternion-configured) so Pinocchio can supply analytic position Jacobians
for the IK solve instead of finite differences.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation


# ── Euler <-> rotation matrix, matching bone_local_rotation()'s formula ────────

def _axis_rotation(axis: str, degrees: float) -> np.ndarray:
    theta = np.deg2rad(degrees)
    c, s = np.cos(theta), np.sin(theta)
    if axis == "X":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "Y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _compose_euler(order: str, angles: dict[str, float]) -> np.ndarray:
    """Sequential intrinsic product R = R[order[0]] @ R[order[1]] @ R[order[2]].

    `order` is upper-case (e.g. "XYZ", from `getRotationOrder().toString()`);
    `angles` is keyed lower-case (e.g. "x"/"y"/"z", matching `bone_metadata()`'s
    `local_euler` and `bone_rotations()`'s convention).
    """
    R = np.eye(3)
    for axis in order:
        R = R @ _axis_rotation(axis, angles[axis.lower()])
    return R


def _decompose_euler(order: str, R: np.ndarray) -> dict[str, float]:
    """Invert `_compose_euler`: angles (degrees) such that compose(order, angles) == R.

    scipy's intrinsic (uppercase) Euler sequence convention is exactly
    `_compose_euler`'s: `Rotation.from_matrix(R).as_euler(seq)` with
    `seq == order` returns angles a0, a1, a2 with
    R == R[order[0]](a0) @ R[order[1]](a1) @ R[order[2]](a2). Returned keys are
    lower-case to match `bone_rotations()`/`set_bone_rotations()`'s convention.
    """
    angles = Rotation.from_matrix(R).as_euler(order, degrees=True)
    return {axis.lower(): float(a) for axis, a in zip(order, angles)}


def _rest_orientation_matrix(bone_meta: dict) -> np.ndarray:
    o = bone_meta["rest_orientation"]
    return pin.Quaternion(o["w"], o["x"], o["y"], o["z"]).toRotationMatrix()


def bone_local_rotation(bone_meta: dict, angles_xyz: dict[str, float]) -> np.ndarray:
    """R_local(bone) = O^-1 @ compose(rotation_order[::-1], angles) @ O."""
    O = _rest_orientation_matrix(bone_meta)
    order = (bone_meta["rotation_order"] or "XYZ")[::-1]
    R_pose = _compose_euler(order, angles_xyz)
    return O.T @ R_pose @ O


def angles_from_local_rotation(bone_meta: dict, R_local: np.ndarray) -> dict[str, float]:
    """Inverse of `bone_local_rotation`: recover pose channel angles from R_local."""
    O = _rest_orientation_matrix(bone_meta)
    order = (bone_meta["rotation_order"] or "XYZ")[::-1]
    R_pose = O @ R_local @ O.T
    return _decompose_euler(order, R_pose)


def euler_rate_jacobian(bone_meta: dict, angles_xyz: dict[str, float]) -> np.ndarray:
    """3x3 matrix E (columns ordered x,y,z) mapping d(angles)/dt to Pinocchio's
    local (body-frame) angular velocity of `bone_local_rotation`'s R_local.

    Needed for active-set joint-limit handling: dropping a DOF at its limit
    is a DAZ XYZ-pose-channel concept, but the IK solve's Jacobian columns
    live in Pinocchio's native SO(3) tangent space (`nv`), not Euler-rate
    space. This is the standard Euler-rate-to-angular-velocity transform
    (omega = E(angles) @ d(euler)/dt), composed with the rest_orientation
    correction `bone_local_rotation` applies, so its columns land directly in
    the same tangent basis the IK solve's `dv` already uses.

    Derivation: for R_pose = R[order[0]](t1) @ R[order[1]](t2) @ R[order[2]](t3)
    (the sequential intrinsic product `_compose_euler` builds), the body
    angular velocity satisfying dR_pose/dt = R_pose @ skew(omega_pose) is
        omega_pose = (R2 R3)^T e1 * t1' + R3^T e2 * t2' + e3 * t3'
    (R2, R3 = the axis-rotation matrices for order[1], order[2]; e_i = unit
    vector along that step's axis). Since R_local = O^T @ R_pose @ O (O =
    rest_orientation), and R^T skew(v) R == skew(R^T v) for any rotation R,
    the corresponding body angular velocity of R_local is omega_local =
    O^T @ omega_pose -- i.e. E_local = O^T @ E_pose, columns re-ordered here
    to physical axis (x/y/z) rather than sequence position so callers can
    index by axis letter regardless of `rotation_order`.
    """
    order = (bone_meta["rotation_order"] or "XYZ")[::-1]
    a0, a1, a2 = order[0], order[1], order[2]
    R1 = _axis_rotation(a0, angles_xyz[a0.lower()])
    R2 = _axis_rotation(a1, angles_xyz[a1.lower()])
    R3 = _axis_rotation(a2, angles_xyz[a2.lower()])

    def _unit(axis: str) -> np.ndarray:
        return np.array([1.0, 0.0, 0.0]) if axis == "X" else (
            np.array([0.0, 1.0, 0.0]) if axis == "Y" else np.array([0.0, 0.0, 1.0])
        )

    col = {
        a0: (R2 @ R3).T @ _unit(a0),
        a1: R3.T @ _unit(a1),
        a2: _unit(a2),
    }
    E_pose = np.column_stack([col["X"], col["Y"], col["Z"]])

    O = _rest_orientation_matrix(bone_meta)
    return O.T @ E_pose


def axis_limit_active(current_angle: float, min_limit: float, max_limit: float,
                       proposed_rate: float, eps_deg: float = 1e-6) -> bool:
    """True if this axis should be dropped from the free-variable set this
    iteration: it's sitting at (or past) a limit and the proposed rate would
    push it further past that limit -- or the axis has no valid range at all
    (min == max, a hinge modeled as a full spherical joint with a dead axis;
    see `_CHAINS`' l_forearm/l_shoulder axis_limits in pose_transfer_photo.py).
    """
    if max_limit - min_limit <= eps_deg:
        return True
    if current_angle >= max_limit - eps_deg and proposed_rate > 0:
        return True
    if current_angle <= min_limit + eps_deg and proposed_rate < 0:
        return True
    return False


def _world_pos(bone_meta: dict) -> np.ndarray:
    w = bone_meta["world_position"]
    return np.array([w["x"], w["y"], w["z"]])


# ── kinematic model builder ─────────────────────────────────────────────────

@dataclass
class FigureModel:
    """A Pinocchio kinematic model for one subset of a figure's bones.

    Built once per figure (per unique chain set) from `bone_metadata()`, then
    reused across every photo run against that figure — the whole point of
    moving the solve off DAZ Studio's main thread.
    """

    model: pin.Model
    data: object
    chain_bones: list[str]
    by_name: dict[str, dict]
    joint_of: dict[str, int]


def build_figure_model(bone_metadata: list[dict], chain_bones: list[str]) -> FigureModel:
    """Build a Pinocchio model with one JointModelSpherical per bone in `chain_bones`.

    `chain_bones` may be a branching set (e.g. `pelvis` shared by both leg
    chains) — bones are wired to their actual parent *within this set* via
    `parent_name`; a bone whose parent isn't in the set is rooted directly to
    the universe frame at its own rest world position (this is exactly how
    `_CHAINS` in pose_transfer_photo.py roots hand chains at `l_shoulder`/
    `r_shoulder` and leg chains at `pelvis` — those roots have no further
    ancestor inside the solve).
    """
    by_name = {b["name"]: b for b in bone_metadata}
    missing = [name for name in chain_bones if name not in by_name]
    if missing:
        raise KeyError(f"Bones not found in bone_metadata(): {missing}")

    # Topologically order chain_bones so a parent is always added before its child.
    remaining = list(chain_bones)
    ordered: list[str] = []
    while remaining:
        progressed = False
        for name in list(remaining):
            parent = by_name[name]["parent_name"]
            if parent is None or parent not in chain_bones or parent in ordered:
                ordered.append(name)
                remaining.remove(name)
                progressed = True
        if not progressed:
            raise ValueError(f"Cyclic or unresolvable chain parentage among: {remaining}")

    model = pin.Model()
    joint_of: dict[str, int] = {}
    for name in ordered:
        meta = by_name[name]
        parent = meta["parent_name"]
        wpos = _world_pos(meta)
        if parent is not None and parent in joint_of:
            parent_id = joint_of[parent]
            parent_wpos = _world_pos(by_name[parent])
        else:
            parent_id = 0  # universe
            parent_wpos = np.zeros(3)
        placement = pin.SE3(np.eye(3), wpos - parent_wpos)
        joint_id = model.addJoint(parent_id, pin.JointModelSpherical(), placement, name)
        model.appendBodyToJoint(joint_id, pin.Inertia.Zero(), pin.SE3.Identity())
        joint_of[name] = joint_id

    data = model.createData()
    return FigureModel(model=model, data=data, chain_bones=chain_bones, by_name=by_name, joint_of=joint_of)


# ── configuration <-> DAZ pose-channel angles ───────────────────────────────

def configuration_from_angles(fm: FigureModel, angles: dict[str, dict[str, float]]) -> np.ndarray:
    q = pin.neutral(fm.model)
    for name in fm.chain_bones:
        meta = fm.by_name[name]
        idx_q = fm.model.joints[fm.joint_of[name]].idx_q
        R_local = bone_local_rotation(meta, angles[name])
        quat = pin.Quaternion(R_local)
        q[idx_q:idx_q + 4] = [quat.x, quat.y, quat.z, quat.w]
    return q


def clamp_angles(fm: FigureModel, angles: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    """Clamp every bone's angles into its `bone_metadata()` axis_limits.

    The finite-difference-based HTTP solver in `pose_transfer_photo.py`
    accumulates deltas in Python without ever re-reading DAZ Studio's
    post-`setValue()` (possibly clamped) angle back, so a target beyond a
    joint's limit silently desyncs the two. Since `bone_metadata()` already
    carries `axis_limits`, this local solver can do better: clamp every
    iteration so the model's own notion of "current angles" never drifts
    from what DAZ Studio will actually apply.
    """
    clamped: dict[str, dict[str, float]] = {}
    for name, a in angles.items():
        limits = fm.by_name[name]["axis_limits"]
        clamped[name] = {
            axis: float(np.clip(value, limits[axis]["min"], limits[axis]["max"]))
            for axis, value in a.items()
        }
    return clamped


def angles_from_configuration(fm: FigureModel, q: np.ndarray) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for name in fm.chain_bones:
        meta = fm.by_name[name]
        idx_q = fm.model.joints[fm.joint_of[name]].idx_q
        x, y, z, w = q[idx_q:idx_q + 4]
        R_local = pin.Quaternion(w, x, y, z).toRotationMatrix()
        result[name] = angles_from_local_rotation(meta, R_local)
    return result


def forward_kinematics_positions(fm: FigureModel, q: np.ndarray) -> dict[str, np.ndarray]:
    pin.forwardKinematics(fm.model, fm.data, q)
    return {name: fm.data.oMi[jid].translation.copy() for name, jid in fm.joint_of.items()}


# ── stacked multi-effector IK, fully local ──────────────────────────────────

def solve_ik(
    fm: FigureModel,
    effector_bones: list[str],
    target_points: np.ndarray,  # (E, 3)
    *,
    initial_angles: dict[str, dict[str, float]],
    max_iterations: int = 200,
    tolerance: float = 0.15,
    damping: float = 0.1,
    rest_pose_weight: float = 0.15,
    max_step_degrees: float = 1.0,
    debug: bool = False,
    history: list[dict[str, dict[str, float]]] | None = None,
) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    """Solve `effector_bones` toward `target_points` simultaneously, entirely locally.

    Same DLS + null-space rest-pose-bias formulation as
    `pose_transfer_photo.solve_stacked_ik`, but every iteration evaluates
    Pinocchio's analytic Jacobian against the local model instead of an HTTP
    finite-difference call. Only the caller pushing `angles_from_configuration`'s
    result to DAZ Studio (one `set_bone_rotations()` call) touches the network.

    Returns `(solved_angles, final_error)` — `solved_angles` matches
    `set_bone_rotations()`'s expected shape once converted to tuples, keyed by
    every bone in `fm.chain_bones`; `final_error` is computed against the
    *clamped* result, so it always agrees with what pushing `solved_angles` to
    DAZ Studio will actually produce.

    Joint limits are enforced during iteration via an active-set method, not
    just clamped once at the end: each iteration, any DOF currently at its
    `axis_limits` boundary (or with no valid range at all, e.g. a hinge's
    dead axis) *and* being pushed further past it is dropped from the
    free-variable set for that step. This has to happen in Euler-rate space
    (see `euler_rate_jacobian`), not Pinocchio's native SO(3) tangent space,
    since "which axis is at its limit" is a DAZ XYZ-pose-channel concept.
    `solve_stacked_ik`'s HTTP-based Jacobian got this for free (perturbing a
    bone already at its limit via DAZ's own `ctrl.setValue()` barely moves
    it, so that Jacobian column reads near zero); Pinocchio's analytic
    Jacobian has no such feedback, so without this masking a target near a
    joint's limit converges beautifully in the unconstrained model and then
    gets pulled substantially off target by the final clamp (see bd
    daz-script-server-hewu).
    """
    model, data = fm.model, fm.data
    q = configuration_from_angles(fm, initial_angles)
    q_rest = q.copy()
    effector_ids = [fm.joint_of[name] for name in effector_bones]
    n_v = model.nv
    n_err = len(effector_bones) * 3
    step_limit_rad = np.deg2rad(max_step_degrees)
    final_error = {name: float("inf") for name in effector_bones}

    for iteration in range(max_iterations):
        # A degenerate axis (min == max, a hinge modeled as a full spherical
        # joint) has exactly one legal value, so unlike a real-ranged limit
        # -- where snapping every iteration corrupts real rotation content
        # apportioned across axes by Euler decomposition, see this
        # function's docstring on why the final clamp isn't done mid-loop --
        # there's no ambiguity to lose here. Left uncorrected, the masked
        # DOF's instantaneous rate is exactly zero (to first order) but a
        # finite integration step still leaks a small curvature-driven
        # rotation into it after Euler re-decomposition each iteration, and
        # since that leakage is never reset, it compounds monotonically over
        # many iterations. Resync it to its exact legal value before this
        # iteration's FK/Jacobian so the leak can't accumulate.
        current_angles = angles_from_configuration(fm, q)
        resynced = False
        for name in fm.chain_bones:
            limits = fm.by_name[name]["axis_limits"]
            for axis in "xyz":
                lo, hi = limits[axis]["min"], limits[axis]["max"]
                if hi - lo <= 1e-6 and current_angles[name][axis] != lo:
                    current_angles[name][axis] = lo
                    resynced = True
        if resynced:
            q = configuration_from_angles(fm, current_angles)

        if history is not None:
            history.append(current_angles)

        pin.forwardKinematics(model, data, q)
        pin.computeJointJacobians(model, data, q)

        base = np.array([data.oMi[jid].translation for jid in effector_ids])
        error_vec = target_points - base
        for i, name in enumerate(effector_bones):
            final_error[name] = float(np.linalg.norm(error_vec[i]))
        if debug:
            errs = "  ".join(f"{n}={final_error[n]:.4f}" for n in effector_bones)
            print(f"    iter {iteration + 1:3d}/{max_iterations}: {errs}")
        if all(e <= tolerance for e in final_error.values()):
            break

        J = np.vstack([
            pin.getJointJacobian(model, data, jid, pin.LOCAL_WORLD_ALIGNED)[:3, :]
            for jid in effector_ids
        ])  # (E*3, nv)
        flat_error = error_vec.flatten()

        # Reparametrize the free variable from Pinocchio's native tangent
        # space (v, per-joint body-frame angular velocity) to Euler-rate
        # space (u, per-joint d(angle)/dt in x/y/z), via the block-diagonal
        # v = B @ u. This is what lets "axis at its limit" -- an XYZ-channel
        # concept -- mask out one specific column instead of an opaque
        # combination of all three.
        B = np.zeros((n_v, n_v))
        for name in fm.chain_bones:
            meta = fm.by_name[name]
            idx_v = model.joints[fm.joint_of[name]].idx_v
            B[idx_v:idx_v + 3, idx_v:idx_v + 3] = euler_rate_jacobian(meta, current_angles[name])
        J_u = J @ B  # (E*3, nv), columns now indexed by (bone, axis in x/y/z)

        # First pass: solve unconstrained in u-space to get a proposed rate
        # per axis, purely to decide which axes are trying to leave their
        # limits this step.
        JuJut = J_u @ J_u.T + damping * np.eye(n_err)
        try:
            JuJut_inv = np.linalg.inv(JuJut)
        except np.linalg.LinAlgError:
            break
        u_proposed = J_u.T @ JuJut_inv @ flat_error

        free_mask = np.ones(n_v, dtype=bool)
        for name in fm.chain_bones:
            meta = fm.by_name[name]
            idx_v = model.joints[fm.joint_of[name]].idx_v
            limits = meta["axis_limits"]
            for offset, axis in enumerate("xyz"):
                free_mask[idx_v + offset] = not axis_limit_active(
                    current_angle=current_angles[name][axis],
                    min_limit=limits[axis]["min"],
                    max_limit=limits[axis]["max"],
                    proposed_rate=u_proposed[idx_v + offset],
                )

        J_u_masked = J_u * free_mask  # zero the locked columns
        JuJut = J_u_masked @ J_u_masked.T + damping * np.eye(n_err)
        try:
            JuJut_inv = np.linalg.inv(JuJut)
        except np.linalg.LinAlgError:
            break
        J_u_pinv = J_u_masked.T @ JuJut_inv
        u = J_u_pinv @ flat_error
        u[~free_mask] = 0.0

        if rest_pose_weight > 0:
            null_space = np.eye(n_v) - J_u_pinv @ J_u_masked
            rest_pull_v = pin.difference(model, q, q_rest)  # q_rest (-) q, tangent space
            rest_pull_u = np.linalg.solve(B, rest_pull_v)
            u = u + null_space @ (rest_pose_weight * rest_pull_u)
            u[~free_mask] = 0.0

        dv = B @ u

        max_step = float(np.abs(dv).max()) if n_v else 0.0
        if max_step > step_limit_rad:
            dv *= step_limit_rad / max_step

        q = pin.integrate(model, q, dv)

    # Clamp only the final result, not every iteration: a bone with a locked
    # or narrow axis (e.g. min==max==0, common for a hinge-like joint modeled
    # as a full spherical joint) can have a nonzero value land on that axis
    # from Euler decomposition alone -- decomposition apportions rotation
    # across axes non-uniquely, so forcing one axis back to a limit mid-loop
    # and re-encoding discards real rotation content the other two axes had
    # already absorbed, which compounds every iteration and diverges the
    # solve. Clamping once at the end keeps the pushed angles within what
    # DAZ Studio will actually accept without corrupting the search itself.
    solved_angles = clamp_angles(fm, angles_from_configuration(fm, q))

    # Recompute final_error against the clamped configuration actually being
    # returned -- clamping can move the effectors, so reporting the pre-clamp
    # error here would silently disagree with what `solved_angles` produces.
    q_clamped = configuration_from_angles(fm, solved_angles)
    clamped_positions = forward_kinematics_positions(fm, q_clamped)
    final_error = {
        name: float(np.linalg.norm(target_points[i] - clamped_positions[name]))
        for i, name in enumerate(effector_bones)
    }
    return solved_angles, final_error
