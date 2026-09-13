"""ARKit blendshape category -> Genesis 9 base-rig facs_bs_*/facs_ctrl_*
morph mapping for pose_transfer_photo.py's --expression-image flag.

Verified live against a Genesis 9 figure ("Jason Cross") on 2026-09-13: all
66 DAZ morph names referenced below were confirmed present via
figure.morphs() before this table was written (see
docs/superpowers/specs/2026-09-13-expression-capture-design.md in the
daz-script-server repo for the verification method, in case this table ever
needs re-checking against a different Genesis 9 install or generation).

DAZ's base rig splits several of MediaPipe's *bilateral* ARKit categories
(one score covering both sides, or upper+lower) into 2-4 separate
Left/Right and/or Upper/Lower morphs -- e.g. ARKit's single "mouthClose"
becomes facs_bs_MouthCloseUpperLeft/Right + facs_bs_MouthCloseLowerLeft/
Right. Applying the same ARKit score to every DAZ name in such a group is
the simplest faithful translation and matches how a symmetric real-world
expression actually looks on this rig.

Three ARKit categories have no Genesis 9 base-rig equivalent at all:
eyeWideLeft/Right (no upper-eyelid-raise blendshape) and tongueOut (the
tongue isn't blendshape-driven on this rig) -- see UNMAPPED_ARKIT_CATEGORIES.
"""
from __future__ import annotations

ARKIT_TO_DAZ_MORPH: dict[str, list[str]] = {
    "browDownLeft": ["facs_bs_BrowDownLeft"],
    "browDownRight": ["facs_bs_BrowDownRight"],
    "browInnerUp": ["facs_bs_BrowInnerUpLeft", "facs_bs_BrowInnerUpRight"],
    "browOuterUpLeft": ["facs_bs_BrowOuterUpLeft"],
    "browOuterUpRight": ["facs_bs_BrowOuterUpRight"],
    "cheekPuff": ["facs_bs_CheekPuffLeft", "facs_bs_CheekPuffRight"],
    "cheekSquintLeft": ["facs_bs_CheekSquintLeft"],
    "cheekSquintRight": ["facs_bs_CheekSquintRight"],
    "eyeBlinkLeft": ["facs_bs_EyeBlinkLeft"],
    "eyeBlinkRight": ["facs_bs_EyeBlinkRight"],
    "eyeLookDownLeft": ["facs_bs_EyeLookDownLeft"],
    "eyeLookDownRight": ["facs_bs_EyeLookDownRight"],
    "eyeLookInLeft": ["facs_bs_EyeLookInLeft"],
    "eyeLookInRight": ["facs_bs_EyeLookInRight"],
    "eyeLookOutLeft": ["facs_bs_EyeLookOutLeft"],
    "eyeLookOutRight": ["facs_bs_EyeLookOutRight"],
    "eyeLookUpLeft": ["facs_bs_EyeLookUpLeft"],
    "eyeLookUpRight": ["facs_bs_EyeLookUpRight"],
    "eyeSquintLeft": ["facs_bs_EyeSquintLeft"],
    "eyeSquintRight": ["facs_bs_EyeSquintRight"],
    "jawForward": ["facs_bs_JawForward"],
    "jawLeft": ["facs_bs_JawLeft"],
    "jawOpen": ["facs_bs_JawOpen"],
    "jawRight": ["facs_bs_JawRight"],
    "mouthClose": [
        "facs_bs_MouthCloseUpperLeft", "facs_bs_MouthCloseUpperRight",
        "facs_bs_MouthCloseLowerLeft", "facs_bs_MouthCloseLowerRight",
    ],
    "mouthDimpleLeft": ["facs_bs_MouthDimpleLeft"],
    "mouthDimpleRight": ["facs_bs_MouthDimpleRight"],
    "mouthFrownLeft": ["facs_bs_MouthFrownLeft"],
    "mouthFrownRight": ["facs_bs_MouthFrownRight"],
    "mouthFunnel": [
        "facs_bs_MouthFunnelUpperLeft", "facs_bs_MouthFunnelUpperRight",
        "facs_bs_MouthFunnelLowerLeft", "facs_bs_MouthFunnelLowerRight",
    ],
    "mouthLeft": ["facs_bs_MouthLeft"],
    "mouthLowerDownLeft": ["facs_bs_MouthLowerDownLeft"],
    "mouthLowerDownRight": ["facs_bs_MouthLowerDownRight"],
    "mouthPressLeft": ["facs_bs_MouthPressUpperLeft", "facs_bs_MouthPressLowerLeft"],
    "mouthPressRight": ["facs_bs_MouthPressUpperRight", "facs_bs_MouthPressLowerRight"],
    "mouthPucker": [
        "facs_bs_MouthPurseUpperLeft", "facs_bs_MouthPurseUpperRight",
        "facs_bs_MouthPurseLowerLeft", "facs_bs_MouthPurseLowerRight",
    ],
    "mouthRight": ["facs_bs_MouthRight"],
    "mouthRollLower": ["facs_bs_MouthRollLowerLeft", "facs_bs_MouthRollLowerRight"],
    "mouthRollUpper": ["facs_bs_MouthRollUpperLeft", "facs_bs_MouthRollUpperRight"],
    "mouthShrugLower": ["facs_bs_MouthShrugLowerLeft", "facs_bs_MouthShrugLowerRight"],
    "mouthShrugUpper": ["facs_bs_MouthShrugUpperLeft", "facs_bs_MouthShrugUpperRight"],
    "mouthSmileLeft": ["facs_bs_MouthSmileLeft"],
    "mouthSmileRight": ["facs_bs_MouthSmileRight"],
    "mouthStretchLeft": ["facs_bs_MouthStretchLeft"],
    "mouthStretchRight": ["facs_bs_MouthStretchRight"],
    "mouthUpperUpLeft": ["facs_bs_MouthUpperUpLeft"],
    "mouthUpperUpRight": ["facs_bs_MouthUpperUpRight"],
    "noseSneerLeft": ["facs_bs_NoseSneerLeft"],
    "noseSneerRight": ["facs_bs_NoseSneerRight"],
}

UNMAPPED_ARKIT_CATEGORIES = frozenset({"eyeWideLeft", "eyeWideRight", "tongueOut"})


def daz_morph_values(blendshapes: dict[str, float], scale: float = 1.0) -> dict[str, float]:
    """Convert MediaPipe blendshape scores to a DazSkeleton.set_morph_values() payload.

    Every DAZ morph named anywhere in ARKIT_TO_DAZ_MORPH is included, even at
    0.0 for a category absent from *blendshapes* -- this lets a caller apply
    a fresh expression variant with a single set_morph_values() call, with no
    separate "zero out the previous variant" pass and no need to track what
    an earlier variant touched.
    """
    values: dict[str, float] = {}
    for category, daz_names in ARKIT_TO_DAZ_MORPH.items():
        score = blendshapes.get(category, 0.0) * scale
        for daz_name in daz_names:
            values[daz_name] = score
    return values


def warn_unmapped_categories(blendshapes: dict[str, float], threshold: float = 0.1) -> list[str]:
    """Return UNMAPPED_ARKIT_CATEGORIES entries scored above *threshold* in
    *blendshapes*, sorted -- for the caller to warn about a source photo
    whose expression relies on a category this rig can't represent (e.g. a
    wide-eyed look, whose intensity is otherwise silently dropped).
    """
    return sorted(
        category for category in UNMAPPED_ARKIT_CATEGORIES
        if blendshapes.get(category, 0.0) > threshold
    )
