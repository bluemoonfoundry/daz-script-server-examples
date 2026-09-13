from __future__ import annotations

import expression_mapping as em

# The full 52-category ARKit blendshape set MediaPipe FaceLandmarker outputs.
ALL_ARKIT_CATEGORIES = [
    "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight",
    "cheekPuff", "cheekSquintLeft", "cheekSquintRight",
    "eyeBlinkLeft", "eyeBlinkRight", "eyeLookDownLeft", "eyeLookDownRight", "eyeLookInLeft",
    "eyeLookInRight", "eyeLookOutLeft", "eyeLookOutRight", "eyeLookUpLeft", "eyeLookUpRight",
    "eyeSquintLeft", "eyeSquintRight", "eyeWideLeft", "eyeWideRight",
    "jawForward", "jawLeft", "jawOpen", "jawRight",
    "mouthClose", "mouthDimpleLeft", "mouthDimpleRight", "mouthFrownLeft", "mouthFrownRight",
    "mouthFunnel", "mouthLeft", "mouthLowerDownLeft", "mouthLowerDownRight",
    "mouthPressLeft", "mouthPressRight", "mouthPucker", "mouthRight",
    "mouthRollLower", "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper",
    "mouthSmileLeft", "mouthSmileRight", "mouthStretchLeft", "mouthStretchRight",
    "mouthUpperUpLeft", "mouthUpperUpRight", "noseSneerLeft", "noseSneerRight", "tongueOut",
]


def test_full_arkit_set_has_52_categories():
    assert len(ALL_ARKIT_CATEGORIES) == 52


def test_mapping_plus_unmapped_covers_every_arkit_category_exactly_once():
    covered = set(em.ARKIT_TO_DAZ_MORPH) | em.UNMAPPED_ARKIT_CATEGORIES
    assert covered == set(ALL_ARKIT_CATEGORIES)
    assert not (set(em.ARKIT_TO_DAZ_MORPH) & em.UNMAPPED_ARKIT_CATEGORIES)


def test_unmapped_categories_are_exactly_the_three_documented_gaps():
    assert em.UNMAPPED_ARKIT_CATEGORIES == frozenset({"eyeWideLeft", "eyeWideRight", "tongueOut"})


def test_mapping_values_are_nonempty_lists_of_globally_unique_daz_names():
    seen: set[str] = set()
    for category, daz_names in em.ARKIT_TO_DAZ_MORPH.items():
        assert daz_names, f"{category} maps to an empty list"
        for name in daz_names:
            assert name.startswith(("facs_bs_", "facs_ctrl_")), f"{name} unexpected prefix"
            assert name not in seen, f"{name} mapped from more than one ARKit category"
            seen.add(name)


def test_daz_morph_values_zeros_every_mapped_morph_when_blendshapes_empty():
    values = em.daz_morph_values({})
    assert values["facs_bs_JawOpen"] == 0.0
    assert values["facs_bs_MouthSmileLeft"] == 0.0
    assert all(v == 0.0 for v in values.values())
    # every DAZ name referenced anywhere in the table must be present, zeroed
    all_daz_names = {n for names in em.ARKIT_TO_DAZ_MORPH.values() for n in names}
    assert set(values) == all_daz_names


def test_daz_morph_values_applies_score_and_scale():
    values = em.daz_morph_values({"jawOpen": 0.6}, scale=0.5)
    assert values["facs_bs_JawOpen"] == 0.3


def test_daz_morph_values_splits_bilateral_arkit_category_to_both_daz_morphs():
    values = em.daz_morph_values({"browInnerUp": 0.8})
    assert values["facs_bs_BrowInnerUpLeft"] == 0.8
    assert values["facs_bs_BrowInnerUpRight"] == 0.8


def test_daz_morph_values_ignores_unmapped_categories():
    values = em.daz_morph_values({"tongueOut": 1.0, "eyeWideLeft": 1.0})
    assert all(name.startswith(("facs_bs_", "facs_ctrl_")) for name in values)


def test_warn_unmapped_categories_reports_above_threshold_only_sorted():
    blendshapes = {"tongueOut": 0.5, "eyeWideLeft": 0.05, "eyeWideRight": 0.2}
    assert em.warn_unmapped_categories(blendshapes, threshold=0.1) == ["eyeWideRight", "tongueOut"]


def test_warn_unmapped_categories_empty_when_nothing_active():
    assert em.warn_unmapped_categories({}) == []
