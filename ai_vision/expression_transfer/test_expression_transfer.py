from __future__ import annotations

import expression_transfer as et


def _face_at(cx: float, cy: float, face_h: float = 100.0) -> list[tuple[float, float]]:
    """Build a minimal 478-point landmark list with a controlled centroid/height.

    Only the indices sort_faces_row_major and _face_centroid actually read
    (FACE_TOP, FACE_BOT, and the general average) need to be meaningful;
    every other point is placed at the centroid so it doesn't skew the mean.
    """
    lm = [(cx, cy)] * 478
    lm[et.FACE_TOP] = (cx, cy - face_h / 2)
    lm[et.FACE_BOT] = (cx, cy + face_h / 2)
    return lm


def test_sort_faces_row_major_orders_left_to_right_top_to_bottom():
    top_left = _face_at(0, 0)
    top_right = _face_at(500, 10)
    bottom_left = _face_at(0, 400)
    bottom_right = _face_at(500, 410)

    shuffled = [bottom_right, top_left, bottom_left, top_right]
    ordered = et.sort_faces_row_major(shuffled)

    assert ordered == [top_left, top_right, bottom_left, bottom_right]


def test_sort_faces_row_major_is_noop_for_single_face():
    face = _face_at(0, 0)
    assert et.sort_faces_row_major([face]) == [face]


def test_au_slug_picks_top_two_active_aus_by_magnitude():
    aus = {
        "mouth_smile_l": 0.9,
        "mouth_smile_r": 0.85,
        "jaw_open": 0.5,
        "eye_blink_l": 0.05,
        "eye_blink_r": 0.02,
    }
    assert et.au_slug(aus) == "mouth_smile-jaw_open"


def test_au_slug_returns_neutral_when_nothing_active():
    aus = {"mouth_smile_l": 0.05, "jaw_open": 0.0}
    assert et.au_slug(aus) == "neutral"


def test_au_slug_collapses_left_right_pairs_using_stronger_side():
    aus = {"mouth_frown_l": 0.9, "mouth_frown_r": 0.2}
    assert et.au_slug(aus, top_n=1) == "mouth_frown"


def test_slugify_label_strips_spaces_and_punctuation():
    assert et.slugify_label("Genesis 9") == "Genesis9"
    assert et.slugify_label("Character #2 (test)") == "Character2test"
