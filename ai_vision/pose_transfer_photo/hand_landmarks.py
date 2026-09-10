# hand_landmarks.py
"""MediaPipe HandLandmarker extraction for pose_transfer_photo.py's
--fingers flag. Mirrors extract_world_landmarks()'s pattern in
pose_transfer_photo.py (auto-downloaded model, Tasks API, world landmarks).
"""
from __future__ import annotations

import os
import sys
import urllib.request

import cv2
import mediapipe as mp

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")


def _ensure_model() -> str:
    if not os.path.exists(_MODEL_PATH):
        print(f"Downloading hand landmarker model -> {_MODEL_PATH}")
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
    return _MODEL_PATH


def extract_hand_world_landmarks(image_path: str) -> dict[str, list[tuple[float, float, float]]]:
    """Decode an image and return each detected hand's 21 metric-scale world
    landmarks, keyed by MediaPipe's own handedness label ("Left"/"Right" --
    per MediaPipe's documented default convention this is the subject's own
    anatomical hand, matching pose_transfer_photo.py's existing assumption
    for body landmarks; see Task 6's live validation note if a run's
    resulting hand poses look mirrored).

    A hand MediaPipe doesn't detect (occluded, out of frame, low confidence)
    is simply absent from the returned dict -- not an error, since the
    caller (pose_transfer_photo.py) treats a missing hand as "skip this
    hand's fingers," matching how a missed body limb is already handled.

    Raises SystemExit only if the image itself cannot be loaded (same
    failure mode as extract_world_landmarks) -- zero hands detected is not
    an error here, unlike the body pose case, since photos are commonly
    framed to show a face/body clearly but crop or blur one or both hands.
    """
    img = cv2.imread(image_path)
    if img is None:
        sys.exit(f"Cannot load image: {image_path!r}")
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    HandLandmarker        = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    BaseOptions           = mp.tasks.BaseOptions
    RunningMode           = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=_ensure_model()),
        running_mode=RunningMode.IMAGE,
        num_hands=2,
    )

    with HandLandmarker.create_from_options(options) as landmarker:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_image)

    hands: dict[str, list[tuple[float, float, float]]] = {}
    for handedness, world_landmarks in zip(result.handedness, result.hand_world_landmarks):
        label = handedness[0].category_name  # "Left" or "Right"
        hands[label] = [(lm.x, lm.y, lm.z) for lm in world_landmarks]
    return hands
