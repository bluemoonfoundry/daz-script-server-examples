"""MediaPipe FaceLandmarker blendshape extraction for pose_transfer_photo.py's
--expression-image flag. Mirrors hand_landmarks.py's pattern (auto-downloaded
model, Tasks API) but returns ARKit-style blendshape *scores* directly from
output_face_blendshapes, rather than raw landmark geometry -- see
expression_mapping.py for how these scores map onto Genesis 9's facs_bs_*
morphs.
"""
from __future__ import annotations

import os
import sys
import urllib.request

import cv2
import mediapipe as mp

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_landmarker.task")


def _ensure_model() -> str:
    if not os.path.exists(_MODEL_PATH):
        print(f"Downloading face landmarker model -> {_MODEL_PATH}")
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
    return _MODEL_PATH


def extract_blendshapes(image_path: str) -> dict[str, float]:
    """Decode an image and return MediaPipe FaceLandmarker's 52 ARKit-style
    blendshape scores (0-1) for the first detected face, keyed by category
    name (e.g. "jawOpen", "mouthSmileLeft").

    Raises SystemExit if the image cannot be loaded or no face is detected.
    Unlike extract_hand_world_landmarks (where a missing hand in an otherwise
    good photo is normal and just means "skip this hand"), an
    --expression-image with no visible face is a real configuration error --
    same severity as extract_world_landmarks' "no pose detected" in the body
    pipeline, not a per-side skip.
    """
    img = cv2.imread(image_path)
    if img is None:
        sys.exit(f"Cannot load image: {image_path!r}")
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    BaseOptions = mp.tasks.BaseOptions
    RunningMode = mp.tasks.vision.RunningMode

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=_ensure_model()),
        running_mode=RunningMode.IMAGE,
        num_faces=1,
        output_face_blendshapes=True,
    )

    with FaceLandmarker.create_from_options(options) as landmarker:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_image)

    if not result.face_blendshapes:
        sys.exit(f"No face detected in image: {image_path!r}")

    return {category.category_name: category.score for category in result.face_blendshapes[0]}
