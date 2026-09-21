"""Phase 3 and Phase 4 hand detection using the current MediaPipe Tasks API."""

import time
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    drawing_styles,
    drawing_utils,
)

MODEL_PATH = Path(__file__).resolve().parents[1] / "model" / "hand_landmarker.task"
OFFICIAL_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"


def initialize_hands(model_path=None):
    """Create and return the current MediaPipe Hand Landmarker."""
    target_path = Path(model_path) if model_path else MODEL_PATH

    if not target_path.exists():
        raise RuntimeError(
            "ERROR: Missing MediaPipe Hand Landmarker model file.\n"
            f"Place the official model here: {target_path}\n"
            f"Official download URL: {OFFICIAL_MODEL_URL}"
        )

    try:
        options = HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(target_path)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        return HandLandmarker.create_from_options(options)
    except Exception as exc:
        raise RuntimeError(f"ERROR: MediaPipe hand detection could not be initialized: {exc}") from exc


def process_hand_frame(frame, hand_landmarker):
    """Detect hands in a frame, draw landmarks, and return the detected landmark sets."""
    annotated_frame = frame.copy()
    detected_hands = []

    rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    timestamp_ms = time.time_ns() // 1_000_000
    result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)

    if result.hand_landmarks:
        hand_count = len(result.hand_landmarks)

        for hand_landmarks in result.hand_landmarks:
            detected_hands.append(hand_landmarks)
            drawing_utils.draw_landmarks(
                annotated_frame,
                hand_landmarks,
                mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS,
                landmark_drawing_spec=drawing_styles.get_default_hand_landmarks_style(),
                connection_drawing_spec=drawing_styles.get_default_hand_connections_style(),
            )

    return annotated_frame, detected_hands
