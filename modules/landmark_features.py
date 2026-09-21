"""Extract numerical landmark coordinates for Phase 4."""


def extract_landmark_features(hand_landmarks):
    """Flatten the MediaPipe hand landmarks into a 63-value feature vector.

    Each hand has 21 landmarks, and every landmark provides x, y, z values.
    Thus: 21 * 3 = 63 features.
    """
    if hand_landmarks is None:
        return None

    try:
        features = []
        for landmark in hand_landmarks:
            features.extend([landmark.x, landmark.y, landmark.z])

        if len(features) != 63:
            return None

        return features
    except (AttributeError, TypeError, ValueError):
        return None
