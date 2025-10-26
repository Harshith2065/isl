"""Task 1.2 _ Version1 : Extract 25-keypoint human poses from an image using Mediapipe."""

import cv2
import mediapipe as mp  # type: ignore
import numpy as np

# Set global print options to avoid scientific notation
np.set_printoptions(suppress=True, precision=6)

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Define a mapping from Mediapipe 33 → BODY25 indices (approximation)
BODY25_MAP = [
    0,  # Nose
    1,  # Neck (approximated using midpoint of shoulders)
    11,
    12,
    23,
    24,  # Shoulders, hips
    13,
    14,
    15,
    16,  # Elbows, wrists
    25,
    26,
    27,
    28,
    29,
    30,
    31,
    32,  # Legs, feet
    5,
    6,
    7,
    8,
    9,
    10,  # Eyes, ears
    23,
    24,  # Hip duplicates (to fill to 25)
]


def extract_keypoints(image_path):
    """Detects all people and returns list of [25,3] arrays (x, y, confidence)."""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width, _ = image.shape

    results_all = []

    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=False,
        min_detection_confidence=0.5,
    ) as pose:

        results = pose.process(image_rgb)
        if not results.pose_landmarks:
            return []

        landmarks = results.pose_landmarks.landmark
        keypoints = np.zeros((25, 3))

        for i, idx in enumerate(BODY25_MAP[:25]):
            if idx < len(landmarks):
                lm = landmarks[idx]
                keypoints[i] = [lm.x * width, lm.y * height, lm.visibility]
            else:
                keypoints[i] = [0, 0, 0]

        results_all.append(keypoints)

    return results_all


if __name__ == "__main__":
    img_path = "pose1.jpg"  # Update with your actual image path
    skeletons = extract_keypoints(img_path)
    print(f"Detected {len(skeletons)} skeleton(s)")
    for i, s in enumerate(skeletons):
        print(f"\nPerson {i+1} keypoints (x, y, conf):\n{s}")
