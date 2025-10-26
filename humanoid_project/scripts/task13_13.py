"""Task 1.3: Convert detected skeletons to initial pose vector θ_init with 13 DOFs."""

import cv2
import mediapipe as mp  # type: ignore
import numpy as np

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# BODY25 indices mapping
BODY25_MAP = [
    0,  # Nose
    1,  # Neck
    11,  # Right shoulder
    12,  # Left shoulder
    23,  # Right hip
    24,  # Left hip
    13,  # Right elbow
    14,  # Left elbow
    15,  # Right wrist
    16,  # Left wrist
    5,  # Right eye
    6,  # Left eye
    7,  # Right ear
    8,  # Left ear
]


def extract_keypoints(image_path):
    """Detect skeletons in the image and return keypoints as [N,25,3]."""
    image = cv2.imread(image_path)
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
            return [], image_rgb

        landmarks = results.pose_landmarks.landmark
        keypoints = np.zeros((25, 3), dtype=float)

        for i, idx in enumerate(BODY25_MAP):
            if idx < len(landmarks):
                lm = landmarks[idx]
                keypoints[i] = [lm.x * width, lm.y * height, lm.visibility]
            else:
                keypoints[i] = [0.0, 0.0, 0.0]

        # Draw the landmarks on the image
        annotated_image = image_rgb.copy()
        mp_drawing.draw_landmarks(
            annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
        )

        results_all.append(keypoints)

    return results_all, annotated_image


def angle_between_points(a, b, c):
    """Return angle (in degrees) at point b formed by points a-b-c."""
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    return np.degrees(angle)


def select_main_skeleton(skeletons):
    """Select skeleton with largest bounding box area."""
    if len(skeletons) == 0:
        return None
    areas = []
    for skel in skeletons:
        xs = skel[:, 0]
        ys = skel[:, 1]
        bbox_area = (np.max(xs) - np.min(xs)) * (np.max(ys) - np.min(ys))
        areas.append(bbox_area)
    return skeletons[np.argmax(areas)]


def skeleton_to_joint_angles(skel):
    """Convert a single skeleton [25,3] to θ_init vector with extended DOFs."""
    theta_init = []

    # Right arm
    theta_init.append(
        angle_between_points(skel[11, :2], skel[13, :2], skel[15, :2])
    )  # Shoulder
    theta_init.append(
        angle_between_points(skel[13, :2], skel[15, :2], skel[15, :2] + [10, 0])
    )  # Elbow-wrist direction

    # Left arm
    theta_init.append(angle_between_points(skel[12, :2], skel[14, :2], skel[16, :2]))
    theta_init.append(
        angle_between_points(skel[14, :2], skel[16, :2], skel[16, :2] + [10, 0])
    )

    # Right leg
    theta_init.append(
        angle_between_points(skel[23, :2], skel[23, :2], skel[23, :2] + [0, 50])
    )  # Hip
    theta_init.append(
        angle_between_points(
            skel[23, :2], skel[23, :2] + [0, 50], skel[23, :2] + [0, 100]
        )
    )  # Knee

    # Left leg
    theta_init.append(
        angle_between_points(skel[24, :2], skel[24, :2], skel[24, :2] + [0, 50])
    )
    theta_init.append(
        angle_between_points(
            skel[24, :2], skel[24, :2] + [0, 50], skel[24, :2] + [0, 100]
        )
    )

    # Torso angles (Neck-Shoulder-Hip)
    theta_init.append(
        angle_between_points(skel[1, :2], skel[11, :2], skel[23, :2])
    )  # Right
    theta_init.append(
        angle_between_points(skel[1, :2], skel[12, :2], skel[24, :2])
    )  # Left

    # Neck angles
    theta_init.append(
        angle_between_points(skel[0, :2], skel[1, :2], skel[11, :2])
    )  # Right
    theta_init.append(
        angle_between_points(skel[0, :2], skel[1, :2], skel[12, :2])
    )  # Left

    # Optional: Eye/head orientation (simple 2D angle)
    theta_init.append(
        angle_between_points(skel[0, :2], skel[5, :2], skel[6, :2])
    )  # Eye angle

    return np.array(theta_init, dtype=float)


if __name__ == "__main__":
    img_path = input("Enter image file path: ").strip()
    skeletons, annotated_img = extract_keypoints(img_path)
    selected_skeleton = select_main_skeleton(skeletons)

    if selected_skeleton is None:
        print("No skeleton detected.")
    else:
        theta_init = skeleton_to_joint_angles(selected_skeleton)
        np.set_printoptions(suppress=True, precision=2)
        print("Initial Pose Vector θ_init:\n", theta_init)

        # Show annotated image
        cv2.imshow("Annotated Pose", cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
# 13joints
