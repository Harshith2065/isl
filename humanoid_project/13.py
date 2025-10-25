import cv2
import mediapipe as mp  # type: ignore
import numpy as np

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# BODY25 mapping (approximation from Mediapipe 33)
BODY25_MAP = [
    0,  # Nose
    1,  # Neck (approximated)
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
    24,  # Hip duplicates
]


def angle_between_points(a, b, c):
    """Compute the angle at point b formed by points a-b-c in radians."""
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return np.arccos(np.clip(cos_angle, -1.0, 1.0))


def skeleton_to_joint_angles(skeleton):
    """Convert a single skeleton [25,3] to humanoid initial pose vector."""
    theta = []

    # Arms
    theta.append(
        angle_between_points(skeleton[11, :2], skeleton[13, :2], skeleton[15, :2])
    )  # Left arm
    theta.append(
        angle_between_points(skeleton[12, :2], skeleton[14, :2], skeleton[16, :2])
    )  # Right arm

    # Legs (approximated using available indices)
    theta.append(
        angle_between_points(skeleton[23, :2], skeleton[21, :2], skeleton[22, :2])
    )  # Left leg
    theta.append(
        angle_between_points(skeleton[24, :2], skeleton[20, :2], skeleton[21, :2])
    )  # Right leg

    # Torso/Neck
    shoulder_center = (skeleton[11, :2] + skeleton[12, :2]) / 2
    theta.append(
        angle_between_points(skeleton[1, :2], shoulder_center, skeleton[0, :2])
    )

    return np.array(theta)


def extract_keypoints(image_path):
    """Detect all people and return list of [25,3] keypoints + annotated image."""
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

        for i, idx in enumerate(BODY25_MAP[:25]):
            if idx < len(landmarks):
                lm = landmarks[idx]
                keypoints[i] = [lm.x * width, lm.y * height, lm.visibility]
            else:
                keypoints[i] = [0.0, 0.0, 0.0]

        results_all.append(keypoints)

        # Draw landmarks
        annotated_image = image_rgb.copy()
        mp_drawing.draw_landmarks(
            annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
        )

    return results_all, annotated_image


def select_main_skeleton(skeletons):
    """Select skeleton with largest bounding box area."""
    if len(skeletons) == 1:
        return skeletons[0]

    max_area = 0
    main_skel = skeletons[0]

    for sk in skeletons:
        x_min = np.min(sk[:, 0])
        x_max = np.max(sk[:, 0])
        y_min = np.min(sk[:, 1])
        y_max = np.max(sk[:, 1])
        area = (x_max - x_min) * (y_max - y_min)
        if area > max_area:
            max_area = area
            main_skel = sk

    return main_skel


if __name__ == "__main__":
    img_path = input("Enter image file path: ").strip()
    skeletons, annotated_img = extract_keypoints(img_path)

    if not skeletons:
        print("No skeletons detected in the image!")
    else:
        selected_skeleton = select_main_skeleton(skeletons)
        theta_init = skeleton_to_joint_angles(selected_skeleton)

        # Print keypoints as float
        np.set_printoptions(suppress=True, precision=6)
        print("Initial Pose Vector (theta_init):\n", theta_init)

        # Display annotated image
        annotated_img_bgr = cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR)
        cv2.imshow("Annotated Pose", annotated_img_bgr)
        print("Press any key in the image window to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
