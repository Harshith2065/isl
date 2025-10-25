import cv2
import mediapipe as mp  # type: ignore
import numpy as np

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Image path
IMAGE_PATH = "pose1.jpg"

# Read image
image = cv2.imread(IMAGE_PATH)
if image is None:
    raise FileNotFoundError(f"Image {IMAGE_PATH} not found!")

# Convert BGR to RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Initialize Pose model
with mp_pose.Pose(
    static_image_mode=True,
    model_complexity=2,
    enable_segmentation=False,
    min_detection_confidence=0.5,
) as pose:

    results = pose.process(image_rgb)

# Extract 33 keypoints from MediaPipe Pose
if not results.pose_landmarks:
    print("No person detected")
    skeletons = []
else:
    skeleton = []
    for lm in results.pose_landmarks.landmark:
        skeleton.append([lm.x, lm.y, lm.z, lm.visibility])
    skeleton = np.array(skeleton)  # Shape: [33, 4]
    # If you want 25 keypoints, select the 25 corresponding indices
    # Here we take a simple mapping (for BODY_25 approximation)
    body25_indices = [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
    ]
    skeleton_25 = skeleton[body25_indices, :3]  # Take x, y, z
    skeletons = [skeleton_25]  # List of 1 person

# Print output
print("Detected skeletons (each [25,3]):")
for s in skeletons:
    print(s.shape)
    print(s)

# Draw skeleton on image
if skeletons:
    annotated_image = image.copy()
    mp_drawing.draw_landmarks(
        annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
    )
    cv2.imwrite("output_pose.jpg", annotated_image)
    print("Annotated image saved as output_pose.jpg")
