import cv2
import numpy as np
import mediapipe as mp  # type: ignore


# ---------- Image Enhancement for pixelated inputs ----------
def enhance_image(image):
    """Denoise, sharpen, and upscale the input image."""
    # Denoise
    image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

    # Sharpen
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    image = cv2.filter2D(image, -1, kernel)

    # Slightly upscale to help MediaPipe
    image = cv2.resize(image, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)

    return image


# ---------- Mediapipe Pose Setup ----------
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    static_image_mode=True,
    model_complexity=2,  # Increase complexity for better accuracy
    enable_segmentation=False,
    min_detection_confidence=0.3,  # Lower threshold to accept weak detections
)


# ---------- Pose Detection Function ----------
def detect_pose_mediapipe(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("❌ Could not read the image file.")
        return

    image = enhance_image(image)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if not results.pose_landmarks:
        print("⚠️ No pose detected — try YOLO version or check image quality.")
        return

    annotated = image.copy()
    mp_drawing.draw_landmarks(
        annotated,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
        mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
    )
    cv2.imwrite
    cv2.imshow("MediaPipe Pose Detection", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ---------- Run ----------
if __name__ == "__main__":
    detect_pose_mediapipe("../test/2.jpg")
