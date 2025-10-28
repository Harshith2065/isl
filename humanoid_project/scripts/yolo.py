from ultralytics import YOLO
import cv2


def detect_pose_yolo(image_path):
    # Load YOLOv8 Pose model (smallest for speed; use 'yolov8m-pose.pt' for more accuracy)
    model = YOLO("yolov8s-pose.pt")

    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print("❌ Could not read the image file.")
        return

    # Run inference
    results = model(image)

    # Plot detections (draw skeletons)
    annotated = results[0].plot()

    cv2.imshow("YOLOv8 Pose Detection", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    detect_pose_yolo("../test/2.jpg")
