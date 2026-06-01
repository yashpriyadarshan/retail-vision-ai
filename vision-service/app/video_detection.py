from ultralytics import YOLO
import cv2

print("Loading model...")
model = YOLO("models/yolov8n.pt")

print("Opening video...")
cap = cv2.VideoCapture("videos/CAM 3.mp4")

if not cap.isOpened():
    print("Failed to open video")
    exit()

while True:

    success, frame = cap.read()

    if not success:
        print("Video finished")
        break

    results = model.track(
        frame,
        persist=True,
        classes=[0],  # person only
        verbose=False
    )

    annotated_frame = results[0].plot()

    for result in results:
        boxes = result.boxes

        if boxes.id is None:
            continue

        track_ids = boxes.id.int().cpu().tolist()

        for box, track_id in zip(boxes, track_ids):

            x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()

            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            cv2.circle(
                annotated_frame,
                (center_x, center_y),
                5,
                (0, 0, 225),
                -1
            )

            print(
                f"ID = {track_id}, "
                f"Center = ({center_x}, {center_y})"
            )

    cv2.line(
        annotated_frame,
        (1550, 0),
        (600, frame.shape[0]),
        (0, 255, 0),
        2
    )

    cv2.imshow("RetailVision Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()