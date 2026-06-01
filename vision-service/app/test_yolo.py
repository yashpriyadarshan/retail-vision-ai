from ultralytics import YOLO

print("Loading model...")
model = YOLO("models/yolov8n.pt")

print("Running Inference...")
results = model("images/images1.jpg")

print("Inference Complete!")

for result in results:

    boxes = result.boxes

    print(f"\nDetected {len(boxes)} objects\n")

    for box in boxes:

        class_id = int(box.cls[0])

        confidence = float(box.conf[0])

        class_name = model.names[class_id]

        print(
            f"Class Id: {class_id},\n"
            f"Class: {class_name},\n"
            f"Confidence: {confidence:.2f}"
        )

