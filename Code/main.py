from ultralytics  import YOLO
import base64
import numpy as np
import cv2
import base64_to_image
import image_to_base64

# Run detection
if _name_ == "_main_":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Image prediction')
    parser.add_argument('-p', '--path', type=str, help='Path to the image file', required=True)
    args = parser.parse_args()

    img = base64_to_image(args.path)

    model = YOLO('best.pt')
    results = model(img)

    # Visualize output
    xyxys = []
    confidences = []
    class_ids = []
    # img = cv2.imread(img)
    for result in results:
        boxes = result.boxes.cpu().numpy()
        xyxy = boxes.xyxy
        for x in xyxy:
            cv2.rectangle(img, (int(x[0]), int(x[1])), (int(x[2]), int(x[3])), (0,255,0))

# Convert to base64
output = image_to_base64(img)

# Output result as base64 string
print(output)
