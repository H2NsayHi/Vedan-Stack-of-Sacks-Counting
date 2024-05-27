from ultralytics  import YOLO
import base64
import numpy as np
import cv2

# Input base64 string image
base64_string = '"data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD..."'

def base64_to_image(base64_string):
    # Convert string Base64 to binary
    image_data = base64.b64decode(base64_string)
    # Convert binary to NumPy array
    np_arr = np.frombuffer(image_data, np.uint8)
    # Convert NumPy array to image
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    return img

def image_to_base64(image):

    base64_encoded_data = base64.b64encode(image)
    # Convert Base64 byte to string
    base64_string = base64_encoded_data.decode('utf-8')
    
    return base64_string

img = base64_to_image(base64_string)

# Run detection
model = YOLO('best.pt')
results = model(img)

# Visualize output
xyxys = []
confidences = []
class_ids = []
#img = cv2.imread(img)
for result in results:
  boxes = result.boxes.cpu().numpy()
  xyxy = boxes.xyxy
  for x in xyxy:
    cv2.rectangle(img, (int(x[0]), int(x[1])), (int(x[2]), int(x[3])), (0,255,0))
      
# Convert to base64
output = image_to_base64(img)

# Output result as base64 string
print(output)
