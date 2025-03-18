import torch
import numpy as np
import cv2  # For capturing images from the camera
from main import MyModel  # Import your model class
from PIL import Image  # For converting to PIL format

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MyModel()
model.load_state_dict(torch.load("fingermodel.pth", map_location=device))
model.to(device)
model.eval()  # Set to evaluation mode

# Initialize the camera (use 0 for the default camera)
cap = cv2.VideoCapture(0)

# Capture a single frame from the camera
ret, frame = cap.read()
if not ret:
    print("Failed to capture image from camera.")
    cap.release()
    exit()

# Convert the image to grayscale
grey_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Resize the image to 128x128
resized_image = cv2.resize(grey_image, (128, 128))

# Convert the resized image to a PIL Image and then to a tensor
image_pil = Image.fromarray(resized_image)
image_tensor = torch.tensor(np.array(image_pil), dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)  # Add batch and channel dimensions

# Normalize the image (if needed, adjust based on your training preprocessing)
image_tensor = image_tensor / 255.0  # Scale the image to [0, 1] if it was in [0, 255]

# Run inference
with torch.no_grad():
    output = model(image_tensor)
    probabilities = torch.softmax(output, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1)

print(f"Predicted Class: {predicted_class.item()}")

# Release the camera and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()
