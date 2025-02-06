import torch
from ultralytics import YOLO
import torch.serialization

# Add safe globals for YOLOv8 DetectionModel
torch.serialization.add_safe_globals([YOLO])

def load_model(model_path):
    try:
        # Load the model with weights_only=True and safe globals added
        model = torch.load(model_path, weights_only=False)
        return model
    except Exception as ex:
        print(f"Error loading model: {ex}")
        return None

# Path to your YOLOv8 trained weights
model_path = r"C:/Users/Yuri/Projects/plant_identification/PHMv25/weights/best.pt"

# Load the model securely
model = load_model(model_path)

# Check if the model is loaded successfully
if model is not None:
    print("Model loaded successfully!")
else:
    print(f"Error: Could not load model from {model_path}")
