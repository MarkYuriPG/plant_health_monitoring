from ultralytics import YOLO
import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Load YOLOv8 model
best_pt = 'C:/Users/Yuri/Projects/plant_identification/weights/phmv2-1.pt'
model = YOLO(best_pt)

st.title("Plant Identification - PHM")

# Upload an image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Convert to OpenCV format
    image = Image.open(uploaded_file)
    image = np.array(image)
    
    # Perform inference
    results = model(image, conf=0.25)

    # Display uploaded image
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Display results
    for result in results:
        st.write(f"Detected: {result.names}")
