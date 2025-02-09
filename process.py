import os
import cv2
from dotenv import load_dotenv
import numpy as np
import streamlit as st
import requests
from PIL import Image, ImageDraw, ImageFont, ExifTags

load_dotenv()

MODEL = os.getenv("MODEL")
KEY = os.getenv("YOLO_KEY")

CLASS_COLORS = {
    "Snake Plant": "blue",
    "Browning": "orange",
    "Weed": "red"
}

BGR_COLORS = {
    "blue": (255, 0, 0),
    "red": (0, 0, 255),
    "orange": (0, 165, 255),
}

def process_frame(frame):
    """Process frame through YOLO model"""
    try:
        # Save frame as temporary image
        temp_path = "temp_frame.jpg"
        cv2.imwrite(temp_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        
        url_yolov8 = "https://predict.ultralytics.com"
        headers = {"x-api-key": KEY}
        data = {
            "model": MODEL,
            "imgsz": 640,
            "conf": 0.25,
            "iou": 0.45
        }

        # Send frame for inference
        with open(temp_path, "rb") as f:
            response = requests.post(url_yolov8, headers=headers, data=data, files={"file": f})
        response.raise_for_status()
        results = response.json()

        # Convert frame to PIL Image
        frame_pil = Image.fromarray(frame)
        draw = ImageDraw.Draw(frame_pil)

        try:
            font = ImageFont.truetype("arial.ttf", 64)
        except IOError:
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 64)
            except IOError:
                font = ImageFont.load_default()

        # Draw predictions on frame
        for detection in results["images"][0]["results"]:
            box = detection["box"]
            x1, y1, x2, y2 = map(int, [box["x1"], box["y1"], box["x2"], box["y2"]])
            confidence = detection["confidence"]
            class_name = detection["name"]
            label = f'{detection["name"]} ({confidence:.2f})'

            # Get color from global CLASS_COLORS and convert BGR to RGB
            bgr_color = CLASS_COLORS.get(class_name, (255, 255, 255))
            rgb_color = (bgr_color[2], bgr_color[1], bgr_color[0])
            
            # Draw rectangle and label
            draw.rectangle([(x1, y1), (x2, y2)], outline=rgb_color, width=3)
            draw.text(
                (x1, y1 - 35),
                label,
                fill=rgb_color,
                font=font
            )
            
        frame = np.array(frame_pil)

        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
    except Exception as e:
        st.error(f"Error in processing: {e}")
    
    return frame

def process_static_image(image_path):
    """Process a static image through YOLO model"""
    try:
        url_yolov8 = "https://predict.ultralytics.com"
        headers = {"x-api-key": KEY}
        data = {"model": MODEL, "imgsz": 640, "conf": 0.25, "iou": 0.45}
        
        with open(image_path, "rb") as f:
            response = requests.post(url_yolov8, headers=headers, data=data, files={"file": f})
        response.raise_for_status()
        results = response.json()
        
        # Process results and display
        predicted_image = Image.open(image_path)
        draw = ImageDraw.Draw(predicted_image)
        
        for detection in results["images"][0]["results"]:
            box = detection["box"]
            x1, y1, x2, y2 = map(int, [box["x1"], box["y1"], box["x2"], box["y2"]])
            class_name = detection["name"]
            label = f'{detection["name"]} ({detection["confidence"]:.2f})'

            bgr_color = BGR_COLORS.get(CLASS_COLORS.get(class_name), (255, 255, 255))
            rgb_color = (bgr_color[2], bgr_color[1], bgr_color[0])  # Convert BGR to RGB
            
            # Draw rectangle and label
            draw.rectangle([(x1, y1), (x2, y2)], outline=rgb_color, width=2)
            draw.text((x1, y1 - 10), label, fill=(255,255,0), font=ImageFont.truetype("arial.ttf", 64))
        
        st.image(predicted_image, caption="Processed Image")
        
    except Exception as e:
        st.error(f"Error in processing: {e}")