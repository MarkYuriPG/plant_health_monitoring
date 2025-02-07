import os
from dotenv import load_dotenv
import streamlit as st
import requests
import json
from PIL import Image, ImageDraw, ImageFont, ExifTags
import io

load_dotenv()

MODEL = os.getenv("MODEL")
KEY = os.getenv("YOLO_KEY")

# Title for the app
st.title("Plant Health Monitoring - Object Detection")
st.markdown("""
    **Plant Health Monitoring** is an AI-powered tool designed to help detect and identify issues in plants using object detection.
    The model classifies images into categories such as **Aloe Vera**, **Browning**, **Rot**, **Rust**, and **Weed**. 
    Simply upload an image, and our AI will analyze it and provide real-time results based on its findings.
""")

# Upload image
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Image handling
image_path = None
if uploaded_image:
    original_image = Image.open(uploaded_image)

    # 🔄 **Fix: Correct orientation using EXIF data**
    try:
        exif = original_image._getexif()
        if exif is not None:
            for tag, value in exif.items():
                if ExifTags.TAGS.get(tag) == 'Orientation':
                    orientation = value
                    if orientation == 3:
                        original_image = original_image.rotate(180, expand=True)
                    elif orientation == 6:
                        original_image = original_image.rotate(270, expand=True)
                    elif orientation == 8:
                        original_image = original_image.rotate(90, expand=True)
                    break
    except (AttributeError, KeyError, IndexError):
        pass  # If no EXIF data, skip rotation

    # Display the correctly oriented image
    st.image(original_image)

    # Extract image details
    image_name = uploaded_image.name.split('.')[0]
    image_extension = uploaded_image.name.split('.')[-1]
    image_path = f"{image_name}_predict.{image_extension}"

    # Save the corrected image before sending it for prediction
    original_image.save(image_path)

# Inference and Overlay
if image_path:
    with st.spinner("Please wait..."):
        url_yolov8 = "https://predict.ultralytics.com"
        headers = {"x-api-key": KEY}
        # data = {"model": MODEL, "imgsz": 640, "conf": 0.25, "iou": 0.45}
        data = {"model": "https://hub.ultralytics.com/models/slKiGV7SUnkhL16Sm5Ow", "imgsz": 640, "conf": 0.25, "iou": 0.45}
        
        try:
            # Send the image for inference
            with open(image_path, "rb") as f:
                response_v8 = requests.post(url_yolov8, headers=headers, data=data, files={"file": f})
            response_v8.raise_for_status()
            results_v8 = response_v8.json()

            # Load the image (ensure it's correctly rotated)
            predicted_image = Image.open(image_path)

            # 🔄 **Fix: Reapply EXIF correction for output**
            try:
                exif = predicted_image._getexif()
                if exif is not None:
                    for tag, value in exif.items():
                        if ExifTags.TAGS.get(tag) == 'Orientation':
                            orientation = value
                            if orientation == 3:
                                predicted_image = predicted_image.rotate(180, expand=True)
                            elif orientation == 6:
                                predicted_image = predicted_image.rotate(270, expand=True)
                            elif orientation == 8:
                                predicted_image = predicted_image.rotate(90, expand=True)
                            break
            except (AttributeError, KeyError, IndexError):
                pass  # If no EXIF data, skip rotation

            draw_v8 = ImageDraw.Draw(predicted_image)

            # Optional: Load a font for text
            try:
                font = ImageFont.truetype("arial.ttf", 20)
            except IOError:
                font = ImageFont.load_default()

            CLASS_COLORS = {
                "Snake Plant": "blue",
                "Browning": "orange",
                "Weed": "red"
            }

            def process_results(results, draw):
                plant_counts = {
                    "Snake Plant": 0,
                    "Browning": 0,
                    "Weed": 0
                }
                
                total_confidence = 0
                total_detections = 0

                for detection in results["images"][0]["results"]:
                    box = detection["box"]
                    x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
                    label = f'{detection["name"]} ({detection["confidence"]:.2f})'

                    plant_name = detection["name"]
                    if plant_name in plant_counts:
                        plant_counts[plant_name] += 1
                        total_confidence += detection["confidence"]
                        total_detections += 1

                    box_color = CLASS_COLORS.get(plant_name, "white")

                    # Draw rectangle and label on the image
                    draw.rectangle([(x1, y1), (x2, y2)], outline=box_color, width=3)
                    label_y = y1 - 15 if y1 - 15 > 0 else y2 + 5
                    draw.text((x1, label_y), label, fill="red", font=font)

            result_v8 = process_results(results_v8, draw_v8)

            # Display the corrected output image
            st.subheader("YOLOv8 Prediction")
            st.image(predicted_image)

            # Save the corrected image for download
            buffered = io.BytesIO()
            predicted_image.save(buffered, format="PNG")
            buffered.seek(0)

            # Add a download button
            st.download_button(
                label="Download Predicted Image",
                data=buffered,
                file_name=f"{image_name}_predict.{image_extension}",
                mime="image/png"
            )

        except Exception as e:
            st.error(f"Error: {e}")
