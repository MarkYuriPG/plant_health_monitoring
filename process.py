import os
from dotenv import load_dotenv
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

# def process_frame(frame):
#     """Process frame through YOLO model"""
#     try:
#         # Save frame as temporary image
#         temp_path = "temp_frame.jpg"
#         cv2.imwrite(temp_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        
#         url_yolov8 = "https://predict.ultralytics.com"
#         headers = {"x-api-key": KEY}
#         data = {
#             "model": MODEL,
#             "imgsz": 640,
#             "conf": 0.25,
#             "iou": 0.45
#         }

#         # Send frame for inference
#         with open(temp_path, "rb") as f:
#             response = requests.post(url_yolov8, headers=headers, data=data, files={"file": f})
#         response.raise_for_status()
#         results = response.json()

#         # Convert frame to PIL Image
#         frame_pil = Image.fromarray(frame)
#         draw = ImageDraw.Draw(frame_pil)

#         try:
#             font = ImageFont.truetype("arial.ttf", 64)
#         except IOError:
#             try:
#                 font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 64)
#             except IOError:
#                 font = ImageFont.load_default()

#         # Draw predictions on frame
#         for detection in results["images"][0]["results"]:
#             box = detection["box"]
#             x1, y1, x2, y2 = map(int, [box["x1"], box["y1"], box["x2"], box["y2"]])
#             confidence = detection["confidence"]
#             class_name = detection["name"]
#             label = f'{detection["name"]} ({confidence:.2f})'

#             # Get color from global CLASS_COLORS and convert BGR to RGB
#             bgr_color = CLASS_COLORS.get(class_name, (255, 255, 255))
#             rgb_color = (bgr_color[2], bgr_color[1], bgr_color[0])
            
#             # Draw rectangle and label
#             draw.rectangle([(x1, y1), (x2, y2)], outline=rgb_color, width=3)
#             draw.text(
#                 (x1, y1 - 35),
#                 label,
#                 fill=rgb_color,
#                 font=font
#             )
            
#         frame = np.array(frame_pil)

#         # Clean up
#         if os.path.exists(temp_path):
#             os.remove(temp_path)
            
#     except Exception as e:
#         st.error(f"Error in processing: {e}")
    
#     return frame

def process_static_image(image_path):
    """Process a static image through YOLO model"""
    try:
        # Verify image exists
        if not os.path.exists(image_path):
            st.error(f"Image file not found: {image_path}")
            return

        # Verify image is readable
        try:
            img = Image.open(image_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
                img.save(image_path)
        except Exception as e:
            st.error(f"Error reading image: {e}")
            return

        url_yolov8 = "https://predict.ultralytics.com"
        headers = {"x-api-key": KEY}
        data = {"model": MODEL, "imgsz": 640, "conf": 0.25, "iou": 0.45}
        
        try:
            with open(image_path, "rb") as f:
                response = requests.post(url_yolov8, headers=headers, data=data, files={"file": f})
            response.raise_for_status()
            results = response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"API request failed: {e}")
            if hasattr(response, 'text'):
                st.error(f"API response: {response.text}")
            return

        # Process results and display
        try:
            predicted_image = Image.open(image_path)
            draw = ImageDraw.Draw(predicted_image)
            
            # Try to load font, with fallbacks
            try:
                font = ImageFont.truetype("arial.ttf", 32)  # Reduced font size
            except IOError:
                try:
                    # Try system fonts on different platforms
                    if os.name == 'nt':  # Windows
                        font = ImageFont.truetype("arial.ttf", 32)
                    else:  # Linux/Mac
                        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 32)
                except IOError:
                    font = ImageFont.load_default()
                    st.warning("Using default font as system font not found")
            
            # Get image dimensions
            img_width, img_height = predicted_image.size
            
            for detection in results["images"][0]["results"]:
                box = detection["box"]
                x1, y1, x2, y2 = map(int, [box["x1"], box["y1"], box["x2"], box["y2"]])
                
                # Ensure coordinates are within image bounds
                x1 = max(0, min(x1, img_width))
                y1 = max(0, min(y1, img_height))
                x2 = max(0, min(x2, img_width))
                y2 = max(0, min(y2, img_height))
                
                class_name = detection["name"]
                confidence = detection["confidence"]
                label = f'{class_name} ({confidence:.2f})'

                # Get color with fallback to white
                bgr_color = BGR_COLORS.get(CLASS_COLORS.get(class_name), (255, 255, 255))
                rgb_color = (bgr_color[2], bgr_color[1], bgr_color[0])
                
                # Draw rectangle
                draw.rectangle([(x1, y1), (x2, y2)], outline=rgb_color, width=2)
                
                # Calculate text size and position
                text_bbox = draw.textbbox((0, 0), label, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
                
                # Draw text background
                text_y = max(0, y1 - text_height - 4)
                draw.rectangle(
                    [(x1, text_y), (x1 + text_width, text_y + text_height)],
                    fill=(0, 0, 0)
                )
                
                # Draw text
                draw.text(
                    (x1, text_y),
                    label,
                    fill=(255, 255, 0),
                    font=font
                )
            
            # Display processed image
            st.image(predicted_image, caption="Processed Image", use_container_width=True)
            
            # Return detection results
            return results

        except Exception as e:
            st.error(f"Error processing detection results: {e}")
            return None

    except Exception as e:
        st.error(f"Unexpected error: {e}")
        return None