import os
from dotenv import load_dotenv
import streamlit as st
import requests
from PIL import Image, ImageDraw, ImageFont, ExifTags

load_dotenv()

MODEL = os.getenv("MODEL")
KEY = os.getenv("YOLO_KEY")

# Color schemes for health status
CLASS_COLORS = {
    "normal_lettuce": "blue",
    "disease_lettuce": "yellow",
}

BGR_COLORS = {
    "blue": (255, 0, 0),
    "yellow": (0, 255, 255),
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

def process_static_image_box(image_path):
    """Process a static image through YOLO model"""
    try:
        confidence_threshold = st.session_state.get('confidence_threshold', 0.25)

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

        url = "https://predict.ultralytics.com"
        headers = {"x-api-key": KEY}
        data = {"model": MODEL, "imgsz": 640, "conf": confidence_threshold, "iou": 0.45}
        
        try:
            with open(image_path, "rb") as f:
                response = requests.post(url, headers=headers, data=data, files={"file": f})
            response.raise_for_status()
            results = response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"API request failed: {e}")
            return

        # Process results and display
        try:
            predicted_image = Image.open(image_path)
            draw = ImageDraw.Draw(predicted_image)
            
            img_width, img_height = predicted_image.size
            
            # Calculate font size based on image dimensions
            # This will scale the font size relative to the image width
            base_font_size = int(min(img_width, img_height) * 0.02)  # 2% of the smaller dimension
            base_font_size = max(6, min(base_font_size, 16))  # Keep font size between 12 and 48
            
            # Setup font with calculated size
            try:
                if os.name == 'nt':  # Windows
                    font = ImageFont.truetype("arial.ttf", base_font_size)
                else:  # Linux/Mac
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", base_font_size)
            except IOError:
                font = ImageFont.load_default()
                st.warning("Using default font as system font not found")
            
            # Calculate line thickness based on image size
            line_thickness = max(1, min(int(min(img_width, img_height) * 0.003), 8))  # 0.3% of smaller dimension, between 1 and 8
            
            legend_items = {}

            for detection in results["images"][0]["results"]:
                # Extract detection information
                class_name = detection.get("name", "Unknown")
                confidence = detection.get("confidence", 0.0)
                box = detection.get("box", {})
                
                # Get bounding box coordinates
                x1 = float(box.get("x1", 0))
                y1 = float(box.get("y1", 0))
                x2 = float(box.get("x2", 0))
                y2 = float(box.get("y2", 0))
                
                # Ensure coordinates are within image bounds
                x1 = max(0, min(x1, img_width))
                y1 = max(0, min(y1, img_height))
                x2 = max(0, min(x2, img_width))
                y2 = max(0, min(y2, img_height))

                color_name = CLASS_COLORS.get(class_name)
                
                base_color = BGR_COLORS.get(color_name, (0, 255, 0))  # Default to green if no color found
                
                rgb_color = (base_color[2], base_color[1], base_color[0])
                
                # Get color for this class
                # base_color = BGR_COLORS.get(CLASS_COLORS.get(class_name), (0, 255, 0))  # Default to green
                # rgb_color = (base_color[2], base_color[1], base_color[0])

                # Draw thicker bounding box
                for i in range(line_thickness):
                    draw.rectangle(
                        [(x1+i, y1+i), (x2-i, y2-i)],
                        outline=rgb_color
                    )
                
                # Add to legend if not already there
                if class_name not in legend_items:
                    legend_items[class_name] = rgb_color
                
            st.sidebar.subheader("Color Legend")

            for class_name, color in legend_items.items():
                # Create colored box using HTML
                color_hex = '#{:02x}{:02x}{:02x}'.format(*color)
                st.sidebar.markdown(
                    f'<div style="display: flex; align-items: center;">'
                    f'<div style="width: 20px; height: 20px; background-color: {color_hex}; margin-right: 10px;"></div>'
                    f'<div>{class_name}</div>'
                    f'</div>',
                    unsafe_allow_html=True  
                )
            
            # Display processed image
            st.image(predicted_image, caption="Processed Image with Detections", use_container_width=True)
            
            return results

        except Exception as e:
            st.error(f"Error processing detection results: {e}")
            return None

    except Exception as e:
        st.error(f"Unexpected error: {e}")
        return None
    
def process_static_image_segment(image_path):
    """Process a static image through YOLO segmentation model"""
    try:
        confidence_threshold = st.session_state.get('confidence_threshold', 0.25)

        if not os.path.exists(image_path):
            st.error(f"Image file not found: {image_path}")
            return

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
        data = {
            "model": MODEL,
            "imgsz": 640,
            "conf": confidence_threshold,
            "iou": 0.45,
            "retina_masks": True
        }
        
        try:
            with open(image_path, "rb") as f:
                response = requests.post(url_yolov8, headers=headers, data=data, files={"file": f})
            response.raise_for_status()
            results = response.json()

        except requests.exceptions.RequestException as e:
            st.error(f"API request failed: {e}")
            return

        try:
            predicted_image = Image.open(image_path)
            draw = ImageDraw.Draw(predicted_image, 'RGBA')  # Enable alpha channel for transparency
            
            img_width, img_height = predicted_image.size
            legend_items = {}
            
            for detection in results["images"][0]["results"]:
                confidence = detection.get("confidence", 0.0)
                
                if confidence >= confidence_threshold:
                    class_name = detection.get("name", "Unknown")
                    segments = detection.get("segments", {})
                    
                    # Get color for this class
                    color_name = CLASS_COLORS.get(class_name)
                    base_color = BGR_COLORS.get(color_name, (0, 255, 0))
                    
                    # Create semi-transparent color for the mask
                    mask_color = (base_color[2], base_color[1], base_color[0], 127)  # RGBA with 50% transparency
                    
                    if segments and "x" in segments and "y" in segments:
                        # Create list of (x,y) points from the segments
                        x_coords = segments["x"]
                        y_coords = segments["y"]
                        
                        # Combine x,y coordinates into points
                        points = list(zip(x_coords, y_coords))
                        
                        # Ensure points are within image bounds
                        points = [(min(max(x, 0), img_width), min(max(y, 0), img_height)) 
                                for x, y in points]
                        
                        if len(points) > 2:  # Need at least 3 points to draw a polygon
                            draw.polygon(points, fill=mask_color)
                    
                    # Add to legend if not already there
                    if class_name not in legend_items:
                        legend_items[class_name] = (base_color[2], base_color[1], base_color[0])
            
            # Display the color legend
            st.sidebar.subheader("Color Legend")
            for class_name, color in legend_items.items():
                color_hex = '#{:02x}{:02x}{:02x}'.format(*color)
                st.sidebar.markdown(
                    f'<div style="display: flex; align-items: center;">'
                    f'<div style="width: 20px; height: 20px; background-color: {color_hex}; margin-right: 10px;"></div>'
                    f'<div>{class_name}</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            
            # Display processed image
            st.image(predicted_image, caption="Processed Image with Segmentation", use_container_width=True)
            
            return results

        except Exception as e:
            st.error(f"Error processing detection results: {e}")
            st.write("Error details:", str(e))
            return None

    except Exception as e:
        st.error(f"Unexpected error: {e}")
        return None
