import os
from dotenv import load_dotenv
import streamlit as st
import requests
from PIL import Image, ImageDraw, ImageFont, ExifTags

load_dotenv()

MODEL = os.getenv("MODEL")
KEY = os.getenv("YOLO_KEY")

# LETTUCE_MODEL = os.getenv("LETTUCE_MODEL")
# WEED_MODEL = os.getenv("WEED_MODEL")
# KEY = os.getenv("YOLO_KEY")


# Color schemes for health status
CLASS_COLORS = {
    "normal_lettuce": "blue",
    "disease_lettuce": "red",
    "weed": "yellow"
}

BGR_COLORS = {
    "blue": (255, 0, 0),
    "red": (0, 0, 255),
    "yellow": (0, 255, 255),
}

def _make_api_request(image_path, confidence_threshold, retina_masks=False):
    """Common API request function for both models"""
    url = "https://predict.ultralytics.com"
    headers = {"x-api-key": KEY}
    data = {
        "model": MODEL,
        "imgsz": 640,
        "conf": confidence_threshold,
        "iou": 0.45
    }
    
    if retina_masks:
        data["retina_masks"] = True
    
    try:
        with open(image_path, "rb") as f:
            response = requests.post(url, headers=headers, data=data, files={"file": f})
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API request failed: {e}")
        return None
    
def _display_legend(legend_items):
    """Helper function to display the color legend"""
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

def _setup_font(img_width, img_height):
    """Helper function to setup font based on image size"""
    base_font_size = int(min(img_width, img_height) * 0.02)
    base_font_size = max(6, min(base_font_size, 16))
    
    try:
        if os.name == 'nt':  # Windows
            font = ImageFont.truetype("arial.ttf", base_font_size)
        else:  # Linux/Mac
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", base_font_size)
    except IOError:
        font = ImageFont.load_default()
        st.warning("Using default font as system font not found")
    
    return font

def _process_image_common(image_path):
    # Check 1: Verify file exists
    if not os.path.exists(image_path):
        st.error(f"Image file not found: {image_path}")
        return None

    try:
        # Check 2: Verify image can be opened and is in correct format
        img = Image.open(image_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
            img.save(image_path)
        return img
    except Exception as e:
        st.error(f"Error reading image: {e}")
        return None

def process_static_image_box(image_path):
    """Process a static image with bounding boxes for both lettuce and weed detection"""
    try:
        confidence_threshold = st.session_state.get('confidence_threshold', 0.25)

        img = _process_image_common(image_path)
        if img is None:
            return
        
        results = _make_api_request(image_path, confidence_threshold)
        if results is None:
            return

        predicted_image = Image.open(image_path)
        draw = ImageDraw.Draw(predicted_image)
        
        img_width, img_height = predicted_image.size
        font = _setup_font(img_width, img_height)
        line_thickness = max(1, min(int(min(img_width, img_height) * 0.003), 8))
        
        legend_items = {}

        for detection in results["images"][0]["results"]:
            class_name = detection.get("name", "Unknown")
            confidence = detection.get("confidence", 0.0)
            box = detection.get("box", {})
            
            x1 = float(box.get("x1", 0))
            y1 = float(box.get("y1", 0))
            x2 = float(box.get("x2", 0))
            y2 = float(box.get("y2", 0))
            
            x1, x2 = max(0, min(x1, img_width)), max(0, min(x2, img_width))
            y1, y2 = max(0, min(y1, img_height)), max(0, min(y2, img_height))

            color_name = CLASS_COLORS.get(class_name)
            base_color = BGR_COLORS.get(color_name, (0, 255, 0))
            rgb_color = (base_color[2], base_color[1], base_color[0])
            
            for i in range(line_thickness):
                draw.rectangle(
                    [(x1+i, y1+i), (x2-i, y2-i)],
                    outline=rgb_color
                )
            
            if class_name not in legend_items:
                legend_items[class_name] = rgb_color
        
        _display_legend(legend_items)
        st.image(predicted_image, caption="Processed Image with Detections", use_container_width=True)
        
        return results

    except Exception as e:
        st.error(f"Unexpected error: {e}")
        return None
    
def process_static_image_segment(image_path):
    """Process a static image with segmentation for both lettuce and weed detection"""
    try:
        confidence_threshold = st.session_state.get('confidence_threshold', 0.25)
        
        img = _process_image_common(image_path)
        if img is None:
            return
        
        results = _make_api_request(image_path, confidence_threshold, retina_masks=True)
        if results is None:
            return

        predicted_image = Image.open(image_path)
        draw = ImageDraw.Draw(predicted_image, 'RGBA')
        
        img_width, img_height = predicted_image.size
        legend_items = {}
        
        for detection in results["images"][0]["results"]:
            confidence = detection.get("confidence", 0.0)
            
            if confidence >= confidence_threshold:
                class_name = detection.get("name", "Unknown")
                segments = detection.get("segments", {})
                
                color_name = CLASS_COLORS.get(class_name)
                base_color = BGR_COLORS.get(color_name, (0, 255, 0))
                mask_color = (base_color[2], base_color[1], base_color[0], 127)
                
                if segments and "x" in segments and "y" in segments:
                    x_coords = segments["x"]
                    y_coords = segments["y"]
                    points = list(zip(x_coords, y_coords))
                    points = [(min(max(x, 0), img_width), min(max(y, 0), img_height)) 
                             for x, y in points]
                    
                    if len(points) > 2:
                        draw.polygon(points, fill=mask_color)
                
                if class_name not in legend_items:
                    legend_items[class_name] = (base_color[2], base_color[1], base_color[0])
        
        _display_legend(legend_items)
        st.image(predicted_image, caption="Processed Image with Segmentation", use_container_width=True)
        
        return results

    except Exception as e:
        st.error(f"Unexpected error: {e}")
        return None