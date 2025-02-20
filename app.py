from dotenv import load_dotenv
import numpy as np
import streamlit as st
from PIL import Image, ExifTags
import process
import os

# Title for the app
st.title("Lettuce Health Monitoring")
st.markdown("""
    This tool monitor lettuce fields to detect:
    - **Healthy Lettuce**
    - **Diseased Lettuce**
    - **Weeds**
    
    Upload an image or take a picture to get real-time analysis of your field.
""")

# Input selection
st.subheader("Input Options")
input_option = st.radio("Choose input type", 
                       ("Upload Image", "Take a Picture"), 
                       key="input_radio")

with st.sidebar:
    st.subheader("Analysis Settings")
    
    # Visualization mode selector
    visualization_mode = st.radio(
        "Visualization Mode",
        ("Segmentation", "Bounding Box"),
        key="visualization_mode"
    )
    
    # Optional: Add confidence threshold
    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.25,
        step=0.05
    )
    st.session_state['confidence_threshold'] = confidence_threshold

def process_image(image_path):
    """Process image with selected visualization mode"""
    with st.spinner("Analyzing image..."):
        if visualization_mode == "Segmentation":
            process.process_static_image_segment(image_path)
        else:
            process.process_static_image_box(image_path)

def handle_image_orientation(image):
    """Fix image orientation based on EXIF data"""
    try:
        exif = image._getexif()
        if exif:
            for tag, value in exif.items():
                if ExifTags.TAGS.get(tag) == 'Orientation':
                    if value == 3:
                        image = image.rotate(180, expand=True)
                    elif value == 6:
                        image = image.rotate(270, expand=True)
                    elif value == 8:
                        image = image.rotate(90, expand=True)
                    break
    except (AttributeError, KeyError, IndexError):
        pass
    return image

if input_option == "Take a Picture":
    img_file_buffer = st.camera_input("Take a picture")
    
    if img_file_buffer:
        # Display original image
        image = Image.open(img_file_buffer)
        image = handle_image_orientation(image)
        st.image(image, caption="Captured Image", use_container_width=True)
        
        # Save and process
        temp_path = "temp_camera.jpg"
        image.save(temp_path)
        
        # Process image
        process_image(temp_path)
        
        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)

elif input_option == "Upload Image":
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    
    if uploaded_image:
        # Display original image
        image = Image.open(uploaded_image)
        image = handle_image_orientation(image)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        # Save and process
        temp_path = f"temp_upload_{uploaded_image.name}"
        image.save(temp_path)
        
        # Process image
        process_image(temp_path)
        
        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)
