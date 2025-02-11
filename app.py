from dotenv import load_dotenv
import numpy as np
import streamlit as st
from PIL import Image, ExifTags
import process

# Title for the app
st.title("Lettuce Health Monitoring - Segmentation Detection")
st.markdown("""
    **Lettuce Health Monitoring** is an AI-powered tool that analyzes lettuce plants to detect their health status.
    The model segments and classifies lettuce plants as either **Healthy** or **Unhealthy**. 
    Upload an image or take a picture to get real-time analysis of your lettuce plants.
""")

# Input selection
st.subheader("Input Options")
input_option = st.radio("Choose input type", 
                       ("Upload Image", "Take a Picture"), 
                       key="input_radio")

with st.sidebar:
    st.subheader("Detection Settings")
    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.25,
        step=0.05
    )
    # Add visualization mode selector
    visualization_mode = st.radio(
        "Visualization Mode",
        ("Segmentation", "Bounding Box"),
        key="visualization_mode"
    )
    # process_every_n_frames = st.slider(  # Define process_every_n_frames here
    #     "Process every N frames",
    #     min_value=1,
    #     max_value=30,
    #     value=10
    # )


# # Initialize variables for frame processing
# if 'frame_counter' not in st.session_state:
#     st.session_state.frame_counter = 0

# if input_option == "Live Video":
#     # Live video implementation
#     run = st.checkbox('Start Camera')
#     FRAME_WINDOW = st.image([])
    
#     try:
#         camera = cv2.VideoCapture(0)
        
#         while run:
#             ret, frame = camera.read()
#             if not ret:
#                 st.error("Failed to read from camera")
#                 break

#             # Convert BGR to RGB
#             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
#             # Process every Nth frame
#             if st.session_state.frame_counter % process_every_n_frames == 0:
#                 frame = process.process_frame(frame)
            
#             # Display the frame
#             FRAME_WINDOW.image(frame)
            
#             # Increment frame counter
#             st.session_state.frame_counter += 1
            
#         else:
#             # Display placeholder when stopped
#             placeholder_img = np.zeros((480, 640, 3), dtype=np.uint8)
#             cv2.putText(placeholder_img, 'Camera Stopped', (180, 240),
#                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
#             FRAME_WINDOW.image(placeholder_img)
#             st.write('Camera Stopped')
            
#         # Release camera when stopped
#         if not run:
#             camera.release()

#     except Exception as e:
#         st.error(f"Error with camera: {e}")
#         if 'camera' in locals():
#             camera.release()

if input_option == "Take a Picture":
    # Single photo capture
    img_file_buffer = st.camera_input("Take a picture")
    
    if img_file_buffer:
        # Your existing image processing code
        original_image = Image.open(img_file_buffer)
        
        # Fix orientation using EXIF data
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
            pass

        st.image(original_image)
        
        # Save and process image
        image_name = "captured_image"
        image_extension = "jpg"
        image_path = f"{image_name}_predict.{image_extension}"
        original_image.save(image_path)

        with st.spinner("Processing image..."):
            if visualization_mode == "Segmentation":
                process.process_static_image_segment(image_path)
            else:
                process.process_static_image_box(image_path)

elif input_option == "Upload Image":
    # Your existing upload image code
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    
    if uploaded_image:
        # Your existing image processing code
        original_image = Image.open(uploaded_image)
        
        # Fix orientation using EXIF data
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
            pass

        st.image(original_image)
        
        image_name = uploaded_image.name.split('.')[0]
        image_extension = uploaded_image.name.split('.')[-1]
        image_path = f"{image_name}_predict.{image_extension}"
        original_image.save(image_path)

        with st.spinner("Processing image..."):
            if visualization_mode == "Segmentation":
                process.process_static_image_segment(image_path)
            else:
                process.process_static_image_box(image_path)