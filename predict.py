from ultralytics import YOLO
import streamlit as st
import cv2
import numpy as np

best_pt = 'C:/Users/Yuri/Projects/plant_identification/weights/phmv2-1.pt'

# weights = 'C:/Users/Yuri/Projects/plant_identification/runs/detect/PHM28/weights/best.pt'

model = YOLO(best_pt)
# model = YOLO('yolov8s.pt')

st.title("PHM")

path = 'D:/School/SP/plants/V2'

save_path='../plant_identification/outputs' #This is where your plant images you want to use for testing are found
results = model(source=path, conf=0.25, save=True, save_dir=save_path)