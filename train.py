from ultralytics import YOLO

if __name__ == '__main__':
    # Load the YOLOv8 small model
    model = YOLO('yolov8s.pt')

    # Train the model
    model.train(data='C:\\Users\\Yuri\\Projects\\plant_identification\\datasets\\Common-Indoor-Plants-in-PH-6\\data.yaml', 
                epochs=100, 
                imgsz=800, 
                plots=True)
