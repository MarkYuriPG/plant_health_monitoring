from ultralytics import YOLO

if __name__ == '__main__':
    # Load the YOLOv8 small model
    model = YOLO('yolov8s.pt')

    # Train the model
    model.train(data='C:\\Users\\Yuri\\Projects\\plant_identification\\datasets\\PHM-v2-2\\data.yaml', 
                epochs=100,
                imgsz=640,
                plots=True,
                name="PHMv2",
                batch=8,
                device='cuda',
            )