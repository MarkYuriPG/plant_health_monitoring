# from ultralytics import YOLO
from ultralytics import YOLO, checks, hub

if __name__ == '__main__':
    # Load the YOLOv8 small model
    # model = YOLO('yolov8m.pt')

    # # Train the model
    # model.train(data='C:/Users/Yuri/Projects/plant_identification/Plant-Diagnosis-12-(Snake-Plant)-1/data.yaml', 
    #             epochs=25,
    #             imgsz=800,
    #             plots=True,
    #             iou=0.6, 
    #             conf=0.3,
    #             name="PHMv2",
    #             batch=8,
    #             device='cuda',
    #             lr0=0.0001,
    #         )
    checks()

    hub.login('5f6e75d0e0c9d0e9b36f473ea16814244ff3f15665')

    model = YOLO('https://hub.ultralytics.com/models/wZSUC30rH332mK1BCxZu')
    results = model.train()