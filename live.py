from ultralytics import YOLO

best_pt = 'C:/Users/Yuri/Projects/plant_identification/weights/phm28.pt'

# weights = 'C:/Users/Yuri/Projects/plant_identification/runs/detect/PHM26/weights/best.pt'

model = YOLO(best_pt)

results = model(source=0, conf=0.25, show=True, save=True)

# results.show()