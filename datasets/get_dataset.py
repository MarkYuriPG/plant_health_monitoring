from roboflow import Roboflow
import os

rf = Roboflow(api_key=os.environ['ROBOFLOW_KEY'])
project = rf.workspace("yuri-workspace").project("common-indoor-plants-in-ph")
version = project.version(6)
dataset = version.download("yolov8")