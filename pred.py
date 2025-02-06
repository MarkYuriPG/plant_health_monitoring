from roboflow import Roboflow
import os, random

# Initialize Roboflow with your API key
rf = Roboflow(api_key=os.environ['ROBOFLOW_KEY'])

# Specify your project and model
project = rf.workspace("yuri-workspace").project("phm-v2")
model = project.version(2).model

# Check if model is loaded properly
assert model, "Model deployment is still loading"

# Define the test set location
test_set_loc = r"C:/Users/Yuri/Projects/plant_identification/datasets/PHMv2-1/test/images"

# Get a random test image
random_test_image = random.choice(os.listdir(test_set_loc))

# Print the image path being used
print("Running inference on " + random_test_image)

# Create the full file path using os.path.join
image_path = os.path.join(test_set_loc, random_test_image)

# Run prediction on the selected test image
pred = model.predict(image_path, confidence=40, overlap=30).json()

# Output the prediction
pred
