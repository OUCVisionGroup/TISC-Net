from ultralytics import YOLO

# Load a model
model = YOLO('yolov8m-pose-FEE.yaml')  # build a new model from YAML


# Train the model
model.train(data='sc_DUO-pose.yaml', epochs=100, imgsz=640, device = 3, batch = 32 )
metrics = model.val()  # evaluate model performance on the validation set
metrics.box.map    # map50-95
metrics.box.map50  # map50
metrics.box.map75  # map75
metrics.box.maps   # a list con
