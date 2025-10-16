import os
from ultralytics import YOLO
from IPython import display
display.clear_output()

import ultralytics
HOME = os.getcwd()
print(HOME)

ultralytics.checks()
model = YOLO(f'{HOME}/yolov8n.pt')


results = model.predict(source='https://media.roboflow.com/notebooks/examples/dog.jpeg', conf=0.25)

print(results)