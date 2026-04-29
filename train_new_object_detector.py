# # This section for google colab only

# from google.colab import drive
# drive.mount('/content/drive')

# # update package list and install Python 3.10
# !sudo apt-get update -y
# !sudo apt-get install python3.10 python3.10-distutils python3.10-venv -y

# # get the correct version of pip for Python 3.10
# !wget https://bootstrap.pypa.io/get-pip.py
# !python3.10 get-pip.py

# # install mediapipe-model-maker in the isolated Python 3.10 environment
# !python3.10 -m pip install mediapipe-model-maker
# %%writefile train_model.py 

###########################################

import os
import shutil
import json
from mediapipe_model_maker import object_detector

# Define dataset path
dataset_path = '/content/drive/MyDrive/GameController.v1i.coco'
train_dir = os.path.join(dataset_path, 'train')
valid_dir = os.path.join(dataset_path, 'valid')

# fix directory structure and Label IDs for MediaPipe
for directory in [train_dir, valid_dir]:
    if not os.path.exists(directory): continue

    # create 'images' subfolder if it doesn't exist
    img_subfolder = os.path.join(directory, 'images')
    os.makedirs(img_subfolder, exist_ok=True)

    # handle json label shifting
    old_json = os.path.join(directory, '_annotations.coco.json')
    new_json = os.path.join(directory, 'labels.json')

    # If file exists, shift ID to avoid index 0 conflict
    source_json = old_json if os.path.exists(old_json) else new_json
    if os.path.exists(source_json):
        with open(source_json, 'r') as f:
            data = json.load(f)

        # shift category ID by 1 because 0 reserved for background
        for category in data['categories']:
            category['id'] = category['id'] + 1
        for annotation in data['annotations']:
            annotation['category_id'] = annotation['category_id'] + 1

        with open(new_json, 'w') as f:
            json.dump(data, f)

    #move images into the 'images' folder
    for filename in os.listdir(directory):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            dest_path = os.path.join(img_subfolder, filename)
            if not os.path.exists(dest_path):
                shutil.move(os.path.join(directory, filename), dest_path)

# load the data
print("loading datasets......")
train_data = object_detector.Dataset.from_coco_folder(
    train_dir, cache_dir="/tmp/od_data/train")
validation_data = object_detector.Dataset.from_coco_folder(
    valid_dir, cache_dir="/tmp/od_data/validation")

# training options setup
hparams = object_detector.HParams(export_dir='game_controller_model', epochs=30, batch_size=8)
# Using the correct model specification enum name
options = object_detector.ObjectDetectorOptions(
    supported_model=object_detector.SupportedModels.MOBILENET_V2_I320,
    hparams=hparams
)

# training
print("start training...")
model = object_detector.ObjectDetector.create(
    train_data=train_data,
    validation_data=validation_data,
    options=options
)

# evaluation
loss, coco_metrics = model.evaluate(validation_data, batch_size=8)
print(f"Validation loss: {loss}")
model.export_model()
print("Export complete!")