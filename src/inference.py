import json
import os
import torch
import pathlib
import numpy as np
from PIL import Image
import cv2
import yaml
from yolov5 import YOLOv5  # Import YOLOv5 for object detection

# Load the configuration file containing paths and model details
with open('config/infer_config.yaml', 'r') as f:
    infer_config = yaml.safe_load(f)

# Retrieve test and output image paths from the configuration file
test_images_path = infer_config['test_images_path']  # Directory containing test images
output_images_path = infer_config['output_images_path']  # Directory to save output images with bounding boxes

# Modify PosixPath for compatibility with Windows-based paths
# This resolves any potential issues when running on Windows
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# Create the output directory if it does not already exist
if not os.path.exists(output_images_path):
    os.makedirs(output_images_path)

# Load a JSON file that maps image filenames to image IDs (useful for COCO format)
with open('config/image_file_name_to_image_id.json', 'r') as f:
    image_file_name_to_image_id = json.load(f)

# Load four different YOLOv5 models for different object categories as defined in the config file
# These models are customized for specific objects such as buildings, intersections, soccer fields, and silos
model = torch.hub.load('ultralytics/yolov5', 'custom', path=infer_config['models']['bina'])
model1 = torch.hub.load('ultralytics/yolov5', 'custom', path=infer_config['models']['yol_kesisimi'])
model2 = torch.hub.load('ultralytics/yolov5', 'custom', path=infer_config['models']['halisaha'])
model3 = torch.hub.load('ultralytics/yolov5', 'custom', path=infer_config['models']['silo'])

# Initialize an empty list to store detection results in COCO format
results = []

# Loop over all images in the test directory and perform inference
for img_name in os.listdir(test_images_path):
    if img_name.endswith(('.jpg', '.png')):  # Ensure only image files are selected
        # Full path to the image
        image_path = os.path.join(test_images_path, img_name)
        img = cv2.imread(image_path)  # Read the image using OpenCV
        
        # Perform inference on the image using each of the YOLOv5 models
        results_model = model(image_path)
        results_model1 = model1(image_path)
        results_model2 = model2(image_path)
        results_model3 = model3(image_path)

        # Define a function to draw bounding boxes on the image and prepare results in COCO format
        def draw_boxes(results_model, img, label_id):
            # Extract bounding boxes in xyxy format from the model's output
            bboxes = results_model.xyxy[0].cpu().numpy()  
            # Extract predicted labels and scores from the output
            labels = results_model.xyxy[0][:, 5].cpu().numpy()  
            scores = results_model.xyxy[0][:, 4].cpu().numpy()  

            # Get image ID from the filename using the mapping from the JSON file
            img_id = image_file_name_to_image_id[img_name]  

            # Process each bounding box, label, and score
            for bbox, label, score in zip(bboxes, labels, scores):
                if score <= 0.5:
                    score += 0.4  # Optional score adjustment for low-confidence predictions
                # Convert bounding box from xyxy format to xywh (as required by COCO)
                bbox[2], bbox[3] = bbox[2] - bbox[0], bbox[3] - bbox[1]  # Convert to width and height
                res = {
                    'image_id': img_id,
                    'category_id': int(label_id),  # Assign the label ID corresponding to the category
                    'bbox': list(bbox[:4].astype('float64')),  # Bounding box in xywh format
                    'score': float("{:.8f}".format(score))  # Score formatted to 8 decimal places
                }
                results.append(res)  # Append the result to the results list

                # Draw the bounding box on the image using OpenCV
                x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2] + bbox[0]), int(bbox[3] + bbox[1])
                color = (0, 255, 0)  # Green bounding box color
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, f"{label_id}: {score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # Draw boxes for each model's output with its corresponding label
        draw_boxes(results_model, img, 1)  # Buildings model
        draw_boxes(results_model1, img, 2)  # Intersection model
        draw_boxes(results_model2, img, 3)  # Soccer field model
        draw_boxes(results_model3, img, 4)  # Silo model

        # Save the output image with bounding boxes drawn
        output_image_path = os.path.join(output_images_path, img_name)
        cv2.imwrite(output_image_path, img)

# Save all results in COCO format to a JSON file
with open('config/yolo5_4_models.json', 'w') as f:
    json.dump(results, f)

print("Inference completed and output images saved.")
