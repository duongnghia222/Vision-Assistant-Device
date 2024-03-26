import torch
from torch.nn import functional as F
from transformers import AutoImageProcessor, ResNetForImageClassification
from PIL import Image
import cv2
import numpy as np
import json

# Load the model
model_path = "../models/resnet-50"
model = ResNetForImageClassification.from_pretrained(model_path)
processor = AutoImageProcessor.from_pretrained(model_path)


def perform_inference_topk(frame, k=5):
    # Convert the frame to PIL image
    image = Image.fromarray(frame)

    # Preprocess the image using the ViT processor
    inputs = processor(images=image, return_tensors="pt")

    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)

    # Get the predicted class index
    logits = outputs.logits
    probs = F.softmax(logits, dim=-1)

    # Get the top 5 predicted class indices and their corresponding confidence scores
    top5_probs, top5_indices = torch.topk(probs, k)
    top5_probs = top5_probs.squeeze().tolist()
    top5_indices = top5_indices.squeeze().tolist()

    # Get the corresponding class labels
    class_labels = [model.config.id2label[idx] for idx in top5_indices]

    return class_labels, top5_probs

# Open color frame json file to read color frame as numpy array
# Load the JSON file
with open('../color_frame_values.json', 'r') as f:
    color_frame_values = json.load(f)

# Convert the color frame values to a numpy array
color_frame = np.array(color_frame_values, dtype=np.uint8)
cv2.imshow('Color Frame Full', color_frame)
cv2.waitKey(0)
color_frame_cut = color_frame[128:452, 208:432]
cv2.imshow('Color Frame', color_frame_cut)
cv2.waitKey(0)
cv2.destroyAllWindows()

color_frame_cut = cv2.cvtColor(color_frame_cut, cv2.COLOR_BGR2RGB)
class_labels, top5_probs = perform_inference_topk(color_frame_cut, k=5)
print("Top 5 predicted class labels:", class_labels)
print("Top 5 confidence scores:", top5_probs)


# Load png image to predict
frame = cv2.imread("electric_fan.png")
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
class_labels, top5_probs = perform_inference_topk(frame, k=5)
print("Top 5 predicted class labels:", class_labels)
print("Top 5 confidence scores:", top5_probs)
