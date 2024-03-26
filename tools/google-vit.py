import torch
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import cv2
import numpy as np
import json

# Load the model
model_path = "../models/google-vit"
model = ViTForImageClassification.from_pretrained(model_path)
processor = ViTImageProcessor.from_pretrained(model_path)


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
    # Get the top k predicted class indices and their corresponding scores
    topk_scores, topk_indices = torch.topk(logits, k, dim=-1)

    # Get the predicted class labels and their corresponding scores
    predicted_classes = [model.config.id2label[idx.item()] for idx in topk_indices[0]]
    confidence_scores = [score.item() for score in topk_scores[0]]

    # Combine the predicted classes and their scores into tuples
    topk_results = [(predicted_classes[i], confidence_scores[i]) for i in range(k)]

    return topk_results

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
top5_predictions = perform_inference_topk(color_frame_cut, k=5)
print("Top 5 Predictions:")
for i, (class_name, confidence) in enumerate(top5_predictions):
    print(f"{i+1}. Class: {class_name}, Confidence: {confidence:.4f}")
