import torch
from torch.nn import functional as F
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

def check_image_format(image):
    """
    Check whether an image is in RGB or BGR format.

    Args:
    - image: numpy.ndarray, input image array

    Returns:
    - str: "RGB" if the image is in RGB format, "BGR" if it's in BGR format,
           "Unknown" if the format cannot be determined
    """
    # Check the shape of the image array
    height, width, channels = image.shape

    # Check the channel order
    if channels == 3:
        if image[0, 0, 0] == image[0, 0, 2]:
            return "RGB"
        elif image[0, 0, 0] == image[0, 0, 1]:
            return "BGR"
        else:
            return "Unknown"
    else:
        return "Unknown"

print(check_image_format(color_frame_cut))

color_frame_cut = cv2.cvtColor(color_frame_cut, cv2.COLOR_BGR2RGB)
print(check_image_format(color_frame_cut))
class_labels, top5_probs = perform_inference_topk(color_frame_cut, k=5)
print("Top 5 predicted class labels:", class_labels)
print("Top 5 confidence scores:", top5_probs)


# Load png image to predict
frame = cv2.imread("electric_fan.png")
print(check_image_format(frame))
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
print(check_image_format(frame))
class_labels, top5_probs = perform_inference_topk(frame, k=5)
print("Top 5 predicted class labels:", class_labels)
print("Top 5 confidence scores:", top5_probs)

