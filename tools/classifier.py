import torch
from torch.nn import functional as F
from PIL import Image
from transformers import AutoImageProcessor, ResNetForImageClassification, AutoModelForImageClassification
import cv2
import os



class ObjectClassifier:
    # Initialize the classifier
    def __init__(self, model_path, visualize=True):
        self.classifier = ResNetForImageClassification.from_pretrained(model_path)
        self.image_processor = AutoImageProcessor.from_pretrained(model_path)
        self.visual = visualize

    def process_image(self, frame):
        if self.visual:
            print("Loop")
            cv2.imshow("Image input to classifier", frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame)
        inputs = self.image_processor(images=image, return_tensors="pt")
        return inputs

    def predict(self, frame):
        inputs = self.process_image(frame)
        with torch.no_grad():
            outputs = self.classifier(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=-1)
        top_prob, top_index = torch.max(probs, dim=-1)
        class_label = self.classifier.config.id2label[top_index.item()]
        return class_label, top_prob.item()

    def predict_top_k(self, frame, k=5):
        inputs = self.process_image(frame)
        with torch.no_grad():
            outputs = self.classifier(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=-1)
        topk_probs, topk_indices = torch.topk(probs, k)
        topk_probs = topk_probs.squeeze().tolist()
        topk_indices = topk_indices.squeeze().tolist()
        class_labels = [self.classifier.config.id2label[idx] for idx in topk_indices]
        return class_labels, topk_probs


class Classifier:
    # Initialize the classifier
    def __init__(self, model_path, use_safetensor=False):
        self.classifier = AutoModelForImageClassification.from_pretrained(model_path, use_safetensors=use_safetensor)
        self.image_processor = AutoImageProcessor.from_pretrained(model_path, use_safetensors=use_safetensor)

    def process_image(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame)
        inputs = self.image_processor(images=image, return_tensors="pt")
        return inputs

    def predict(self, frame):
        inputs = self.process_image(frame)
        with torch.no_grad():
            outputs = self.classifier(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=-1)
        top_prob, top_index = torch.max(probs, dim=-1)
        class_label = self.classifier.config.id2label[top_index.item()]
        return class_label, top_prob.item()
    



