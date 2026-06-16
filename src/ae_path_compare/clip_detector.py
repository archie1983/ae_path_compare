from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image
import requests


class CLIPDoorDetector:
	def __init__(self, device="cuda:0"):
		# We will use the base model from OpenAI, loaded via Hugging Face.
		model_name = "openai/clip-vit-base-patch32"

		# The CLIPProcessor handles both image preprocessing (resizing, normalizing)
		# and text tokenization.
		self.processor = CLIPProcessor.from_pretrained(model_name)

		# The CLIPModel contains the two towers:
		# 1. model.get_text_features()
		# 2. model.get_image_features()
		self.model = CLIPModel.from_pretrained(model_name).to(device)
		self.door_query = "an open door leading to another room"

	def is_there_door(self, image):
		inputs = self.processor(
			text=[self.door_query, "a wall", "a window", "a sofa"],
			images=image,
			return_tensors="pt",
			padding=True
		).to(device)
		with torch.no_grad():
			outputs = self.model(**inputs)
			probs = outputs.logits_per_image.softmax(dim=1)

		door_prob = probs[0, 0].item()  # Probability of "a door"
		return round(door_prob, 2)