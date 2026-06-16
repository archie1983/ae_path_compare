from transformers import Sam3Processor, Sam3Model
import torch
from PIL import Image
import requests


class SAMDetector:
	def __init__(self, device="cuda:0"):
		self.device = device
		# device = "cuda" if torch.cuda.is_available() else "cpu"
		self.model = Sam3Model.from_pretrained("facebook/sam3").to(device)
		self.processor = Sam3Processor.from_pretrained("facebook/sam3")
		self.door_query = "an open door leading to another room"

	def is_there_door(self, image):
		# Load image
		# image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")

		# Segment using text prompt
		inputs = processor(images=image, text=self.door_query, return_tensors="pt").to(device)

		with torch.no_grad():
			outputs = model(**inputs)

		# Post-process results
		results = processor.post_process_instance_segmentation(
			outputs,
			threshold=0.5,
			mask_threshold=0.5,
			target_sizes=inputs.get("original_sizes").tolist()
		)[0]

		# print(float(max(results["scores"])))
		if len(results['masks']) > 0:
			return round(float(max(results["scores"])), 2)
		else:
			return 0.0
# Results contain:
# - masks: Binary masks resized to original image size
# - boxes: Bounding boxes in absolute pixel coordinates (xyxy format)
# - scores: Confidence scores
