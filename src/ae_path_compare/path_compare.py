import os, glob, re
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import requests
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

from .distribution_confidence import DistributionConfidence

class PathCompare:
	def __init__(self):
		self.device = "cuda:0"
		# We will use the base model from OpenAI, loaded via Hugging Face.
		model_name = "openai/clip-vit-base-patch32"
		# The CLIPProcessor handles both image preprocessing (resizing, normalizing)
		# and text tokenization.
		self.processor = CLIPProcessor.from_pretrained(model_name)
		# The CLIPModel contains:
		# 1. model.get_text_features()
		# 2. model.get_image_features()
		# although we realistically will only need image one
		self.model = CLIPModel.from_pretrained(model_name).to(self.device)
		print("CLIP Processor and Model loaded successfully.")
		self.confidence = DistributionConfidence()

	def extract_number(self, filename):
		# Extract the number from the filename (assuming it's the step count)
		# This regex looks for digits at the beginning, end, or between non-digits
		numbers = re.findall(r'\d+', filename)
		return int(numbers[-1]) if numbers else 0

	def load_images(self, path):
		imgs_path = glob.glob(path)
		imgs_path = sorted(imgs_path, key=self.extract_number)
		pil_images = [Image.open(fname).convert('RGB') for fname in imgs_path]
		return pil_images

	def load_ref_path(self):
		return self.load_images("/home/hp20024/robotics/ref_path_embedding/hab_img2/path1_*.png")

	def load_cur_path(self):
		return self.load_images("/home/hp20024/robotics/ref_path_embedding/hab_img2/path2_*.png")

	def load_alien_path(self):
		return self.load_images("/home/hp20024/robotics/ref_path_embedding/hab_img2/path3/path3_*.png")

	def compare_paths(self, ref_path, cur_path):
		captions = [""]
		ref_path_inputs = self.processor(text=captions, images=ref_path, return_tensors="pt", padding=True)
		cur_path_inputs = self.processor(text=captions, images=cur_path, return_tensors="pt", padding=True)

		ref_path_inputs = {k: v.to(self.device) for k, v in ref_path_inputs.items()}  # Move to device
		cur_path_inputs = {k: v.to(self.device) for k, v in cur_path_inputs.items()}  # Move to device

		# get embeddings
		with torch.no_grad():
			ref_path_outputs = self.model(**ref_path_inputs)
			ref_path_embeds = ref_path_outputs.image_embeds
			cur_path_outputs = self.model(**cur_path_inputs)
			cur_path_embeds = cur_path_outputs.image_embeds

		# print(ref_path_embeds)

		ideal_path_normalized = F.normalize(ref_path_embeds, dim=1)
		current_path_normalized = F.normalize(cur_path_embeds, dim=1)

		# Get similarity to all reference frames in one matrix multiplication
		similarities = torch.mm(current_path_normalized, ideal_path_normalized.t()).squeeze()

		logit_scale = self.model.logit_scale.exp()
		logits = similarities * logit_scale
		# logits[range(len(logits)), range(len(logits[0]))] = 0 # we're not interested in each image compared to itself, so set the diagonal to 0
		# 4. Convert logits to probabilities using Softmax
		probs = F.softmax(logits, dim=-1)
		return probs

	def fit_single_img_to_ref_path(self, ref_path, img):
		probs = self.compare_paths(ref_path, [img])
		# print("Distribution 1 Analysis:")
		# print(confidence.analyze(dist1))
		# print("decision: ", confidence.agent_decision(confidence.analyze(dist1)))
		probs = probs.cpu().detach()
		dec = self.confidence.agent_decision(self.confidence.analyze(probs))
		return dec

	def fit_cur_path_to_ref_path(self, ref_path, cur_path):
		fittings = [self.fit_single_img_to_ref_path(ref_path, cp) for cp in cur_path]
		ft = [f[1] if f[0] != 'U' else -1 for f in fittings]
		cont = self.confidence.path_continuity(ft)
		return fittings, cont

	def visualize_probs(self, probs):
		# To accomodate probs having only 1 dimension because we only compare 1 image against a whole reference path, we need to check shape and unsqueeze if needed
		if len(probs.shape) == 1:
			probs = probs.unsqueeze(dim=0)
		rows = probs.shape[0]
		cols = probs.shape[1]
		probs = probs.cpu().detach().numpy()
		plt.figure(figsize=(56, 20))
		plt.subplot(1, 2, 2)
		plt.imshow(probs, cmap='viridis', vmin=0, vmax=1)
		plt.title("Path image similarities")
		plt.xlabel("Reference path")
		plt.ylabel("Current path")
		plt.xticks(range(cols), [f"{i + 1}" for i in range(cols)], rotation=45)
		plt.yticks(range(rows), [f"{i + 1}" for i in range(rows)])
		plt.colorbar()

		# Add text for probabilities
		for i in range(rows):
			for j in range(cols):
				plt.text(j, i, f"{probs[i, j]:.2f}", ha='center', va='center',
						 color='white' if probs[i, j] < 0.5 else 'black')

		plt.tight_layout()
		plt.show()


pc = PathCompare()
