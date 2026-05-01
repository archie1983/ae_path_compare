import os, glob, re
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import requests
from PIL import Image

from .distribution_confidence import DistributionConfidence
from .dino_encoder import DINOEncoder
from .clip_encoder import CLIPEncoder

class PathCompare:
	def __init__(self, use_dino = True):
		if use_dino:
			self.encoder = DINOEncoder()
		else:
			self.encoder = CLIPEncoder()
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

	def get_embeddings(self, ref_path, cur_path):
		ref_path_embeds = self.encoder.encode_batch(ref_path)
		cur_path_embeds = self.encoder.encode_batch(cur_path)
		return ref_path_embeds, cur_path_embeds

	def compare_paths(self, ref_path, cur_path):
		probs = self.encoder.compare_paths(ref_path, cur_path)
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
