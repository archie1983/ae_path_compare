import torch
from transformers import CLIPProcessor, CLIPModel
import torch.nn.functional as F

class CLIPEncoder:
	def __init__(self, device="cuda:0"):
		self.device = device
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

	def encode_image(self, image):
		return self.encode_batch([image])

	def encode_batch(self, images):
		captions = [""]
		inputs = self.processor(text=captions, images=images, return_tensors="pt", padding=True)

		inputs = {k: v.to(self.device) for k, v in inputs.items()}  # Move to device

		# get embeddings
		with torch.no_grad():
			outputs = self.model(**inputs)
			embeds = outputs.image_embeds

		return embeds # torch.stack(embeddings)

	def compare_paths(self, ref_path, cur_path):
		#(ref_path_embeds, cur_path_embeds) = self.get_embeddings(ref_path, cur_path)
		# print(ref_path_embeds)
		ref_path_embeds = self.encode_batch(ref_path)
		cur_path_embeds = self.encode_batch(cur_path)

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