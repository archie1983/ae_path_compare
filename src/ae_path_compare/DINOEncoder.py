import torch, glob, re
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import numpy as np

class DINOEncoder:
	def __init__(self, device="cuda:0"):
		self.device = device
		# Load DINOv2 model (base version, 768-dim embeddings)
		# 94 vs 58 = 36; 95-45=50
		#        self.processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
		#        self.model = AutoModel.from_pretrained('facebook/dinov2-base').to(device)

		# 91 vs 47 = 44; 89 - 40=49
		#        self.processor = AutoImageProcessor.from_pretrained('facebook/dinov3-vitl16-pretrain-lvd1689m')
		#        self.model = AutoModel.from_pretrained('facebook/dinov3-vitl16-pretrain-lvd1689m').to(device)

		# 92 vs 59 = 33; 91 - 40=51
		self.processor = AutoImageProcessor.from_pretrained('facebook/dinov3-vits16-pretrain-lvd1689m')
		self.model = AutoModel.from_pretrained('facebook/dinov3-vits16-pretrain-lvd1689m').to(device)

		# 94 vs 64 = 30; 94 - 54=40
		#        self.processor = AutoImageProcessor.from_pretrained('facebook/dinov3-vits16plus-pretrain-lvd1689m')
		#        self.model = AutoModel.from_pretrained('facebook/dinov3-vits16plus-pretrain-lvd1689m').to(device)

		# 95 vs 58 = 37; 93 - 53=40
		#        self.processor = AutoImageProcessor.from_pretrained('facebook/dinov3-vitb16-pretrain-lvd1689m')
		#        self.model = AutoModel.from_pretrained('facebook/dinov3-vitb16-pretrain-lvd1689m').to(device)

		# 88 vs 30 = 58; 88 vs 24=64
		#        self.processor = AutoImageProcessor.from_pretrained('facebook/dinov3-vith16plus-pretrain-lvd1689m')
		#        self.model = AutoModel.from_pretrained('facebook/dinov3-vith16plus-pretrain-lvd1689m').to(device)

		# 93 vs 43 = 50; 94 - 40=54
		#        self.processor = AutoImageProcessor.from_pretrained('facebook/dinov3-convnext-large-pretrain-lvd1689m')
		#        self.model = AutoModel.from_pretrained('facebook/dinov3-convnext-large-pretrain-lvd1689m').to(device)

		# 93 vs 62 = 31; 97 - 68=29
		#        self.processor = AutoImageProcessor.from_pretrained('facebook/dinov3-convnext-tiny-pretrain-lvd1689m')
		#        self.model = AutoModel.from_pretrained('facebook/dinov3-convnext-tiny-pretrain-lvd1689m').to(device)

		# 96 vs 68 = 28; 96 - 65=31
		#        self.processor = AutoImageProcessor.from_pretrained('facebook/dinov3-convnext-small-pretrain-lvd1689m')
		#        self.model = AutoModel.from_pretrained('facebook/dinov3-convnext-small-pretrain-lvd1689m').to(device)

		# 93 vs 56 = 37; 96 - 54=42
		#        self.processor = AutoImageProcessor.from_pretrained('facebook/dinov3-convnext-base-pretrain-lvd1689m')
		#        self.model = AutoModel.from_pretrained('facebook/dinov3-convnext-base-pretrain-lvd1689m').to(device)

		# 88 vs 34; 85-27=58
		#        self.processor = AutoImageProcessor.from_pretrained('facebook/dinov3-vit7b16-pretrain-lvd1689m')
		#        self.model = AutoModel.from_pretrained('facebook/dinov3-vit7b16-pretrain-lvd1689m').to(device)

		#        self.processor = AutoImageProcessor.from_pretrained('facebook/dinov3-vitl16-chmv2-dpt-head')
		#        self.model = AutoModel.from_pretrained('facebook/dinov3-vitl16-chmv2-dpt-head').to(device)

		# 99 vs 87; 99-88=11
		#        self.processor = AutoImageProcessor.from_pretrained('facebook/dinov3-vit7b16-pretrain-sat493m')
		#       self.model = AutoModel.from_pretrained('facebook/dinov3-vit7b16-pretrain-sat493m').to(device)

		# 88 vs 34; 85 - 27=58
		#        self.processor = AutoImageProcessor.from_pretrained('mirekphd/dinov3-vit7b16-pretrain-lvd1689m-fp16')
		#        self.model = AutoModel.from_pretrained('mirekphd/dinov3-vit7b16-pretrain-lvd1689m-fp16').to(device)

		self.processor.size = {'height': 448, 'width': 448}  # 88 vs 24 with vith16plus

	# mirekphd/dinov3-vit7b16-pretrain-lvd1689m-fp16
	def encode_image(self, image):
		"""
		Encode a single image into a feature vector.

		Args:
			image: PIL Image or numpy array (BGR from AI2-THOR)

		Returns:
			torch.Tensor of shape (768,) - feature embedding
		"""
		# Convert BGR to RGB if needed
		if isinstance(image, np.ndarray):
			if image.shape[-1] == 3:
				image = Image.fromarray(image[:, :, ::-1])  # BGR to RGB
		elif not isinstance(image, Image.Image):
			image = Image.fromarray(image)

		# Process and encode
		inputs = self.processor(images=image, return_tensors='pt')
		inputs = {k: v.to(self.device) for k, v in inputs.items()}

		with torch.no_grad():
			outputs = self.model(**inputs)
			# DINOv2 outputs: last_hidden_state shape (1, 197, 768)
			# We take the [CLS] token (first token) as the image representation
			embedding = outputs.last_hidden_state[:, 0, :].squeeze()  # Shape: (768,)

		return embedding

	def encode_batch(self, images):
		"""
		Encode a batch of images.

		Args:
			images: List of PIL Images or numpy arrays

		Returns:
			torch.Tensor of shape (N, 768)
		"""
		embeddings = [self.encode_image(img) for img in images]
		return torch.stack(embeddings)

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

	def load_path(self, base_dir):
		return self.load_images(base_dir + "/*.png")