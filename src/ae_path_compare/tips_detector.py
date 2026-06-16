import torch
from transformers import AutoModel
import torch.nn.functional as F
from torchvision import transforms


class TIPSDetector:
	def __init__(self, device="cuda:0"):
		self.device = device
		# device = "cuda" if torch.cuda.is_available() else "cpu"
		self.model = AutoModel.from_pretrained("google/tipsv2-b14", trust_remote_code=True)
		# self.model.eval()

		# Pad 64x64 to 70x70 (3px on right/bottom, or center padding)
		# padded = F.pad(tensor, (0, 6, 0, 6), value=0)

		self.transform_600 = transforms.Compose([
			transforms.Resize((448, 448)),
			# transforms.Pad((0, 0, 2, 2)),
			transforms.ToTensor(),
		])

		self.transform = transforms.Compose([
			# transforms.Resize((448, 448)),
			transforms.Pad((0, 0, 6, 6)),
			transforms.ToTensor(),
		])

		classes = ["door to another room", "window", "floor"]
		self.text_emb = F.normalize(self.model.encode_text(classes), dim=-1).to(device)

	def is_there_door(self, image):
		# url = "https://huggingface.co/spaces/google/TIPSv2/resolve/main/examples/zeroseg/pascal_context_00049_image.png"
		# image = Image.open(requests.get(url, stream=True).raw)
		# print(image.size[0])
		if image.size[0] == 600:
			pixel_values = self.transform_600(image).unsqueeze(0).to(self.device)
		else:
			pixel_values = self.transform(image).unsqueeze(0).to(self.device)
		# pixel_values = self.transform(image).unsqueeze(0).to(self.device)
		out = self.model.encode_image(pixel_values)
		cls = F.normalize(out.cls_token[:, 0, :], dim=-1).to(self.device)
		similarity = cls @ self.text_emb.T
		# print(classes[similarity.argmax()])
		# print(out.cls_token.shape)     # (1, 1, 768) — global image embedding
		# print(out.patch_tokens.shape)
		return bool(similarity.argmax() == 0)

# Results contain:
# - masks: Binary masks resized to original image size
# - boxes: Bounding boxes in absolute pixel coordinates (xyxy format)
# - scores: Confidence scores
