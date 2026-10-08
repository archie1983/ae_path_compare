import torch, glob, re
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import numpy as np
import torch.nn.functional as F
from collections.abc import Iterable

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

    # def encode_batch_mean(self, images):
    # 	path_embeds_cls = self.encode_batch(images)
    # 	path_embeds_mean = path_embeds_cls.mean(dim=0)
    # 	return path_embeds_mean

    def encode_batch_mean(self, images):
        # 1. Get raw [CLS] embeddings for the path (Shape: N, 768)
        path_embeds_cls = self.encode_batch(images)

        # 2. Normalize each individual frame's vector to unit length
        path_embeds_norm = F.normalize(path_embeds_cls, p=2, dim=1)

        # 3. Take the directional average of the paths
        path_embeds_mean = path_embeds_norm.mean(dim=0)

        # 4. Re-normalize the final average vector so it has a magnitude of 1.0
        # This guarantees flawless alignment with ChromaDB's internal HNSW cosine index
        final_signature = F.normalize(path_embeds_mean, p=2, dim=0)

        return final_signature

    def encode_batch_normalized(self, images):
        # 1. Get raw [CLS] embeddings for the path (Shape: N, 768)
        path_embeds_cls = self.encode_batch(images)
        # 2. Normalize each individual frame's vector to unit length
        path_embeds_norm = F.normalize(path_embeds_cls, p=2, dim=1)
        return path_embeds_norm

    # def weed_out_odd_images(self, embeddings):
    #     final_embeddings = []
    #     similarity_threshold = 0.60
    #
    #     # handle cases with 0, 1 and 2 images only
    #     if len(embeddings) < 2: return embeddings
    #     if len(embeddings) == 2 and F.cosine_similarity(embeddings[0], embeddings[1], dim=0) >= similarity_threshold:
    #         return embeddings
    #     elif (len(embeddings) == 2 and F.cosine_similarity(embeddings[0], embeddings[1], dim=0) < similarity_threshold):
    #         return final_embeddings
    #
    #     # if we're here, then there's at least 3 embeddings, let's see if the 1st one is similar to any others
    #     sim_0_1 = F.cosine_similarity(embeddings[0], embeddings[1], dim=0)
    #     sim_0_2 = F.cosine_similarity(embeddings[0], embeddings[2], dim=0)
    #     if (sim_0_1 < similarity_threshold and sim_0_2 < similarity_threshold):
    #         #print("AE: throwing out 0: ", img_names_full[0])
    #         return self.weed_out_odd_images(embeddings[1:])  # the first image is not a good match, let's drop it
    #     else:  # the first image matches with either the 2nd or 3rd, the rest will be handled by a loop
    #         final_embeddings.append(embeddings[0])
    #
    #         img1_index = 0
    #         img2_index = 1
    #         while img2_index < len(embeddings):
    #             img1 = embeddings[img1_index]
    #             img2 = embeddings[img2_index]
    #             cos_sim = float(F.cosine_similarity(img1, img2, dim=0))
    #             if cos_sim >= similarity_threshold:  # img2 is goog, let's add it and move on to the next pair (img2 with img3)
    #                 # print("AE: keeping: ", img_names_full[img2_index], " cos_sim = ", cos_sim)
    #                 final_embeddings.append(img2)
    #                 img1_index = img2_index
    #                 img2_index += 1
    #             else:  # img2 is bad, let's drop it and compare img1 with img3
    #                 #print("AE: throwing out: ", img2_index, " # ", img_names_full[img2_index], " cos_sim = ", cos_sim)
    #                 img2_index += 1
    #
    #     return final_embeddings

    def weed_out_odd_images(self, embeddings):
        similarity_threshold = 0.60
        num_frames = len(embeddings)
        #filenames = img_names_full

        # 1. Stack list into a unified tensor (Shape: N, 768)
        # If embeddings is already a stacked tensor, this is a zero-cost operation
        if isinstance(embeddings, list):
            embeddings_tensor = torch.stack(embeddings)
        else:
            embeddings_tensor = embeddings

        # 2. Compute the complete all-to-all similarity matrix in one GPU cycle
        # Matrix multiplication of normalized vectors yields the exact cosine similarities
        sim_matrix = torch.mm(embeddings_tensor, embeddings_tensor.t())

        # 3. Establish consensus: Count how many other frames each image agrees with
        # We check how many values in each row cross our similarity_threshold requirement
        agreement_mask = sim_matrix >= similarity_threshold
        agreement_counts = agreement_mask.sum(dim=1)  # Shape: (N,)

        # 4. Filter logic: Keep frames that match the global majority
        # A true frame should agree with at least 50% of the trajectory cohort
        majority_cutoff = num_frames // 2

        final_embeddings = []
        for idx in range(num_frames):
            if agreement_counts[idx] >= majority_cutoff:
                # fname = filenames[idx] if idx < len(filenames) else f"Index_{idx}"
                # print(f"AE: [KEPT]: {fname} | Global agreement count: {agreement_counts[idx]}/{num_frames}")
                final_embeddings.append(embeddings_tensor[idx])
            # else:
            #     # localized filename extraction
            #     fname = filenames[idx] if idx < len(filenames) else f"Index_{idx}"
            #     print(f"AE: [Vector Purge] throwing out: {fname} | Global agreement count: {agreement_counts[idx]}/{num_frames}")

        return final_embeddings

    def encode_potentially_ambiguous_batch(self, images):
        """
        If we have a batch of images where some may be odd and shouldn't belong to the passed collection, then we can filter them
        out by doing a cosine similarity on all of the images and throwing out the odd ones. That's what we do here.
        :param images:
        :return:
        """
        embeddings = self.encode_batch_normalized(images)
        return self.weed_out_odd_images(embeddings)

    def encode_potentially_ambiguous_batch_mean(self, images):
        # encode and filter out ambiguous ones
        path_embeds_norm = self.encode_potentially_ambiguous_batch(images)
        # Take the directional average of the paths
        path_embeds_mean = path_embeds_norm.mean(dim=0)
        # Re-normalize the final average vector so it has a magnitude of 1.0
        # This guarantees flawless alignment with ChromaDB's internal HNSW cosine index
        final_signature = F.normalize(path_embeds_mean, p=2, dim=0)

        return final_signature

    def compare_mean_embeddings(self, mean_embeds1, mean_embeds2):
        # if we have a set of embeddings to compare, then compare all, otherwise just the one
        if isinstance(mean_embeds1, Iterable):
            return {F.cosine_similarity(me, mean_embeds2, dim=0) for me in mean_embeds1}
        else:
            return F.cosine_similarity(mean_embeds1, mean_embeds2, dim=0)

    def compare_paths(self, ref_path, cur_path):
        #(ref_path_embeds, cur_path_embeds) = self.get_embeddings(ref_path, cur_path)
        # print(ref_path_embeds)
        ref_path_embeds = self.encode_batch(ref_path)
        cur_path_embeds = self.encode_batch(cur_path)

        ideal_path_normalized = F.normalize(ref_path_embeds, dim=1)
        current_path_normalized = F.normalize(cur_path_embeds, dim=1)

        # Get similarity to all reference frames in one matrix multiplication
        similarities = torch.mm(current_path_normalized, ideal_path_normalized.t()).squeeze()

        # logits[range(len(logits)), range(len(logits[0]))] = 0 # we're not interested in each image compared to itself, so set the diagonal to 0
        # 4. Convert logits to probabilities using Softmax
        probs = F.softmax(similarities, dim=-1)
        return probs

    def compare_paths_mean(self, ref_path, cur_path):
        ref_path_embeds = self.encode_batch(ref_path)
        cur_path_embeds = self.encode_batch(cur_path)

        ref_mean = ref_path_embeds.mean(dim=0)
        cur_mean = cur_path_embeds.mean(dim=0)
        return F.cosine_similarity(ref_mean, cur_mean, dim=0)

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