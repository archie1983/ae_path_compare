import torch
import numpy as np

class DistributionConfidence:
	def __init__(self):
		pass

	def agent_decision(self, analysis):
		"""
		Make decisions based on confidence analysis
		"""
		if analysis['combined_score'] > 0.8:
			# Very confident - trust the match
			return "E", analysis['peak_index']

		elif analysis['combined_score'] > 0.5:
			# Moderately confident - useful but be cautious
			return "P1", analysis['peak_index']

		elif analysis['peak_to_wall_ratio'] > 5:
			# Strong single peak despite lower combined score
			return "P2", analysis['peak_index']

		else:
			# Not confident - don't rely on this match
			return "U", None

	def analyze(self, probs):
		"""
		Analyze a probability distribution and return multiple confidence metrics.

		Args:
			probs: torch.Tensor of probability values

		Returns:
			dict with various confidence scores
		"""
		probs = probs / probs.sum()  # Ensure valid distribution
		peak_val, peak_idx = torch.max(probs, dim=0)

		# Individual metrics
		peak_conf = peak_val.item()
		entropy = self._entropy(probs)
		ipr = self._ipr(probs)
		peak_to_wall = self._peak_to_wall(probs, peak_idx)
		kl_uniform = self._kl_from_uniform(probs)

		# Normalized metrics (0-1 scale)
		n_components = len(probs)
		max_entropy = np.log(n_components)  # For uniform distribution
		normalized_entropy = 1 - (entropy / max_entropy)  # Higher = more confident

		max_ipr = 1.0  # One peak at 1.0
		min_ipr = 1.0 / n_components  # Uniform distribution
		normalized_ipr = (ipr - min_ipr) / (max_ipr - min_ipr)

		# Combined confidence score (weighted average)
		combined_score = (
				peak_conf * 0.3 +
				normalized_entropy * 0.2 +
				normalized_ipr * 0.2 +
				min(peak_to_wall / 10, 1.0) * 0.2 +  # Cap at 1.0
				min(kl_uniform / 2, 1.0) * 0.1
		)

		return {
			'peak_confidence': peak_conf,
			'peak_index': peak_idx.item(),
			'entropy': entropy,
			'normalized_entropy': normalized_entropy,
			'ipr': ipr,
			'normalized_ipr': normalized_ipr,
			'peak_to_wall_ratio': peak_to_wall,
			'kl_from_uniform': kl_uniform,
			'combined_score': combined_score,
			'is_confident': combined_score > 0.7  # Adjust threshold as needed
		}

	def _entropy(self, probs):
		"""Calculate Shannon entropy"""
		probs = probs[probs > 0]
		return -torch.sum(probs * torch.log(probs)).item()

	def _ipr(self, probs):
		"""Calculate Inverse Participation Ratio"""
		return torch.sum(probs ** 2).item()

	def _peak_to_wall(self, probs, peak_idx):
		"""Calculate peak-to-wall ratio"""
		peak = probs[peak_idx]
		other_probs = torch.cat([probs[:peak_idx], probs[peak_idx + 1:]])
		mean_other = torch.mean(other_probs)
		return (peak / mean_other).item()

	def _kl_from_uniform(self, probs):
		"""Calculate KL divergence from uniform distribution"""
		n = len(probs)
		uniform = torch.ones(n) / n
		kl = torch.sum(probs * torch.log(probs / uniform))
		return kl.item()

	def path_continuity(self, fittings):
		"""
		Estimate how continuous is the path. E.g., [1,2,3,4] is continuous, [1,2,-1,4] is less so and [1,-1,1,4] is not continuous.
		"""
		fitting_p = -1
		fitting_pp = -1
		fitting_c = -1
		score_good = 0
		score_cnt = 0

		for i in range(len(fittings)):
			# If current index is 0, then we can't evaluate the fitting
			if i > 0:
				# If current fitting is previous + 1 or prev previous + 2, then that's good
				if (fittings[i] == fitting_p + 1) or (fittings[i] == fitting_pp + 2):
					score_good += 1
				score_cnt += 1

			fitting_pp = fitting_p
			fitting_p = fittings[i]

		return (score_good / score_cnt)

# Example usage
confidence = DistributionConfidence()

dist1 = torch.tensor([0.01, 0.01, 0.90, 0.04, 0.04])
dist2 = torch.tensor([0.2, 0.35, 0.36, 0.15, 0.12])

print("Distribution 1 Analysis:")
print(confidence.analyze(dist1))
print("decision: ", confidence.agent_decision(confidence.analyze(dist1)))
print("\nDistribution 2 Analysis:")
print(confidence.analyze(dist2))
print("decision: ", confidence.agent_decision(confidence.analyze(dist2)))