import zmq
import numpy as np
import cv2
import time

class NavigationAgent:
	def __init__(self, jetson_ip, port=5555):
		self.context = zmq.Context()
		self.socket = self.context.socket(zmq.REQ)  # REQuest socket
		self.socket.connect(f"tcp://{jetson_ip}:{port}")
		print(f"Connected to Jetson at {jetson_ip}:{port}")

	def send_image_and_get_confidence(self, image_np):
		"""
		Send an image to Jetson and wait for confidence analysis.

		Args:
			image_np: numpy array (H, W, C) in BGR order (typical from OpenCV/AI2-THOR)

		Returns:
			dict with confidence metrics, or None if error
		"""
		# Serialize the image
		data = {
			'shape': image_np.shape,
			'dtype': str(image_np.dtype),
			'bytes': image_np.tobytes()
		}

		# Send request
		self.socket.send_pyobj(data)

		# Wait for response (this BLOCKS until Jetson replies)
		try:
			response = self.socket.recv_pyobj()
			return response
		except zmq.ZMQError as e:
			print(f"Error receiving response: {e}")
			return None

	def navigate_with_feedback(self, get_image_func, max_steps=100):
		"""
		Main navigation loop with real-time confidence feedback.

		Args:
			get_image_func: Function that captures current FPV image from AI2-THOR
			max_steps: Maximum number of steps to take
		"""
		for step in range(max_steps):
			# 1. Capture current view
			current_image = get_image_func()  # Returns numpy array

			# 2. Send to Jetson and get confidence
			result = self.send_image_and_get_confidence(current_image)

			if result and result['status'] == 'success':
				confidence = result['confidence_score']

				# 3. Use confidence to guide navigation
				if confidence > 0.7:
					print(f"✅ Step {step}: High confidence ({confidence:.2f}) - continue")
				# Continue with planned movement
				# agent.move_forward()

				elif confidence > 0.4:
					print(f"⚠️ Step {step}: Medium confidence ({confidence:.2f}) - proceed with caution")
				# Slow down, look around more
				# agent.move_forward(speed=0.5)

				else:
					print(f"❌ Step {step}: Low confidence ({confidence:.2f}) - LOST!")
				# Trigger recovery behavior
				# agent.stop_and_reorient()
				# agent.explore_until_recovered()

			# Optional: Store confidence history for trajectory analysis
			# self.confidence_history.append(confidence)

			else:
				print(f"Step {step}: Failed to get confidence from Jetson")
			# Implement fallback behavior

			# Small delay to avoid overwhelming the system
			time.sleep(0.05)


# Example usage
def capture_from_ai2thor():
	"""Replace with your actual AI2-THOR frame capture"""
	# Simulate a 640x480 RGB image
	return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


# Create agent and connect to Jetson
agent = NavigationAgent(jetson_ip="192.168.1.100", port=5555)

# Start navigation
agent.navigate_with_feedback(capture_from_ai2thor, max_steps=50)