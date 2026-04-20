import zmq
import numpy as np
from PIL import Image
import json

class PathCompareServer:
	def __init__(self, path_compare_instance, port=5555):
		self.pc = path_compare_instance
		self.port = port

		# Set up ZeroMQ with REP (REPly) socket
		self.context = zmq.Context()
		self.socket = self.context.socket(zmq.REP)  # Changed from PULL to REP
		self.socket.bind(f"tcp://*:{self.port}")

		print(f"Path Compare server running on port {self.port}, waiting for images...")

	# def __init__(self, path_compare_instance, image_port=5555, response_port=5556):
	# 	self.pc = path_compare_instance
	# 	self.context = zmq.Context()
	#
	# 	# Receive images on PULL socket
	# 	self.image_socket = self.context.socket(zmq.PULL)
	# 	self.image_socket.bind(f"tcp://*:{image_port}")
	#
	# 	# Send responses on PUSH socket
	# 	self.response_socket = self.context.socket(zmq.PUSH)
	# 	self.response_socket.bind(f"tcp://*:{response_port}")
	#
	# 	print(f"Server ready: receiving images on {image_port}, sending responses on {response_port}")

	def run(self):
		while True:
			# 1. Receive the image data
			data = self.socket.recv_pyobj()  # This BLOCKS until a request arrives

			# 2. Process the image
			received_array = np.frombuffer(data['bytes'], dtype=data['dtype'])
			received_image = received_array.reshape(data['shape'])

			# Convert to PIL Image (adjust color channels as needed)
			if received_image.shape[2] == 3:
				pil_image = Image.fromarray(received_image[:, :, ::-1])  # BGR to RGB
			else:
				pil_image = Image.fromarray(received_image)

			# 3. Get confidence using your PathCompare class
			confidence_analysis = self.pc.get_view_confidence(pil_image)

			# 4. Send the response back
			response = {
				'confidence_score': confidence_analysis['combined_score'],
				'peak_index': confidence_analysis['peak_index'],
				'peak_confidence': confidence_analysis['peak_confidence'],
				'entropy': confidence_analysis['entropy'],
				'is_confident': confidence_analysis['is_confident'],
				'status': 'success'
			}

			self.socket.send_pyobj(response)
			print(f"Sent response: confidence={confidence_analysis['combined_score']:.3f}")

if __name__ == "__main__":
	pcs = PathCompareServer