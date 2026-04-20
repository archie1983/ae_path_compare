import zmq
import numpy as np
from PIL import Image
from .path_compare import PathCompare

class PathCompareServer:
	def __init__(self, port=5555):
		self.pc = PathCompare()
		self.port = port

		# Set up ZeroMQ with REP (REPly) socket
		self.context = zmq.Context()
		self.socket = self.context.socket(zmq.REP)  # Changed from PULL to REP
		self.socket.bind(f"tcp://*:{self.port}")

		print(f"Path Compare server running on port {self.port}, waiting for images...")
		self.path_refs = {}

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

			if (data['action'] == 'store_ref_path'):
				# 2. Process the images
				received_array = np.frombuffer(data['bytes'], dtype=data['dtype'])
				received_images = received_array.reshape(data['shape'])
				pil_images = [Image.fromarray(img) for img in received_images]
				path_id = data['path_id']
				self.path_refs[path_id] = pil_images
				# Send the response back
				response = {
					'success': True
				}
			elif(data['action'] == 'cmp_path'):
				# 2. Process the images
				received_array = np.frombuffer(data['bytes'], dtype=data['dtype'])
				received_images = received_array.reshape(data['shape'])
				pil_images = [Image.fromarray(img) for img in received_images]
				#result_list =
				cmp_res = {k: self.pc.fit_cur_path_to_ref_path(v, pil_images)[1] for k, v in self.path_refs.items()}
				#max(cmp_res, key=cmp_res.get)
				print(cmp_res)
				best_match = max(cmp_res.items(), key=lambda k: k[1])
				response = {
					'best_match_ref': best_match[0],
					'best_match_score': best_match[1],
					'success': True
				}
				#for k, v in self.path_refs.items():
				#	cmp_res = self.pc.compare_paths(v, pil_images)
			self.socket.send_pyobj(response)


			# # Convert to PIL Image (adjust color channels as needed)
			# if received_image.shape[2] == 3:
			# 	pil_image = Image.fromarray(received_image[:, :, ::-1])  # BGR to RGB
			# else:
			# 	pil_image = Image.fromarray(received_image)

if __name__ == "__main__":
	pcs = PathCompareServer()
	pcs.run()
