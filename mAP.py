from map import map
from face_detector import FaceDetector as fd
from gryfo.blocks import FaceDetector as gfd
from gryfo.blocks import GenericDetector as ggd
import numpy as np
import os
import pickle
import json




class mAP():
	def __init__(self,env):

		CONFIG = 'datasets.json'
		self.env_params =  json.load(open(CONFIG))
		if env == '1' or env == 'wider':
			self.params = self.env_params['wider']
		elif env == '2' or env == 'monitora':
			self.params = self.env_params['monitora']

		self.model = self.params['model']
		self.groundtruth_file = self.params['groundtruth_file']
		self.image_path = self.params['image_path']

		self.facedetector = fd(self.model, gpu_memory_fraction=0.25, visible_device_list='0')
		self.gryfo_facedetector = gfd(detector_type='mssd')
		self.gryfo_genericdetector =  ggd(classes=["person"])
		self.map = map()
		self.gtimages, self.nofimg = self.check_gt()
		self.dataset_images, self.nofdsimg = self.check_dataset()


	def check_gt(self):
		images = []
		gt = self.groundtruth_file
		gtfile = open(gt , 'rt')
		lines = gtfile.readlines()
		c2 = 0
		for c, l in enumerate(lines):
			li = lines[c].split()
			if len(li) == 1:
				if len(li[0])>=5:
					images.append(li[0])
					c2 +=1
		return sorted(images), c2

	def check_dataset(self):
		image_dir= self.image_path
		len1 = len(os.listdir(image_dir))
		embeds = []
		for c, image_dir2 in enumerate(os.listdir(image_dir)):
			for c2, image_file in enumerate(os.listdir(os.path.join(image_dir, image_dir2))):
				embeds.append(os.path.join(image_dir2, image_file))
		return sorted(embeds), len(embeds)
	
	def check_consistency(self):
		f=0
		for i in self.gtimages:
			if i not in self.dataset_images:
				f = 1
		return f
	
	def check_consistency_2(self):
		print(len(self.gtimages))
		for i in range(0,len(self.gtimages)):
			if self.gtimages[i] != self.dataset_images[i]:
				print('{} inconsistente'.format(i))
		return 0



					

	def gt_to_dict(self):
		#Reads the groundtruth file and outputs a dicionary with:
		# keys: image identificators
		# values: lists of rectangles.
		m = self.map
		gt = self.groundtruth_file
		gtfile = open(gt , 'rt')
		lines = gtfile.readlines()
		images = {}
		c, c2  = 0, 0
		f1, f2 = open('i1','wt'),  open('i2','wt') 
		while c < len(lines):
			li = lines[c].split()
			ret = []
			ret2 = []
			if len(li) == 1:
				if len(li[0]) < 5:
					i = int(li[0])
					ret.extend(list(lines[c+1:c+i+1]))
					ret2 = m.toyxyx(m.tofloat2(ret))
					images[image] = ret2
					c2 +=1 
					c = c + i +1
					continue
				else:
					image = li[0]
			c += 1
		pickle.dump(images,open('data/faceboxes/images_dictionary.pkl','wb'))
		return images



	def embed_to_dict(self, threshold = 0.5):
		#Performs face detection in all images specified in self.image_path and
		#returns a dictionary with:
		#keys: image identificators
		#values: lists of rectangles.
		#obs: uses faceboxes face detector.
		mp = self.model
		fd = self.facedetector
		m = self.map
		image_dir= self.image_path

		len1 = len(os.listdir(image_dir))
		embeds = {}
		for c, image_dir2 in enumerate(os.listdir(image_dir)):
			for c2, image_file in enumerate(os.listdir(os.path.join(image_dir, image_dir2))):
				len2 = len(os.listdir(os.path.join(image_dir, image_dir2)))
				image_full_path = os.path.join(image_dir, image_dir2, image_file)
				#print(image_full_path)
				boxes, scores = m.embed_fb(fd, image_full_path, threshold)
				#print(boxes)
				embeds[image_full_path] = boxes
				print('Embedding: {0:0.2f}%'.format(c/len1*100 + c2/len2/len1 *100), end='\r')
				
		try:        
			pickle.dump(embeds, open('data/faceboxes/embeds.pkl','wb'))
		except Exception as e:
			print('Could not write embedls.pkl\n',e)
		return embeds


	def embed(self, image_path,  threshold = 0.5):
			mp = self.model
			fd = self.facedetector
			m = self.map
			image_dir= self.image_path

			embeds = {}
			boxes, scores = m.embed_fb(fd, image_path, threshold)
			embeds[image_full_path] = boxes
				
			return embeds




	def gen_matches(self, embeds, gt ):
		#input:
		#	embds: a list of detections.
		#   gt: a list of detections from groundtruth
		#output:
		#	a list with 
		#		lists with
		#			two integers that represenst the indexes of correxponding
		#			rectangles in embeds and gt.
		m = self.map

		#embeds = pickle.load(open('data/faceboxes/embeds.pkl','rb'))
		#gt = pickle.load(open('data/faceboxes/images_dictionary.pkl','rb'))

		matches = {}
		lembeds = len(embeds)
		for c, i in enumerate(embeds):
			i2 = i.split('/')[-1]
			for c2, j in enumerate(gt):
				j2 = j.split('/')[-1]
				if i2 == j2:
					matches[j] = m.find_match3(embeds[i], gt[j])
					#print(' embeds :{}\n gt : {}\n'.format(embeds[i], gt[j]))
					#print('matches: {}'.format(matches))
					print('Generating Matches: {0:.2f} %'.format(c/lembeds*100), end='\r')
		try:
			pickle.dump(matches, open('data/faceboxes/matches.pkl','wb'))
		except erro:
			print('Could not write pickle file\n',erro)
		return matches


	def calcprecs(self, images, embeds, matches):
		#input:
		#	images: a dictionary: key= image identificator. value=list of rectangles. From groundtruth
		#	embeds: a dictionary: key= image identificator. value=list of rectangles. From detector.
		#	matches: a list with lists with two integers that represent the indexes of corresponding 	rectangles in embeds and images.
		#output:
		#	preci: value of precision (float)	
		#	reca: value of recall (float)
		m = map()
		precs, recs, numbers = [], [], []

		nrets = 0
		for i in images.values():
			nrets += len(i)

		preci, reca = [], []
		if matches != None:
			lm = len(matches)
		else:
			print('Nenhum match encontrado')
			return 0, 0

		for c, i in enumerate(matches):
			i2 = i.split('/')[-1]
			for j in embeds:
				j2 = j.split('/')[-1]
				if i2 == j2: 
					for k in images:
						k2 = k.split('/')[-1]
						if j2 == k2: 
							#print(i, matches[i], '\n')
							#print(embeds[j], images[k])
							l1, l2= m.analysis2(embeds[j], images[k], matches[i])
							precision, recall, nexamples  = m.prec_rec(l1,l2)
							precs.append(precision)
							recs.append(recall)
							numbers.append(nexamples)
							print('Calculating precisions {0:1.2f}%'.format(c/lm*100), end='\r')

		preci = m.mediap(precs, numbers)
		reca = m.mediap(recs, numbers)

		pickle.dump(preci,open('data/faceboxes/precision.pkl','wb'))
		pickle.dump(reca,open('data/faceboxes/recalls.pkl','wb'))


		print('precision ;A {} \n recalls : {}'.format(preci, reca))
		return preci, reca, nrets





	def gf_embed_to_dict(self, threshold, algoritmo='facedetector', batch_size = 100):

		#Performs face detection in all images specified in self.image_path and
		#returns a dictionary with:
		#keys: image identificators
		#values: lists of rectangles.
		#obs: uses gryfo's mssd face detector.

		m = map()
		prec, rec = [], []
		embeds = {}

		if algoritmo == 'facedetector' or algoritmo == 'gd':
			fd = self.gryfo_facedetector
			print('algoritmo: {}'.format(algoritmo))
		elif algoritmo == 'genericdetector' or algoritmo == 'gd':
			gd = self.gryfo_facedetector
			print('algoritmo: {}'.format(algoritmo))

		mp = '/weights/mssd_face'
		pkl = 'data/mssd/gembeds-' + str(threshold) + '.pkl'
		image_dir= '/media/nfs_datasets/WIDER/WIDER_train_val/images/'
		len1 = len(os.listdir(image_dir))
		for c, image_dir2 in enumerate(os.listdir(image_dir)):
			for c2, image_file in enumerate(os.listdir(os.path.join(image_dir, image_dir2))):
				len2 = len(os.listdir(os.path.join(image_dir, image_dir2)))
				image_full_path = os.path.join(image_dir, image_dir2, image_file)
				if algoritmo == 'facedetector' or algoritmo == 'gd':
					print('2------algoritmo: {}'.format(algoritmo))
					detections = m.embed_gr(fd, image_full_path, threshold) 
				elif algoritmo == 'genericdetector' or algoritmo == 'gd':
					print('2------algoritmo: {}'.format(algoritmo))
					detections = m.gf_generic_detect(gd, image_full_path, threshold)
				
				embeds[image_full_path] = detections
				#porcentagem abaixo é apenas aproximada
				print('Embedding: {0:0.2f}%'.format(c/len1*100 + c2/len2/len1 *100), end='\r') 
				if c2 % batch_size == 0:
					try:
						print('writen file {}'.format(pkl))
						pickle.dump(embeds, open(pkl,'wb'))
						return embeds
					except Exception as e:
						print('Could not write gryfo-embeds.pkl\n',e)
						return []

		try:
			print('writen file {}'.format(pkl))
			pickle.dump(embeds, open(pkl,'wb'))
			return embeds
		except Exception as e:
			print('Could not write gryfo-embeds.pkl\n',e)
			return []


				


	def gf_gen_matches(self, embeds, gt):

		#input:
		#	embds: a list of detections.
		#   gt: a list of detections from groundtruth
		#output:
		#	a list with 
		#		lists with
		#			two integers that represenst the indexes of correxponding
		#			rectangles in embeds and gt.

		m = self.map
		matches = {}
		if type(embeds) == None:
			print('Embeds vazio')
			return 1
		lembeds = len(embeds)
		for c, i in enumerate(embeds):
			i2 = i.split('/')[-1]
			for c2, j in enumerate(gt):
				j2 = j.split('/')[-1]
				if i2 == j2:
					matches[j] = m.find_match3(embeds[i], gt[j])
					#print(' embeds :{}\n gt : {}\n'.format(embeds[i], gt[j]))
					#print('matches: {}'.format(matches))
					print('Computing matches {0:.2f} %'.format(c/lembeds*100), end='\r')
		#try:
		#	pickle.dump(matches, open('data/mssd/gf-matches.pkl','wb'))
		#except erro:
		#	print('Could not write pickle file\n',erro)
		return matches

