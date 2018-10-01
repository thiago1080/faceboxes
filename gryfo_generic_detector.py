#Código para calcular mAP do generic detector yolov2 nos datasets WIDER ou MONITORA
from mAP import mAP
import os
import pickle
import numpy as np
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

mapi  = mAP(sys.argv[1])

#sys.argv[1] ---> 1 = wider 2 = monitora.

gt = mapi.gt_to_dict()
for proto_threshold in range(9, 1, -1):
	threshold = proto_threshold / 10 
	sthreshold = str(threshold)
	print('Threshold = {}'.format(sthreshold))
	emb_dict = mapi.gf_embed_to_dict(threshold, algoritmo='genericdetector')
	pickle.dump(emb_dict, open('data/mssd/gembeds-'+ sthreshold + '.pkl','wb'))

precisions, recalls  = [], []
for proto_threshold in range(9, 1, -1):
	threshold = proto_threshold / 10 
	sthreshold = str(threshold)
	pklpath = 'data/mssd/gembeds-'+ sthreshold + '.pkl'
	matchpath = 'data/mssd/matches-'+ sthreshold + '.pkl'
	print('Reading {}'.format(pklpath))
	emb_dict = pickle.load(open(pklpath,'rb'))
	matches_dict = mapi.gf_gen_matches(emb_dict, gt)
	pickle.dump(matches_dict, open(matchpath,'wb'))
	print('Matches len = {}'.format(len(matches_dict)))
	precision, recall = mapi.calcprecs(gt, emb_dict, matches_dict)
	precisions.append(precision)
	recalls.append(recall)
pickle.dump(precisions , open('data/mssd/precisions.pkl','wb'))
pickle.dump(recalls, open('data/mssd/recalls.pkl','wb'))
APs = mapi.map.mAp(precisions)
pickle.dump(APs,  open('data/mssd/APs.pkl','wb'))
map_value = np.average(APs)
pickle.dump(map_value , open('data/mssd/mAP.pkl','wb'))
