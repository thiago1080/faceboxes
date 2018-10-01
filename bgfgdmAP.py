#bgfgdmAP = (b)atches (g)ry(f)o (g)round(t)ruth (m)ean (A)verage (P)recision
from mAP import mAP
import os
import pickle
import numpy as np
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

mapi  = mAP(sys.argv[1])
nbatch = int(sys.argv[2])

#sys.argv[1] ---> 1 = wider 2 = monitora.

gt = mapi.gt_to_dict()
'''
for proto_threshold in range(9, 1, -1):
	threshold = proto_threshold / 10 
	sthreshold = str(threshold)
	print('Threshold = {}'.format(sthreshold))
	emb_dict = mapi.gf_embed_to_dict(threshold, algoritmo='genericdetector')
	pickle.dump(emb_dict, open('data/mssd/gembeds-'+ sthreshold + '.pkl','wb'))
'''

precisions, recalls, lnrets  = [], [], []
i1, i2 = 0, nbatch
gt2 = {}
embeds2 = {}
matches_dict2 = {}
gti = mapi.gtimages 
for step in range(i1, i2):
	for pt in range(9,0,-1):
		threshold = pt / 10 
		sthreshold = str(threshold)
		pklpath = 'data/mssd/gembeds-'+ sthreshold + '.pkl'
		if os.path.exists(pklpath):
			embeds = pickle.load(open(pklpath,'rb'))
		else:
			embeds = {}
		if gti[step]  in gt:
			gt2[gti[step]] = gt[gti[step]]
		if gti[step]  in embeds:
			embeds2[gti[step]] = embeds[gti[step]]
		matches_dict2 = mapi.gf_gen_matches(embeds2, gt2)
		precision, recall, nrets = mapi.calcprecs(gt2, embeds2, matches_dict2)
		precisions.append(precision)
		recalls.append(recall)
	i1 += nbatch
	i2 = (i2 + nbatch) if (i2+nbatch < mapi.nofimg) else mapi.nofimg
	lnrets.append(nrets)
	print(precisions)
