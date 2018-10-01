from mAP import mAP
import os
import pickle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
mapi = mAP(sys.argv[1])

gt = mapi.gt_to_dict()
'''
for proto_threshold in range(11, 0, -1):
	threshold = proto_threshold / 10 
	sthreshold = str(threshold)
	emb_dict = mapi.embed_to_dict(threshold)
	pickle.dump(emb_dict, open('data/faceboxes/embeds-'+ sthreshold + '.pkl','wb'))
'''
tiny_emb = {}
tiny_gt = {}
precisions, recalls  = [], []
for proto_threshold in range(10, 0, -1):
	threshold = proto_threshold / 10 
	sthreshold = str(threshold)
	pklpath = 'data/faceboxes/embeds-'+ sthreshold + '.pkl'
	print('lendo aquivo --> {}'.format(pklpath))
	emb_dict = pickle.load(open(pklpath,'rb'))
	for c,key in enumerate(emb_dict):
		if c < 50:
			tiny_emb[key] = emb_dict[key]
	for i in tiny_emb:
		for j in gt:
			if i.split('/')[-1] == j.split('/')[-1]:
				tiny_gt[j] = gt[j]
			
	matches_dict = mapi.gen_matches(emb_dict, gt)
	precision, recall = mapi.calcprecs(gt, emb_dict, matches_dict)
	precisions.append(precision)
	recalls.append(recall)
pickle.dump(precisions , open('data/faceboxes/precisions.pkl','wb'))
pickle.dump(recalls, open('data/faceboxes/recalls.pkl','wb'))

APs = mapi.map.mAp(precisions)
pickle.dump(APs,  open('data/faceboxes/APs.pkl','wb'))
map_value = np.average(APs)
pickle.dump(map_value , open('data/faceboxes/mAP.pkl','wb'))
