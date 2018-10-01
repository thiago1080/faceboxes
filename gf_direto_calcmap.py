from mAP import mAP
import os
import pickle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
mapi = mAP()


gt = mapi.gt_to_dict()

precisions, recalls  = [], []
for proto_threshold in range(10, 0, -1):
	threshold = proto_threshold / 10 
	emb_dict = mapi.gf_embed_to_dict(threshold)
	matches_dict = mapi.g_ogen_matches(emb_dict, gt)
	precision, recall = mapi.calcprecs(gt, emb_dict, matches_dict)
	precisions.append(precision)
	recalls.append(recall)

mAP_value = mapi.map.mAp(precisions)
