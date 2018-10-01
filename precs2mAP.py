from map import map
import numpy as np
m = map()
#escreva ,de forma hardcoded, um dicionário como o abaixo, no formato { recall : precision }
#gfprecsd = {0.171: 0.659, 0.224:0.656, 0.259:0.643, 0.283:0.620, 0.303: 0.595, 0.318: 0.565, 0.330: 0.524, 0.336:0.468}

for i in sorted(gfprecsd.keys()):
	precs.append(gfprecsd[i])
ap0  = m.mAp(precs)
print(ap0)
ap = ap0[0]
print(ap)
meanAP = np.average(ap)
print(meanAP)

