from gryfo.blocks import GenericDetector
from map import map
import cv2
m = map()
ret = []
gd = GenericDetector(classes=["person"])
img = '../images2/0_Parade_Parade_0_628.jpg'
img = cv2.imread(img)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
dets = gd.detect(img)
for det in dets:
	ret = m.det2yx(det,img)
	print(ret)
