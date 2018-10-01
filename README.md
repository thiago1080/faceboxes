Resultados do mAP (Mean Average Precision).
	Faceboxes / dataset WIDER: 0.64
	MSSD	/ dataset WIDER: 0.59
	Generic Detector yolov2 / setado para detectar 'person' / dataset WIDER: 0.61



IMPORTANTE: O facedetector mssd é utilizado diretamente da plataforma Gryfo, enquanto o faceboxes está no presente diretório (é o arquivo face_detector.py)

1.Verifique o arquivo datasets.json se os diretórios estão corretos na sua máquina.

Para calcular o mAP do faceboxes:
	1. executar calcmap
		$ python3 'um_dos_scripts.py' '1 ou 2'
			(1) Dataset WIDER
			(2) Dataset MONITORA
		'um_dos_scripts.py'
			calcmap.py -> faceboxes
			gf_calcmap.py -> mssd
			gfgdmap.py -> genericdetector yolov2

Caso não obtenha o mAP ao final da execução, copie as precisões e recalls do output e edite o script precs2map.py, seguindo as instruções dos comentários desse script.


