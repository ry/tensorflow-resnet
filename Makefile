train:
	python resnet.py --images ~/data/ILSVRC2012/ILSVRC2012_img_train.txt --batch_size=25

clean:
	rm -f checkpoint*
	rm -f log/*

.PHONY: clean train
