.PHONY : install
.PHONY : clean

train: train.py
	python train.py &

evaluate: evaluate.py
	python evaluate.py

predict: predict.py
	python predict.py

install: transformers
	cd transformers && python setup.py install

transformers:
	git clone https://github.com/huggingface/transformers

scibert_scivocab_uncased: scibert_scivocab_uncased.tar
	tar -xvf scibert_scivocab_uncased.tar
	rm scibert_scivocab_uncased.tar

scibert_scivocab_uncased.tar:
	curl -o scibert_scivocab_uncased.tar \
	https://s3-us-west-2.amazonaws.com/ai2-s2-research/scibert/\
	huggingface_pytorch/scibert_scivocab_uncased.tar

model:
	mkdir model

clean:
	rm model/* result/*
