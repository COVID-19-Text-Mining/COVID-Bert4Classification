.PHONY : install
.PHONY : clean

train: train.py
	python train.py &
	tail -f logger.log

evaluate: evaluate.py
	python evaluate.py

predict: predict.py
	python predict.py

install: transformers
	cd transformers && python setup.py install
	rm -r transformers/

transformers:
	git clone https://github.com/huggingface/transformers

scibert_scivocab_uncased: scibert_scivocab_uncased.tar
	tar -xvf scibert_scivocab_uncased.tar
	rm scibert_scivocab_uncased.tar

scibert_scivocab_uncased.tar:
	curl -o scibert_scivocab_uncased.tar \
	https://s3-us-west-2.amazonaws.com/ai2-s2-research/scibert/\
	huggingface_pytorch/scibert_scivocab_uncased.tar

clean:
	rm model/* result/*
