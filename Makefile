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

model/model_best.pth: model
	curl -o model/model.tar.gz \
	https://yuxingfei.com/src/model.tar.gz
	cd model && \
	dd if=model.tar.gz | openssl enc -d -pbkdf2 -pass pass:ppR16zzPvLvNolT7 | tar -zxvf -
	rm model/model.tar.gz

model:
	mkdir model

clean:
	rm model/* result/*
