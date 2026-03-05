.PHONY: train eval

train:
	python -m src.train

eval:
	python -m src.evaluate
