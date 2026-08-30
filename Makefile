PYTHON := python

CACHE     := $(shell find . -name __pycache__ -type d)
GENERATED := data/train_data.csv data/val_data.csv data/norm_params.npz \
             data/trained_model.npz data/learning_curves.png data/model_comparison.png
EVAL      := data/data_training.csv data/data_test.csv

.PHONY: help all split train predict compare eval clean fclean re

help:
	@echo "make all      split + train + predict"
	@echo "make split    cut data.csv into training and validation sets"
	@echo "make train    train a model and save it"
	@echo "make predict  score the test set with the saved model"
	@echo "make compare  overlay every saved run on one graph"
	@echo "make eval     run 42's evaluation.py on data.csv"
	@echo "make clean    remove caches and training intermediates"
	@echo "make fclean   clean + model, history, plots, evaluation.py output"
	@echo "make re       fclean + all"
	@echo
	@echo "for anything non-default, call main.py directly:"
	@echo "  python main.py --train --layer 24 24 24 --optimizer adam --early_stop"

all: split train predict

split:
	$(PYTHON) main.py --split

train:
	$(PYTHON) main.py --train

predict:
	$(PYTHON) main.py --predict

compare:
	$(PYTHON) main.py --compare

eval:
	cd data && $(PYTHON) ../evaluation.py

clean:
	rm -rf $(CACHE)
	rm -f data/train_data.csv data/val_data.csv data/norm_params.npz

fclean: clean
	rm -f $(GENERATED) $(EVAL)
	rm -rf data/history

re: fclean all
