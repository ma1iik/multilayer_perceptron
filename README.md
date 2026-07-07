# multilayer-perceptron

A neural network written from scratch with NumPy (42 machine learning project).
It learns to classify breast cancer tumors as malignant or benign using the
Wisconsin diagnostic dataset.

No ML libraries — forward pass, backpropagation and gradient descent are all
implemented by hand in `core/`.

## Usage

```
python main.py --split                # shuffle + split data.csv into train/validation sets
python main.py --train                # train the network and save it to data/trained_model.npz
python main.py --predict              # load the model and evaluate it on the validation set
```

Training can be tweaked:

```
python main.py --train --layer 24 24 24 --epochs 500 --learning_rate 0.1 --loss categoricalCrossentropy
```

- `--layer` sets the hidden layer sizes (default: two layers of 24)
- `--loss` picks between binaryCrossentropy and categoricalCrossentropy

The model file stores the topology along with the weights, so predict
rebuilds the right network on its own.
