import argparse
from phases.split import split_dataset
from phases.train import train_network
from phases.predict import make_pred
from bonus.compare import compare_runs
from bonus.optimizers import DEFAULT_LR

def check_hyperparams(args):
    if args.epochs < 1:
        raise SystemExit("--epochs must be at least 1")
    if args.learning_rate <= 0:
        raise SystemExit("--learning_rate must be greater than 0")
    if args.batch_size < 1:
        raise SystemExit("--batch_size must be at least 1")
    if args.patience < 1:
        raise SystemExit("--patience must be at least 1")
    if any(size < 1 for size in args.layer):
        raise SystemExit("--layer sizes must all be at least 1")
    if len(args.layer) < 2:
        print(f"warning: {len(args.layer)} hidden layer, the subject asks for at least two")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multilayer Perceptron from Scratch")
    
    parser.add_argument('--split', action='store_true')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--predict', action='store_true')
    parser.add_argument('--compare', action='store_true', help="Overlay every saved run on one graph")
    
    # hyperparameters
    parser.add_argument('--epochs', type=int, default=200, help="Epochs number for training")
    parser.add_argument('--learning_rate', type=float, default=None,
                        help="Learning rate (default 0.05 for sgd, 0.001 for adam)")
    parser.add_argument('--batch_size', type=int, default=8, help="Batch size for training")
    parser.add_argument('--layer', type=int, nargs='+', default=[24, 24], help="Hidden layer sizes (e.g. --layer 24 24)")
    parser.add_argument('--seed', type=int, default=None, help="Fix the seed to reproduce a run (default: random each time)")
    parser.add_argument('--dataset', type=str, default='data/data.csv', help="File for --split to cut up")
    parser.add_argument('--loss', type=str, default='binaryCrossentropy',
                        choices=['binaryCrossentropy', 'categoricalCrossentropy'], help="Loss function")

    # bonus
    parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd', 'adam'], help="Weight update rule")
    parser.add_argument('--early_stop', action='store_true', help="Stop early once validation loss stops improving")
    parser.add_argument('--patience', type=int, default=30, help="Epochs without improvement tolerated by --early_stop")
    args = parser.parse_args()

    if args.learning_rate is None:
        args.learning_rate = DEFAULT_LR[args.optimizer]

    if args.split:
        split_dataset(args.dataset)
    elif args.train:
        check_hyperparams(args)
        print(f"Training for {args.epochs} epochs with {args.optimizer}, LR: {args.learning_rate}...")
        train_network(epochs=args.epochs, lr=args.learning_rate, hlayers = args.layer, loss_name=args.loss, batch_s = args.batch_size,
                      early_stop=args.early_stop, patience=args.patience, optimizer_name=args.optimizer,
                      seed=args.seed)
    elif args.predict:
        make_pred()
    elif args.compare:
        compare_runs()
    else:
        parser.print_help()