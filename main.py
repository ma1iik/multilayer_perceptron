import argparse
from split import split_dataset
from train import train_network
from predict import make_pred

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multilayer Perceptron from Scratch")
    
    parser.add_argument('--split', action='store_true')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--predict', action='store_true')
    
    # hyperparameters
    parser.add_argument('--epochs', type=int, default=100, help="Epochs number for training")
    parser.add_argument('--learning_rate', type=float, default=0.1, help="Learning rate for gradient descent")
    parser.add_argument('--batch_size', type=int, default=8, help="Batch size for training")
    parser.add_argument('--layer', type=int, nargs='+', default=[24, 24], help="Hidden layer sizes (e.g. --layer 24 24)")
    parser.add_argument('--loss', type=str, default='binaryCrossentropy',
                        choices=['binaryCrossentropy', 'categoricalCrossentropy'], help="Loss function")

    # bonus
    parser.add_argument('--early_stop', action='store_true', help="Stop early once validation loss stops improving")
    parser.add_argument('--patience', type=int, default=30, help="Epochs without improvement tolerated by --early_stop")
    args = parser.parse_args()

    if args.split:
        split_dataset()
    elif args.train:
        print(f"Training for {args.epochs} epochs with LR: {args.learning_rate}...")
        train_network(epochs=args.epochs, lr=args.learning_rate, hlayers = args.layer, loss_name=args.loss, batch_s = args.batch_size,
                      early_stop=args.early_stop, patience=args.patience)
    elif args.predict:
        print("Predicting results...")
        make_pred()
    else:
        parser.print_help()