import argparse
import pickle
import time

from mb_detect.models.deep import train


def parse_cmd():
    parser = argparse.ArgumentParser(description="DeepAnt Model")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=56,
        metavar="N",
        help="input batch size for training (default: 12)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        metavar="N",
        help="number of epochs to train (default: 35)",
    )
    parser.add_argument(
        "--no-cuda",
        action="store_true",
        default=False,
        help="disables CUDA training",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        metavar="LR",
        help="learning rate (default: 0.001",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.0,
        metavar="LR",
        help="momentum rate (default: 0.0)",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        default=True,
        help="For Saving the current Model",
    )
    parser.add_argument("--trainpath", default=None, help="train dataset path")
    parser.add_argument("--testpath", default=None, help="test dataset path")
    parser.add_argument("--dataseperator", default="\\s", help="dataset seperator")
    parser.add_argument(
        "--datasetname",
        choices=["shuttle", "nab", "yahoo", "lorenz"],
        default="lorenz",
        help="Dataset to choose",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=int(time.time()),
        metavar="S",
        help="random seed (default: int(time.time()))",
    )
    parser.add_argument(
        "--model",
        choices=["deepant", "rnn", "tcn"],
        default="rnn",
        help="Architecture to choose",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        metavar="Threshold",
        help="Threshold for anomaly score (default: 0.8)",
    )
    parser.add_argument(
        "--historywindow",
        type=int,
        default=50,
        metavar="HW",
        help="History window (defualt: 100)",
    )
    parser.add_argument(
        "--predictwindow",
        type=int,
        default=5,
        metavar="HW",
        help="Predict window (defualt: 100)",
    )
    return parser.parse_args()


def main():
    # vargs = sys.argv[1:]
    args = parse_cmd()
    print(args, flush=True)

    no_cuda = args.no_cuda
    seed = args.seed
    datasetname = args.datasetname
    trainpath = args.trainpath
    testpath = args.testpath
    _model = args.model
    historywindow = args.historywindow
    predictwindow = args.predictwindow
    batch_size = args.batch_size
    lr = args.lr
    epochs = args.epochs
    _save_model = args.save_model

    train.train(
        no_cuda,
        seed,
        datasetname,
        trainpath,
        testpath,
        _model,
        historywindow,
        predictwindow,
        batch_size,
        lr,
        epochs,
        _save_model,
    )
    name = (
        "trained"
        + "/"
        + "Model_"
        + args.model
        + "_"
        + args.datasetname
        + "_"
        + str(args.batch_size)
        + "_epochs_"
        + str(args.epochs)
    )
    with open(name + ".args", "wb") as model_file:
        pickle.dump(args, model_file)


if __name__ == "__main__":
    main()
