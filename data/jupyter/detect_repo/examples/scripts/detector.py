# TODO "--modelpath required", unclear instruction
# TODO n_dim missing

import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from mb_detect.dataloader.model_loader import DeepAntLoader, RNNPredictionLoader
from mb_detect.dataloader.ds_loader import shuttle
from mb_detect.dataloader.chaotic_pde import LorenzDataSet
from mb_detect.models.deep.deepant_model import DeepAntNet
from mb_detect.models.deep.rnn_model import GatedCell
from mb_detect.models.deep.tcn_model import TCN

from mb_detect.anomaly_detection.anomaly_detector_module import AnomalyDetection

print("Loaded modules", flush=True)


def parse_cmd():
    parser = argparse.ArgumentParser(description="DeepAnt Model")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        metavar="N",
        help="input batch size for training (default: 12)",
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument("--testpath", default=None, help="test dataset path")
    parser.add_argument("--dataseperator", default="\\s", help="dataset seperator")
    parser.add_argument(
        "--datasetname",
        choices=["shuttle", "nab", "yahoo", "lorenz"],
        default="shuttle",
        help="Dataset to choose",
    )
    parser.add_argument(
        "--model",
        choices=["deepant", "rnn", "tcn"],
        default="rnn",
        help="Architecture to choose",
    )
    parser.add_argument(
        "--historywindow",
        type=int,
        default=100,
        metavar="HW",
        help="History window (defualt: 100)",
    )
    parser.add_argument(
        "--predictwindow",
        type=int,
        default=50,
        metavar="HW",
        help="Predict window (defualt: 50)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        metavar="Threshold",
        help="Threshold for anomaly score (default: 0.8)",
    )
    parser.add_argument(
        "--modelpath", required=True, help="Path to the saved model weights"
    )
    parser.add_argument(
        "--distancetype",
        default="euclidean",
        choices=["euclidean", "multigaussianerror"],
        help="type of distance technique for anomaly detection",
    )
    return parser.parse_args()


def main():
    torch.autograd.set_detect_anomaly(True)
    # Training parameter settings
    args = parse_cmd()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    print("Cuda: ", torch.cuda.is_available())
    print(args, flush=True)
    print("Use cuda: ", use_cuda, flush=True)

    # Test dataset loading
    # Test dataset loading
    test_datasetloader = DatasetLoader(args.testpath, args.dataseperator)
    func_name = None
    # Fucntion name to load the data
    if args.datasetname == "shuttle":
        func_name = test_datasetloader.shuttle_preprocessing
        b_size = 4
    elif args.datasetname == "lorenz":
        total_seq_len = args.historywindow + args.predictwindow
        test_datasetloader = LorenzDataSet(
            args.historywindow, args.predictwindow, tmax=total_seq_len * 0.01
        )
        # func_name = test_datasetloader.data_function
        b_size = args.batch_size
    else:
        print("Improper dataset")

    if args.model == "cnn":
        test_dataset = DeepAntLoader(
            func_name,
            history_window=args.historywindow,
            predict_window=args.predictwindow,
        )
    elif args.model == "rnn" or args.model == "tcn":
        test_dataset = RNNPredictionLoader(
            test_datasetloader,
            context_length=args.historywindow,
            prediction_length=args.predictwindow,
        )
    testloader = DataLoader(
        test_dataset, batch_size=b_size, shuffle=False, num_workers=4
    )
    print("Test dataloader done", flush=True)
    #############################################################################
    # Model
    if args.model == "deepant":
        model = DeepAntNet(predict_window=args.predictwindow, n_dim=n_dim)
    elif args.model == "rnn":
        model = GatedCell(
            input_size=1,
            hidden_size=1024,
            output_size=1,
            window_size=args.predictwindow,
        )
    elif args.model == "tcn":
        no_hidunits = 50
        levels = 3
        print("In the temporal conv network.....", flush=True)
        model = TCN(
            input_size=1,
            predict_window=args.predictwindow,
            num_channels=[no_hidunits] * levels,
            kernel_size=2,
            dropout=0.2,
        )
    else:
        raise ValueError

    if use_cuda:
        model = model.cuda()
        model.load_state_dict(torch.load(args.modelpath))
    else:
        model.load_state_dict(torch.load(args.modelpath, map_location="cpu"))
    print("Model done", flush=True)

    # Anomaly detection instantiation
    anomaly = AnomalyDetection()
    model.eval()
    with torch.no_grad():
        for idx, (data, targets) in enumerate(testloader):
            if use_cuda:
                data = data.cuda()
                targets = targets.cuda()
            preds = model(data)
            preds = preds.reshape(targets.shape)
            batchpred_labels = []
            for i in range(0, preds.shape[0]):
                if args.distancetype == "euclidean":
                    _, pred_labels = anomaly.euclidean_distance(
                        preds[i, :, :].cpu(), targets[i, :, :].cpu(), args.threshold
                    )
                    batchpred_labels.append(pred_labels)
            batchpred_labels = np.asarray(batchpred_labels)
    print("done...")


if __name__ == "__main__":
    main()
