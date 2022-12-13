# import argparse
# import time
# import pickle
# import argparse
import pickle
import torch
import numpy as np
import sys

import matplotlib.pyplot as plt

from dataloaders.model_loader import RNNDetectionLoader
from dataloaders.chaotic_differentail_equation_data import LorenzDataSet

from models.deep.deepant_model import DeepAntNet
from train import parse_cmd
from models.deep.rnn_model import GatedCell
from models.deep.tcn_model import TCN

def load_args(argspath):
    args = parse_cmd()
    with open(argspath, 'rb') as f:
        args.Namespace = pickle.load(f)
    return args

if __name__ == '__main__':
    argspath = './trained/Model_deepant_shuttle_64_epochs_150.args'
    anomaly_error_magnitude = 0.25

    # Load argmodel argparse
    args = load_args(argspath)
    print()
    print(args, flush=True)
    print('######')
    print()
    # Dataloader creation
    if args.datasetname == 'lorenz':
        lorenz = LorenzDataSet(tmax=300 * 0.01, anomalies=True)
        detectionloader = RNNDetectionLoader(lorenz, prediction_length=args.predictwindow,
                                         context_length=args.historywindow)
        context_list, pred_list = detectionloader.__getitem__(0)
    elif args.datasetname == 'shuttle':
        pass
    
    model = None
    model_weights = None
    # Model instantiation
    if args.model == 'deepant':
        model = DeepAntNet(predict_window=args.predictwindow, n_dim=1)
    elif args.model == 'rnn':
        model = GatedCell(input_size=1, hidden_size=256, output_size=1, 
                          window_size=args.predictwindow)
    elif args.model == 'tcn':
        no_hidunits = 75
        levels = 3
        model = TCN(input_size=1, predict_window=args.predictwindow,
                    num_channels=[no_hidunits] * levels, kernel_size=2,
                    dropout=0.2)
    print(model.eval(), flush=True)
    # Load model weights
    model_weights = torch.load(open('./Model_deepant_lorenz_64_epochs_150.pt', 'rb'), map_location=torch.device('cpu'))
    model.load_state_dict(model_weights)

    context_list, prediction_list = detectionloader.__getitem__(0)
    # go through the test signal.
    output_list = []
    error_list = []
    measured_future_list = []
    for position, context in enumerate(context_list):
        measured_future = prediction_list[position].unsqueeze(0)
        context = context.unsqueeze(0)
        predicted_future = model(context)
        absolute_error = torch.abs(measured_future - predicted_future)
        output_list.append(predicted_future)
        error_list.append(absolute_error)
        measured_future_list.append(measured_future)
        # plt.plot(torch.cat([context[0,0, :], predicted_future[0,0,:].detach()]))
        # plt.plot(torch.cat([context[0,0, :], measured_future[0,0,:].detach()]))
        # plt.show()
    measured_signal = torch.cat(measured_future_list, -1)
    output = torch.cat(output_list, -1)
    error = torch.cat(error_list, -1)

    # detect anomalies based on error magnitude
    plt.plot(measured_signal[0, 0, :].numpy(), label='signal')
    plt.plot(output[0, 0, :].detach().cpu().numpy(), label='net')
    numpy_error = error[0, 0, :].detach().cpu().numpy()
    anomalies = np.where(numpy_error > anomaly_error_magnitude, 1., 0.)
    plt.plot(anomalies, '.', label='anomalies')
    plt.title('error magnitude based detection')
    plt.legend()
    plt.savefig('anomaly.png')
    print('stop')
