import argparse
import time
import pickle
import torch
import numpy as np

import matplotlib.pyplot as plt

from dataloaders.model_loader import RNNDetectionLoader
from dataloaders.chaotic_differentail_equation_data import LorenzDataSet

from models.deep.deepant_model import DeepAntNet
from models.deep.rnn_model import GatedCell
from models.deep.tcn_model import TCN



if __name__ == '__main__':
    pred = 1
    context = 50
    anomaly_error_magnitude = 0.125


    lorenz = LorenzDataSet(tmax=1500*0.01, anomalies=True)
    detectionloader = RNNDetectionLoader(lorenz, prediction_length=pred,
                                         context_length=context)
    context_list, pred_list = detectionloader.__getitem__(0)

    model_weights = torch.load(open('./trained/Model_rnn_lorenz_56_epochs_100.pt', 'rb'))
    model = GatedCell(input_size=1, hidden_size=256, output_size=1,
                      window_size=pred)
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
    # plt.plot(output[0, 0, :].detach().cpu().numpy(), label='net')
    numpy_error = error[0, 0, :].detach().cpu().numpy()
    anomalies = np.where(numpy_error > anomaly_error_magnitude, 1., 0.)
    plt.plot(anomalies, '.', label='anomalies')
    plt.title('error magnitude based detection')
    plt.legend()
    plt.show()
     

    print('stop')
        

