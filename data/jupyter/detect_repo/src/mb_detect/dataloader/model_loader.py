import torch
from torch.utils.data import Dataset

import numpy as np


def compute_mean_and_std(data_loader):
    dataset = data_loader.dataset
    input_lst = []
    for no in range(dataset.__len__()):
        item = dataset.__getitem__(no)
        input_lst.append(item)
    train_input_data = torch.stack(input_lst, 0)

    # calculate mean and std in double to avoid precision problems
    mean = torch.mean(train_input_data.double()).float()
    std = torch.std(train_input_data.double()).float()

    return mean, std


class RNNPredictionLoader(Dataset):
    def __init__(self, dataset: Dataset, prediction_length: int, context_length: int):
        """Split an input data-set into past an future components.

        dataset.__getitem__ is expected to have time as the last dimension.

        Args:
            dataset (torch.utils.data.Dataset): [description]
            prediction_length (): [description]
            context_length ([type]): [description]
        """
        self.dataset = dataset.loc[:, dataset.columns != "is_anomaly"]
        self.prediction_length = prediction_length
        self.context_length = context_length

    def __len__(self) -> int:
        return len(self.dataset) - self.context_length - self.prediction_length

    def __getitem__(self, index: int) -> tuple:
        """Return a future and context pair from dataset.

        Args:
            index (int): Position in data set.

        Returns:
            tuple: [description]
        """
        data = self.dataset.loc[index:]
        history = data[0 : self.context_length]
        predict = data[
            self.context_length : self.context_length + self.prediction_length
        ]
        history = history.to_numpy(dtype=np.float32)
        predict = predict.to_numpy(dtype=np.float32)
        return history.T, predict.T


class RNNDetectionLoader(Dataset):
    def __init__(self, dataset, prediction_length, context_length):
        self.dataset = dataset
        self.prediction_length = prediction_length
        self.context_length = context_length

    def __len__(self) -> int:
        return self.dataset.__len__()

    def __getitem__(self, index: int) -> tuple:
        data = self.dataset.__getitem__(index)
        total_length = self.context_length + self.prediction_length
        windows = (data.shape[-1] - total_length * 2) // self.prediction_length
        context_list = []
        future_list = []
        for w in range(windows):
            start_context = w * self.prediction_length + self.context_length
            start_prediction = start_context + self.context_length
            context = data[:, start_context : (start_context + self.context_length)]
            target_prediction = data[
                :,
                start_prediction : (start_prediction + self.prediction_length),
            ]
            context_list.append(context)
            future_list.append(target_prediction)
        return context_list, future_list
