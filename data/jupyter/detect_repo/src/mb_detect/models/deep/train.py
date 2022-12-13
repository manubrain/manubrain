import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from ...dataloader import ds_loader
from ...dataloader.model_loader import RNNPredictionLoader
from ...dataloader.preprocess import Preprocess
from .rnn_model import GatedCell
from .tcn_model import TCN


def train_loop(model, optimizer, loss_fun, dataloader, writer, epochs, use_cuda):
    if use_cuda:
        model.cuda()

    # Training loop
    n_iter = 0
    model.train()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        tidx = 0
        # Batch training
        for idx, (train_data, target_predictions) in enumerate(dataloader):
            if use_cuda:
                train_data = train_data.cuda()
                target_predictions = target_predictions.cuda()

            optimizer.zero_grad()
            # Forward pass
            predictions = model(train_data)
            loss = loss_fun(predictions, target_predictions)
            epoch_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), 0.5)
            optimizer.step()
            # print(loss.item())
            writer.add_scalar("training/loss", loss, n_iter)

            figure = plt.figure()
            target = target_predictions[0, 0, ...].detach().cpu().numpy()
            predictions = predictions[0, 0, ...].detach().cpu().numpy()
            abs_error = np.abs(target - predictions)
            plt.plot(target, label="target")
            plt.plot(predictions, label="network")
            plt.plot(abs_error, label="abs_error")
            plt.title("output versus target")
            plt.legend()
            writer.add_figure("training/train_prediction", figure, global_step=n_iter)
            plt.close()

            n_iter += 1
            tidx = idx
        # Each sample loss in the batch
        epoch_loss /= tidx + 1
        print(
            "Epoch:{}/{}=====>Loss:{}".format(epoch, epochs, epoch_loss),
            flush=True,
        )
        writer.add_scalar("training/epoch", epoch, n_iter)
        writer.add_scalar("training/loss_per_epoch", epoch_loss, n_iter)


def save_model(model, _model, datasetname, batch_size, epochs):
    if not os.path.exists("./trained"):
        os.mkdir("./trained")
    name = (
        "trained"
        + "/"
        + "Model_"
        + _model
        + "_"
        + datasetname
        + "_"
        + str(batch_size)
        + "_epochs_"
        + str(epochs)
    )
    torch.save(model.state_dict(), name + ".pt")
    print("model saved")


def test_model(model, testloader, loss_fun, use_cuda):
    # Testing code
    model.eval()
    # auc_scores = []
    mse_error = []
    print()
    print("Testing loop")
    with torch.no_grad():
        for _, (data, targets) in enumerate(testloader):
            if use_cuda:
                data = data.cuda()
                targets = targets.cuda()
            preds = model(data)
            preds = preds.reshape(targets.shape)
            error = loss_fun(preds, targets)
            mse_error.append(error.detach().cpu().numpy())
    print("Mean MSE loss: ", np.mean(mse_error))


# TODO put this into modules and behind interfaces
def train(
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
):
    torch.autograd.set_detect_anomaly(True)
    use_cuda = not no_cuda and torch.cuda.is_available()
    print("Cuda: ", torch.cuda.is_available())
    torch.manual_seed(seed)
    print("Use cuda: ", use_cuda, flush=True)
    ##############################################################################
    # Load the specific dataset
    prep = Preprocess()
    if datasetname == "nab":
        train_df = ds_loader.nab(trainpath)
        test_df = ds_loader.nab(testpath)
        train_df = prep.normalize(prep.roll(train_df, 7))
        test_df = prep.normalize(prep.roll(test_df, 7))
    elif datasetname == "shuttle":
        train_df = ds_loader.discord_pkl(trainpath)
        test_df = ds_loader.discord_pkl(testpath)
    elif datasetname == "lorenz":
        train_df = ds_loader.lorenz_data(anomalies=True)
        test_df = ds_loader.lorenz_data(anomalies=True)
    else:
        raise Exception("Improper dataset")
    ##############################################################################
    # Load the preprocessed data into pytorch dataloader
    # Note: Input to the dataloader must be shape (input_dimension, len(dataset))
    # if _model == "rnn" or _model == "tcn":
    loader = RNNPredictionLoader
    data_dim = sum([column.startswith("value") for column in train_df.columns])

    print(train_df.shape, test_df.shape)
    dataset = loader(
        train_df,
        context_length=historywindow,
        prediction_length=predictwindow,
    )
    test_dataset = loader(
        test_df,
        context_length=historywindow,
        prediction_length=predictwindow,
    )
    # Dataloader creation
    trainloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )
    testloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )
    ##########################################################################
    # Model
    if _model == "rnn":
        model = GatedCell(
            input_size=data_dim,
            hidden_size=256,
            output_size=data_dim,
            window_size=predictwindow,
        )
    elif _model == "tcn":
        no_hidunits = 75
        levels = 3
        print("In the temporal conv network.....", flush=True)
        model = TCN(
            input_size=data_dim,
            predict_window=predictwindow,
            num_channels=[no_hidunits] * levels,
            kernel_size=2,
            dropout=0.2,
        )
    else:
        raise ValueError
    if use_cuda:
        model = model.cuda()
    print("Model creation done", flush=True)
    #############################################################################
    # Loss and optimizer declaration

    loss_fun = torch.nn.MSELoss(reduction="sum")
    optimizer = Adam(model.parameters(), lr=lr)

    # Logfile management.
    writer = SummaryWriter(comment="_" + _model)

    train_loop(model, optimizer, loss_fun, trainloader, writer, epochs, use_cuda)

    if _save_model:
        save_model(model, _model, datasetname, batch_size, epochs)

    test_model(model, testloader, loss_fun, use_cuda)
