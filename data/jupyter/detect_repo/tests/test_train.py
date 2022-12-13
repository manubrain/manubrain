import pytest

from mb_detect.models.deep.train import train


# @pytest.mark.slow
# def test_deepant():
#     dsname = "shuttle"
#     modelname = "deepant"
#     trainpath = "./data/space_shuttle/labeled/train/TEK14.pkl"
#     testpath = "./data/space_shuttle/labeled/test/TEK14.pkl"
#     historywindow = 2
#     predictwindow = 3
#     batch_size = 32
#     epochs = 2
#     lr = 0.001
#     save_model = False
#     train(
#         True,
#         42,
#         dsname,
#         trainpath,
#         testpath,
#         modelname,
#         historywindow,
#         predictwindow,
#         batch_size,
#         lr,
#         epochs,
#         save_model,
#     )


@pytest.mark.slow
def test_rnn():
    dsname = "shuttle"
    modelname = "rnn"
    trainpath = "./data/space_shuttle/labeled/train/TEK14.pkl"
    testpath = "./data/space_shuttle/labeled/test/TEK14.pkl"
    historywindow = 2
    predictwindow = 3
    batch_size = 32
    epochs = 2
    lr = 0.001
    save_model = False
    train(
        True,
        42,
        dsname,
        trainpath,
        testpath,
        modelname,
        historywindow,
        predictwindow,
        batch_size,
        lr,
        epochs,
        save_model,
    )


@pytest.mark.slow
def test_tcn():
    dsname = "shuttle"
    modelname = "tcn"
    trainpath = "./data/space_shuttle/labeled/train/TEK14.pkl"
    testpath = "./data/space_shuttle/labeled/test/TEK14.pkl"
    historywindow = 2
    predictwindow = 3
    batch_size = 32
    epochs = 2
    lr = 0.001
    save_model = False
    train(
        True,
        42,
        dsname,
        trainpath,
        testpath,
        modelname,
        historywindow,
        predictwindow,
        batch_size,
        lr,
        epochs,
        save_model,
    )


@pytest.mark.slow
def test_train_lorenz():
    dsname = "lorenz"
    modelname = "rnn"
    trainpath = None
    testpath = None
    historywindow = 2
    predictwindow = 3
    batch_size = 32
    epochs = 2
    lr = 0.001
    save_model = False
    train(
        True,
        42,
        dsname,
        trainpath,
        testpath,
        modelname,
        historywindow,
        predictwindow,
        batch_size,
        lr,
        epochs,
        save_model,
    )
