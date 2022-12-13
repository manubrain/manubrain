import pytest

from mb_detect.dataloader import ds_loader, iter_reader


@pytest.mark.slow
def test_nab():
    reader = iter_reader.NabIter()
    for data, metadata in reader:
        print(metadata, data.shape)


@pytest.mark.slow
def test_yahoo():
    reader = iter_reader.YahooIter()
    for data, metadata in reader:
        print(metadata, data.shape)


@pytest.mark.slow
def test_deepant_shuttle():
    fp = "./data/deepant-shuttle/shuttle_test.txt"
    df = ds_loader.shuttle(fp)
    print("read from", fp, ":")
    print(df.columns, df.shape)


if __name__ == "__main__":
    test_deepant_shuttle()
