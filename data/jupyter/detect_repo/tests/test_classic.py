import pytest

from mb_detect.dataloader.iter_reader import NabIter
from mb_detect.models.classic.arima_anomaly import ArimaAnomaly


@pytest.mark.slow
def test_arma():
    reader = NabIter()
    detector = ArimaAnomaly()
    for data, _ in reader:
        value = data["value"].to_numpy()
        detector.fit_predict(value)
        break


if __name__ == "__main__":
    test_arma()
