import pytest
from sklearn.ensemble import IsolationForest

from mb_detect.anomaly_detection import benchmarker
from mb_detect.dataloader import iter_reader


@pytest.mark.slow
def test_unsup():
    reader = iter_reader.NabIter()
    data, metadata = reader.__iter__().__next__()
    model = IsolationForest(n_estimators=100, max_samples=0.5, warm_start=False)
    benchmarker.test_unsupervised(data, model, window_size=50, plot=False)
