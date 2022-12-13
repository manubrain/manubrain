from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM


class SKModel:
    def fit(self, X):
        self._df = X
        cs = [c for c in self._df.columns if c.startswith("value")]
        self._values = self._df[cs].to_numpy()
        self._labels = self._model.fit_predict(self._values)
        self._df[self._column_name] = self._labels != -1

    def fit_predict(self, X):
        self.fit(X)
        return self._df

    def predict(self, X):
        cs = [c for c in X.columns if c.startswith("value")]
        _values = X[cs].to_numpy()
        return self._model.predict(_values) != -1


class SK_IsoForest(SKModel):
    def __init__(self, n_estimators: int = 10, warm_start: bool = True):
        self._column_name = "is_anomaly_SKIsoForest"
        self._model = IsolationForest(n_estimators=n_estimators, warm_start=warm_start)


class SK_OneClassSVM(SKModel):
    def __init__(self, nu: float = 0.5, kernel: str = "rbf", gamma: str = "scale"):
        self._column_name = "is_anomaly_SKOneClassSVM"
        self._model = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)


class SK_Dbscan(SKModel):
    def __init__(self, eps: float = 15, min_samples: int = 100):
        self._column_name = "is_anomaly_SKDbscan"
        self._model = DBSCAN(eps=eps, min_samples=min_samples)
