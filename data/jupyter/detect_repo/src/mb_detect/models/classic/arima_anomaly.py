import numpy as np
from statsmodels.tsa.arima.model import ARIMA

# from sklearn.model_selection import train_test_split
# train_data, test_data = train_test_split(data, test_size=test_split, shuffle=False)


class ArimaAnomaly:
    def __init__(self, order=(2, 1, 2)):
        self.order = order

    def get_values(self, X):
        cs = [c for c in X.columns if c.startswith("value")]
        cs = cs[0]  # Standard is one-dim!
        return X[cs].to_numpy()

    def fit(self, X):
        self._df = X
        values = self.get_values(X)
        ts = np.arange(len(values))
        self._model = ARIMA(values, exog=ts, order=self.order)
        self.results = self._model.fit()
        self.threshold = self.results.params[np.array(self.results.param_names) == "sigma2"][0]
        self.t = ts[-1]

    def predict(self, X):
        values = self.get_values(X)
        ts = np.arange(len(values)) + self.t
        values_pred = self.results.predict(ts[0], ts[-1])
        self.t += len(values)
        error = np.abs(values - values_pred)
        return error > self.threshold

    def fit_predict(self, X):
        self.fit(X)
        return self.predict(X)
