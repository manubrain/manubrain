class Preprocess:
    def __init__(self):
        self.means = []
        self.stds = []

    def normalize(self, df):
        for column in [c for c in df.columns if c.startswith("value")]:
            self.means.append(df[column].mean())
            self.stds.append(df[column].std())
            df[column] -= df[column].mean()
            df[column] /= df[column].std()
        return df

    def de_normalize(self, df):
        i = 0
        for column in [c for c in df.columns if c.startswith("value")]:
            df[column] += self.means[i]
            df[column] *= self.stds[i]
            i += 1
        return df

    def roll(self, df, size):
        for column in [c for c in df.columns if c.startswith("value")]:
            roll_value = df[column].rolling(size, center=True).mean()
            roll_value = roll_value.dropna()
            df = df.loc[roll_value.index]
            df[column] = roll_value
        return df

    def window(data: np.array, window_size: int, step_size: int = 1) -> np.array:
        """Convert an array into an array of windows.

        Window the data by repeating and staking window size numbers
            every step_size setps, to allow sklearn to process context.

        Args:
            data (np.array): The input time series.
            window_size (int): The size of the window.
            step_size (int): The distance between the windows.
                Defaults to 1.

        Returns:
            np.array: Array containing the repeating data windows.
        """
        # window the data
        assert len(data.shape) == 1, "only time series allowed."
        window_lst = []
        half_window = window_size // 2
        for i in range(half_window, len(data) - half_window, step_size):
            window_lst.append(data[(i - half_window) : (i + half_window)])
        window_array = np.stack(window_lst, axis=0)
        return window_array
