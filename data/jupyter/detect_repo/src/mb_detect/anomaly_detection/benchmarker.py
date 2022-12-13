import numpy as np
from sklearn.metrics import confusion_matrix

from ..anomaly_detection.detect_helper import plot_outliers
from ..models.classic.sklearn_dect import window


def test_unsupervised(data, model, window_size=0, plot=False):
    """Predicts the full dataset against an unsupervised model.

    The labels are ignored for the predicted, hence unsupervised.
    Args:
        data (pandas.dataframe): a timeseries with equidistant timesteps, anomaly detection against "value" column
    Returns:
        numpy.array: the predicted outliers
    """
    values = data["value"].to_numpy()
    if window_size:
        pred_outliers = model.fit_predict(window(values, window_size=window_size))
    else:
        pred_outliers = model.fit_predict(values.reshape(-1, 1))
    if plot:
        plot_outliers(values, pred_outliers, plot_windows=window_size, title="")
    return pred_outliers


# def test_supervised(data, model, test_split=0.2):
# TODO
# train_data, test_data = train_test_split(data, test_size=test_split, shuffle=False)
# pass


def compare_to_labels(outliers, pred_outliers):
    """Compare predicted outliers to actual outliers.

    If using a windowed approach this comparison is forgiving. The confusion matrix is defined by:
    True positives: If within the window of point predicted as positive, there is an anomaly,
     we call that a true positive.
    False positives: If within the window of point predicted as positive, there is no anomaly,
     we call that a false positive.
    False negative: If the point predicted as negative is labeled as positive,
     we call it a false negative.
    True negative: If the point predicted as negative is labeled as negative,
     we call it a true negative.
    Args:
        outliers (numpy.array): the labels
        pred_outliers (numpy.array): the predictions
    Returns:
        numpy.array: 2x2 confusion matrix
    """
    w = (len(outliers) - len(pred_outliers)) // 2
    if w == 0:
        # standard, non-forgiving way to predict outliers:
        return confusion_matrix(outliers, pred_outliers, labels=[True, False])

    outl_length = len(pred_outliers)
    cm = np.zeros((2, 2), dtype=int)

    for i in np.where(pred_outliers == 1)[0]:
        if np.sum(outliers[max(0, i - w) : min(i + w + 1, outl_length)]) > 0:
            cm[0, 0] += 1
        else:
            cm[0, 1] += 1

    subset_outliers = outliers[w:-w]
    cm[1, 0] = np.sum(subset_outliers[pred_outliers == 0])
    cm[1, 1] = len(subset_outliers[pred_outliers == 0]) - cm[1, 0]
    return cm


def f_one_score(cm):
    """Calculate the f1 score of a 2x2 confusion matrix.

    Args:
        cm (numpy.array): 2x2 confusion matrix
    Returns:
        f1-score
    """
    if len(cm) == 1:
        return 1
    if (cm[0, 0] + cm[1, 0] + cm[0, 1]) == 0:
        return 1
    else:
        return 2 * cm[0, 0] / (2 * cm[0, 0] + cm[1, 0] + cm[0, 1])


def test_unsup_against_full_ds(reader, model, anomaly_marker=-1):
    """Test every dataset in an iterator against the model that uses an unsupervised approach.

    Prints the name and the confusion matrix
    """
    for data, name in reader:
        pred_outliers = test_unsupervised(data, model, window_size=50)
        pred_outliers = pred_outliers == anomaly_marker
        true_outliers = data["is_anomaly"]
        true_outliers = true_outliers.to_numpy()
        cm = compare_to_labels(true_outliers, pred_outliers)
        print(name, cm.flatten(), f_one_score(cm))
