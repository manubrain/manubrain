#  TODO not a script

from hyperopt import Trials, fmin, hp, tpe
from mb_detect.anomaly_detection.detect_helper import detect_outliers


def hyperopt_outliers(
    data_type,
    filename,
    detector,
    augment_test_data=False,
    window_size=100,
    window_step_size=1,
):
    # Load the space and function to optimize for dbscan hyperopt
    if detector == "dbscan":
        space = {
            "eps": hp.randint("eps", 30),
            "min_samples": hp.randint("min_samples", 101),
        }

        def hyperparameter_tuning(params):
            params = {
                "data_type": data_type,
                "filename": filename,
                "detector": detector,
                "eps": params["eps"] + 1,
                "min_samples": params["min_samples"] + 1,
            }
            _, _, _, loss = detect_outliers(**params)
            return loss

        trials = Trials()
        best = fmin(
            fn=hyperparameter_tuning,
            space=space,
            algo=tpe.suggest,
            max_evals=20,
            trials=trials,
        )
        data, y_pred, y_truth, mse_loss = detect_outliers(
            data_type,
            filename,
            detector,
            eps=best["eps"],
            min_samples=best["min_samples"],
        )
        return data, y_pred, y_truth, mse_loss

    # Load the space and function to optimize for isolation hyperopt
    elif detector == "forest":
        space = {"n_estimators": hp.randint("n_estimators", 100)}

        def hyperparameter_tuning(params):
            params = {
                "data_type": data_type,
                "filename": filename,
                "detector": detector,
                "n_estimators": params["n_estimators"],
            }
            _, _, _, loss = detect_outliers(**params)
            return loss

        trials = Trials()
        best = fmin(
            fn=hyperparameter_tuning,
            space=space,
            algo=tpe.suggest,
            max_evals=20,
            trials=trials,
        )
        data, y_pred, y_truth, mse_loss = detect_outliers(
            data_type, filename, detector, n_estimators=best["n_estimators"]
        )
        return data, y_pred, y_truth, mse_loss
    # ToDo: Extend the hyperopt spaces for OneClassSVM
    else:
        data, y_pred = detect_outliers(data_type, filename, detector)
        return data, y_pred
