# #  TODO module broken preprocess data is missing

# import argparse

# import matplotlib.pyplot as plt
# import numpy as np

# from mb_detect.models.classic.sklearn_dect import models, window
# from mb_detect.models.deep.rnn_model import preprocess_data

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='Sklearn Anomaly Detection')
#     parser.add_argument('--prediction_window_size', type=int, default=10,
#                         help='prediction_window_size')
#     parser.add_argument('--data', type=str, default='ecg',
#                         help='type of the dataset (ecg, gesture, power_demand, space_shuttle, respiration, nyc_taxi')
#     parser.add_argument('--filename', type=str, default='chfdb_chf13_45590.pkl',
#                         help='filename of the dataset')
#     parser.add_argument('--save_fig', action='store_true',
#                         help='save results as figures')
#     parser.add_argument('--compensate', action='store_true',
#                         help='compensate anomaly score using anomaly score estimation')
#     parser.add_argument('--beta', type=float, default=1.0,
#                         help='beta value for f-beta score')
#     parser.add_argument('--detector', type=str, default='forest',
#                         help='Choose a detector. Options forest, svm.')
#     parser.add_argument('--window_size', type=int, default=100,
#                         help='The size of the classification window.')
#     args = parser.parse_args()

#     # Load data
#     timeseries_data = preprocess_data.PickleDataLoad(data_type=args.data,
#                                                      filename=args.filename,
#                                                      augment_test_data=False)
#     timeseries_data.numpy()
#     train_data = timeseries_data.trainData
#     series_train_data = np.reshape(train_data, -1)
#     series_test_data = timeseries_data.testData[:, 0]
#     window_train_data = window(series_train_data, args.window_size)
#     window_test_data = window(series_test_data, args.window_size)

#     model = models.get_model(model_type=args.detector,
#                       window_size=args.window_size)

#     model.fit(window_train_data)
#     y_pred_outliers = model.predict(window_test_data)
#     plt.plot(series_test_data)
#     plt.plot(y_pred_outliers, '.')
#     plt.show()
