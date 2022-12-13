import matplotlib.pyplot as plt
import numpy as np


def plot_outliers(series_data, y_pred_outliers, plot_windows=False, title=""):
    half_window_length = (len(series_data) - len(y_pred_outliers)) // 2
    plt.plot(series_data, color="xkcd:lightish blue", zorder=1)
    x_values_outliers = np.arange(len(y_pred_outliers)) + half_window_length
    ylim = plt.gca().get_ylim()
    window_locations = np.zeros(len(y_pred_outliers), bool)

    if plot_windows:
        for i in range(len(y_pred_outliers)):
            if y_pred_outliers[i]:
                left = max(i - half_window_length, 0)
                right = min(i + half_window_length, len(y_pred_outliers))
                window_locations[left:right] = True
        window_locations[y_pred_outliers] = False
        plt.fill_between(
            x_values_outliers,
            ylim[0],
            ylim[1],
            where=window_locations,
            color="xkcd:pale yellow",
            alpha=0.5,
            zorder=3,
            linewidth=0,
        )

    switch_next = False
    for i in range(1, len(y_pred_outliers) - 1):
        if y_pred_outliers[i]:
            y_pred_outliers[i - 1] = True
            switch_next = True
        elif switch_next:
            y_pred_outliers[i] = True
            switch_next = False

    plt.fill_between(
        x_values_outliers,
        ylim[0],
        ylim[1],
        where=y_pred_outliers,
        color="xkcd:light red",
        alpha=0.5,
        zorder=2,
        linewidth=0,
    )
    plt.gca().set_ylim(ylim)
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.title(title)
    plt.show()


# def accuracy_plot(
#     target,
#     mean_prediction,
#     one_step_prediction,
#     Nstep_prediction,
#     sorted_errors_mean,
#     score,
#     predicted_score,
#     test_dataset,
#     title,
#     compensate=True,
# ):
#     fig, ax1 = plt.subplots()
#     # , markersize=1, linewidth=0.5)
#     ax1.plot(target, label="Target", marker=".", linestyle="--")
#     ax1.plot(
#         mean_prediction, label="Mean predictions", marker=".", linestyle="--"
#     )  # , markersize=1, linewidth=0.5)
#     # ax1.plot(oneStep_prediction, label='1-step predictions',
#     #          marker='.', linestyle='--') # , markersize=1, linewidth=0.5)
#     # ax1.plot(Nstep_prediction, label=str(args.prediction_window_size) + '-step predictions',
#     #          marker='.', linestyle='--') # , markersize=1, linewidth=0.5)
#     ax1.plot(
#         sorted_errors_mean,
#         label="Absolute mean prediction errors",
#         marker=".",
#         linestyle="--",
#     )  # , markersize=1, linewidth=1.0)
#     ax1.legend(loc="upper left")
#     ax1.set_ylabel("Value")
#     ax1.set_xlabel("Index")
#     ax2 = ax1.twinx()
#     ax2.plot(
#         score.numpy().reshape(-1, 1),
#         label="Anomaly scores from \nmultivariate normal distribution",
#         marker="o",
#         linestyle="--",
#         color="C3",
#     )  # , markersize=1, linewidth=1)
#     ax2.legend(loc="upper right")
#     ax2.set_ylabel("anomaly score")  # ,fontsize=15)
#     # fontsize=18, fontweight='bold')
#     plt.title("Anomaly Detection on " + title + " Dataset")
#     plt.tight_layout()
#     plt.xlim([0, len(test_dataset)])


# TODO preprocess_data missing
#
# from mb_detect.models.deep.rnn_model.anomalyDetector import anomalyScore, get_precision_recall
# from mb_detect.models.deep.rnn_model import RNNPredictor
# def detect_outliers_rnn(data_type, filename, augment_test_data=False):
#     checkpoint = torch.load(str(Path('save', data_type, 'checkpoint', filename).with_suffix('.pth')))
#     args = checkpoint['args']

#     # Set the random seed manually for reproducibility.
#     torch.manual_seed(args.seed)
#     torch.cuda.manual_seed(args.seed)

#     # Load data
#     TimeseriesData = preprocess_data.PickleDataLoad(data_type=args.data,
#                                                     filename=args.filename, augment_test_data=False)
#     train_dataset = TimeseriesData.batchify(args,TimeseriesData.trainData[:TimeseriesData.length], bsz=1)
#     test_dataset = TimeseriesData.batchify(args,TimeseriesData.testData, bsz=1)

#     # Build the model
#     nfeatures = TimeseriesData.trainData.size(-1)
#     model = RNNPredictor(rnn_type = args.model,
#                             enc_inp_size=nfeatures,
#                             rnn_inp_size = args.emsize,
#                             rnn_hid_size = args.nhid,
#                             dec_out_size=nfeatures,
#                             nlayers = args.nlayers,
#                             res_connection=args.res_connection).to(args.device)
#     model.load_state_dict(checkpoint['state_dict'])
#     #del checkpoint

#     scores, predicted_scores, precisions, recalls, f_betas = list(), list(), list(), list(), list()
#     targets, mean_predictions, oneStep_predictions, Nstep_predictions = list(), list(), list(), list()
#     try:
#         # For each channel in the dataset
#         for channel_idx in range(nfeatures):
#             # 1. Load mean and covariance if they are pre-calculated, if not calculate them.
#             # Mean and covariance are calculated on train dataset.
#             if 'means' in checkpoint.keys() and 'covs' in checkpoint.keys():
#                 print('=> loading pre-calculated mean and covariance')
#                 mean, cov = checkpoint['means'][channel_idx], checkpoint['covs'][channel_idx]
#             else:
#                 print('=> calculating mean and covariance')
#                 mean, cov = fit_norm_distribution_param(args, model, train_dataset, channel_idx=channel_idx)

#             # 2. Train anomaly score predictor using support vector regression (SVR). (Optional)
#             # An anomaly score predictor is trained
#             # given hidden layer output and the corresponding anomaly score on train dataset.
#             # Predicted anomaly scores on test dataset can be used for the baseline of the adaptive threshold.
#             print('=> training an SVR as anomaly score predictor')
#             train_score, _, _, hiddens, _ = anomalyScore(args, model, train_dataset, mean, cov, channel_idx=channel_idx)
#             score_predictor = GridSearchCV(SVR(), cv=5,param_grid={"C": [1e0, 1e1, 1e2],"gamma": np.logspace(-1, 1, 3)})
#             score_predictor.fit(torch.cat(hiddens,dim=0).numpy(), train_score.cpu().numpy())

#             # 3. Calculate anomaly scores
#             # Anomaly scores are calculated on the test dataset
#             # given the mean and the covariance calculated on the train dataset
#             print('=> calculating anomaly scores')
#             score, sorted_prediction, sorted_error, _, predicted_score = anomalyScore(
#                 args, model, test_dataset, mean, cov,
#                 score_predictor=score_predictor,
#                 channel_idx=channel_idx)

#             # 4. Evaluate the result
#             # The obtained anomaly scores are evaluated by measuring precision, recall, and f_beta scores
#             # The precision, recall, f_beta scores are are calculated repeatedly,
#             # sampling the threshold from 1 to the maximum anomaly score value, either equidistantly or logarithmically.
#             print('=> calculating precision, recall, and f_beta')
#             precision, recall, f_beta = get_precision_recall(args, score, num_samples=1000, beta=1.0,
#                                                             label=TimeseriesData.testLabel.to(args.device))
#             print('data: ',args.data,' filename: ',args.filename,
#                 ' f-beta (no compensation): ', f_beta.max().item(),' beta: ', 1.0)
#             precision, recall, f_beta = get_precision_recall(args, score, num_samples=1000, beta=1.0,
#                                                             label=TimeseriesData.testLabel.to(args.device),
#                                                             predicted_score=predicted_score)
#             print('data: ',args.data,' filename: ',args.filename,
#                 ' f-beta    (compensation): ', f_beta.max().item(),' beta: ', 1.0)

#             target = preprocess_data.reconstruct(test_dataset.cpu()[:, 0, channel_idx],
#                                                 TimeseriesData.mean[channel_idx],
#                                                 TimeseriesData.std[channel_idx]).numpy()
#             mean_prediction = preprocess_data.reconstruct(sorted_prediction.mean(dim=1).cpu(),
#                                                         TimeseriesData.mean[channel_idx],
#                                                         TimeseriesData.std[channel_idx]).numpy()
#             oneStep_prediction = preprocess_data.reconstruct(sorted_prediction[:, -1].cpu(),
#                                                             TimeseriesData.mean[channel_idx],
#                                                             TimeseriesData.std[channel_idx]).numpy()
#             Nstep_prediction = preprocess_data.reconstruct(sorted_prediction[:, 0].cpu(),
#                                                         TimeseriesData.mean[channel_idx],
#                                                         TimeseriesData.std[channel_idx]).numpy()
#             sorted_errors_mean = sorted_error.abs().mean(dim=1).cpu()
#             sorted_errors_mean *= TimeseriesData.std[channel_idx]
#             sorted_errors_mean = sorted_errors_mean.numpy()
#             score = score.cpu()

#             accuracy_plot(target, mean_prediction,
#                             oneStep_prediction, Nstep_prediction,
#                             sorted_errors_mean, score,
#                             predicted_score, test_dataset, str(channel_idx))
#             plt.show()
#             plt.close()

#     except KeyboardInterrupt:
#         print('Exiting from training early')

#     return target, mean_prediction, sorted_errors_mean, score.numpy().reshape(-1, 1)
