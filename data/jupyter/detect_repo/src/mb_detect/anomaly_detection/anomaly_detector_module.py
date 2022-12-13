import numpy as np
import torch


class AnomalyDetection:
    def __init__(self) -> None:
        pass

    def euclidean_distance(self, predictions, targets, threshold):
        distances = np.linalg.norm(predictions - targets, axis=0)
        indices = distances > threshold
        return distances, indices

    def detect_anomalies(
        self, model, testloader, threshold, use_cuda=True, type="euclidean"
    ):

        with torch.no_grad():
            for _, (data, targets) in enumerate(testloader):
                if use_cuda:
                    data = data.cuda()
                    targets = targets.cuda()
                preds = model(data)
                preds = preds.reshape(targets.shape)
                batchpred_labels = []
                for i in range(0, preds.shape[0]):
                    if type == "euclidean":
                        _, pred_labels = self.euclidean_distance(
                            preds[i, :, :].cpu(),
                            targets[i, :, :].cpu(),
                            threshold,
                        )
                        batchpred_labels.append(pred_labels)
                batchpred_labels = np.asarray(batchpred_labels)

                # TO-DO: Need to add metrics here
