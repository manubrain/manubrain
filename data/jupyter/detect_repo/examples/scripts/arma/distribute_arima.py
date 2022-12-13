from mpi4py import MPI

import numpy as np

# Numenta DS has 58 time series
from mb_detect.dataloader.iter_reader import NabIter
from mb_detect.models.classic.arima_anomaly import ArimaAnomaly
from sklearn.metrics import roc_auc_score

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

order = (1, 0, 0)

model = ArimaAnomaly(order=order)

if rank == 0:
    print("rank , ds_name, roc_auc_score", order)

j = 0
for ds, metadata in NabIter():
    if j == rank:
        model.fit(ds["value"].to_numpy())
        score = roc_auc_score(ds["is_anomaly"].to_numpy(), model.anomaly_score)
        print(rank, ",", metadata["name"], ",", score)
    j += 1
