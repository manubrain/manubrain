import argparse

from mb_detect.dataloader import iter_reader, ds_loader, preprocess, benchmarker
from mb_detect.models.classic.arima_anomaly import ArimaAnomaly

def parse_cmd():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--datapath", 
        default=None, 
        help="dataset path"
    )
    
    parser.add_argument(
        "--dataseperator", 
        default="\\s", 
        help="dataset seperator"
    )

    parser.add_argument(
        "--dataset",
        choices=["shuttle", "nab", "yahoo", "lorenz"],
        default="nab",
        help="Dataset to choose",
    )

    parser.add_argument(
        "--ds_basedir",
        #TODO
        default="/home/afeiden/Dokumente/git/detect/data/nab", 
        help="search this directory for the data",
    )

    parser.add_argument(
        "--preprocess_normalize",
        default=False
    )

    parser.add_argument(
        "--preprocess_roll",
        default=0
    )

    #TODO
    parser.add_argument(
        "--single_timeseries",
        default=None
    )

    #TODO
    parser.add_argument(
        "--model",
        choices=["arima"],
        default="arima",
        help="Architecture to choose",
    )

    parser.add_argument(
        "--model_order_p",
        default=1,
        type=int,
        help="the number of lag observations in the model also known as the lag order",
    )

    parser.add_argument(
        "--model_order_d",
        default=0,
        type=int,
        help="the number of times that the raw observations are differenced also known as the degree of differencing",
    )

    parser.add_argument(
        "--model_order_q",
        default=1,
        type=int,
        help="the size of the moving average window also known as the order of the moving average",
    )

    parser.add_argument(
        "--load_config",
        type=str,
        help="json-file to load the start up config from"
    )

    parser.add_argument(
        "--save_config",
        type=str,
        help="json-file to save the start up config to"
    )

    #TODO
    parser.add_argument(
        "--mpi_n",
        type=int,
        default=0,
        help="for parallel use with mpi"
    )
    return vars(parser.parse_args())

def parse_config():
    config = parse_cmd()
    if config["load_config"]:
        print("TODO load from config", config["load_config"])
        # config = load_config(config["load_config"])
    elif config["save_config"]:
        print("TODO save config to", config["save_config"])
    return config

def load_ds_iter(config):
    if config["dataset"] == "nab":
        print("load numenta dataset")
        data_dir = config["ds_basedir"] + "/data/"
        label_dir = config["ds_basedir"] + "/labels/"
        ds_iterator = iter_reader.NabIter(data_dir=data_dir, label_dir=label_dir)
    return ds_iterator

def preprocess_data(config, ds):
    p = preprocess.Preprocess()
    if config["preprocess_normalize"]:
        ds = p.normalize(ds)
    if config["preprocess_roll"] > 0:
        ds = p.roll(ds, config["preprocess_roll"])
    return ds

def load_model(config):
    if config["model"] == "arima":
        order = [config["model_order_p"], config["model_order_d"], config["model_order_q"]]
        return ArimaAnomaly(order=order)

def main():
    config = parse_config()
    ds_iterator = load_ds_iter(config)

    for ds, metadata in ds_iterator:
        ds = preprocess_data(config, ds)
        model = load_model(config)
        ds["anomaly_score"] = model.fit_predict(ds_loader.get_values(ds))
        print(metadata["name"], benchmarker.auc_score(ds["is_anomaly"], ds["anomaly_score"]))
        break
    

    # model = 

if __name__ == "__main__":
    main()