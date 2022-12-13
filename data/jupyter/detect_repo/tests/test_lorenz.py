from mb_detect.dataloader import ds_loader


def test_lorenz():
    data = ds_loader.lorenz_data(anomalies=True)
    print("generated lorenz data", data)
    print("generated blockified", data[data["is_anomaly"]])


if __name__ == "__main__":
    test_lorenz()
