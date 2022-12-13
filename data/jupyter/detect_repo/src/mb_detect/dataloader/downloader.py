import os
import pickle
import shutil
import zipfile
from pathlib import Path

import requests

import wget


def nab_download(
    nab_path="./data/nab/", nab_url="https://github.com/numenta/NAB/archive/master.zip"
):
    """Download the Numenta Dataset.

    A description of the data is available at:
    <https://www.sciencedirect.com/science/article/pii/S0925231217309864>
    """
    if os.path.exists(nab_path):
        print("nab data folder exists already .")
        return
    os.makedirs(nab_path)
    print("Downloading nab data set.")
    file_name = wget.download(nab_url)
    os.replace(file_name, nab_path + file_name)
    with zipfile.ZipFile(nab_path + file_name, "r") as zipf:
        zipf.extractall(nab_path)

    # extract the nab specific processed zip file.
    shutil.move(nab_path + "NAB-master/data", nab_path + "data")
    shutil.move(nab_path + "NAB-master/labels", nab_path + "labels")
    shutil.rmtree(nab_path + "NAB-master/")
    os.remove(nab_path + "NAB-master.zip")
    print("download successful")


def disc_pkl(_dir, _path, _data):
    with open(str(_dir.joinpath(_path.name).with_suffix(".pkl")), "wb") as pkl:
        pickle.dump(_data, pkl)


def discord_download(target_dir="./data/discord/"):
    # based on
    # https://github.com/chickenbestlover/RNN-Time-series-Anomaly-Detection
    urls = dict()
    urls["ecg"] = [
        "http://www.cs.ucr.edu/~eamonn/discords/ECG_data.zip",
        "http://www.cs.ucr.edu/~eamonn/discords/mitdbx_mitdbx_108.txt",
        "http://www.cs.ucr.edu/~eamonn/discords/qtdbsele0606.txt",
        "http://www.cs.ucr.edu/~eamonn/discords/chfdbchf15.txt",
        "http://www.cs.ucr.edu/~eamonn/discords/qtdbsel102.txt",
    ]
    urls["gesture"] = ["http://www.cs.ucr.edu/~eamonn/discords/ann_gun_CentroidA"]
    urls["space_shuttle"] = [
        "http://www.cs.ucr.edu/~eamonn/discords/TEK16.txt",
        "http://www.cs.ucr.edu/~eamonn/discords/TEK17.txt",
        "http://www.cs.ucr.edu/~eamonn/discords/TEK14.txt",
    ]
    urls["respiration"] = [
        "http://www.cs.ucr.edu/~eamonn/discords/nprs44.txt",
        "http://www.cs.ucr.edu/~eamonn/discords/nprs43.txt",
    ]
    urls["power_demand"] = ["http://www.cs.ucr.edu/~eamonn/discords/power_data.txt"]

    for dataname in urls:
        raw_dir = Path(target_dir, dataname, "raw")
        raw_dir.mkdir(parents=True, exist_ok=True)
        for url in urls[dataname]:
            filename = raw_dir.joinpath(Path(url).name)
            print("Downloading", url)
            resp = requests.get(url)
            filename.write_bytes(resp.content)
            if filename.suffix == "":
                filename.rename(filename.with_suffix(".txt"))
            print("Saving to", filename.with_suffix(".txt"))
            if filename.suffix == ".zip":
                print("Extracting to", filename)
                shutil.unpack_archive(str(filename), extract_dir=str(raw_dir))

        for filepath in raw_dir.glob("*.txt"):
            with open(str(filepath)) as f:
                # Label anomaly points as 1 in the dataset
                labeled_data = []
                for i, line in enumerate(f):
                    tokens = [float(token) for token in line.split()]
                    if raw_dir.parent.name == "ecg":
                        # Remove time-step channel
                        tokens.pop(0)
                    if filepath.name == "chfdbchf15.txt":
                        tokens.append(1.0) if 2250 < i < 2400 else tokens.append(0.0)
                    elif filepath.name == "xmitdb_x108_0.txt":
                        tokens.append(1.0) if 4020 < i < 4400 else tokens.append(0.0)
                    elif filepath.name == "mitdb__100_180.txt":
                        tokens.append(1.0) if 1800 < i < 1990 else tokens.append(0.0)
                    elif filepath.name == "chfdb_chf01_275.txt":
                        tokens.append(1.0) if 2330 < i < 2600 else tokens.append(0.0)
                    elif filepath.name == "ltstdb_20221_43.txt":
                        tokens.append(1.0) if 650 < i < 780 else tokens.append(0.0)
                    elif filepath.name == "ltstdb_20321_240.txt":
                        tokens.append(1.0) if 710 < i < 850 else tokens.append(0.0)
                    elif filepath.name == "chfdb_chf13_45590.txt":
                        tokens.append(1.0) if 2800 < i < 2960 else tokens.append(0.0)
                    elif filepath.name == "stdb_308_0.txt":
                        tokens.append(1.0) if 2290 < i < 2550 else tokens.append(0.0)
                    elif filepath.name == "qtdbsel102.txt":
                        tokens.append(1.0) if 4230 < i < 4430 else tokens.append(0.0)
                    elif filepath.name == "ann_gun_CentroidA.txt":
                        tokens.append(1.0) if 2070 < i < 2810 else tokens.append(0.0)
                    elif filepath.name == "TEK16.txt":
                        tokens.append(1.0) if 4270 < i < 4370 else tokens.append(0.0)
                    elif filepath.name == "TEK17.txt":
                        tokens.append(1.0) if 2100 < i < 2145 else tokens.append(0.0)
                    elif filepath.name == "TEK14.txt":
                        tokens.append(
                            1.0
                        ) if 1100 < i < 1200 or 1455 < i < 1955 else tokens.append(0.0)
                    elif filepath.name == "nprs44.txt":
                        tokens.append(
                            1.0
                        ) if 16192 < i < 16638 or 20457 < i < 20911 else tokens.append(
                            0.0
                        )
                    elif filepath.name == "nprs43.txt":
                        tokens.append(
                            1.0
                        ) if 12929 < i < 13432 or 14877 < i < 15086 or 15729 < i < 15924 else tokens.append(
                            0.0
                        )
                    elif filepath.name == "power_data.txt":
                        tokens.append(
                            1.0
                        ) if 8254 < i < 8998 or 11348 < i < 12143 or 33883 < i < 34601 else tokens.append(
                            0.0
                        )
                    labeled_data.append(tokens)

                # Fill in the point where there is no signal value.
                if filepath.name == "ann_gun_CentroidA.txt":
                    for i, datapoint in enumerate(labeled_data):
                        for j, channel in enumerate(datapoint[:-1]):
                            if channel == 0.0:
                                labeled_data[i][j] = (
                                    0.5 * labeled_data[i - 1][j]
                                    + 0.5 * labeled_data[i + 1][j]
                                )

                # Save the labeled dataset as .pkl extension
                labeled_whole_dir = raw_dir.parent.joinpath("labeled", "whole")
                labeled_whole_dir.mkdir(parents=True, exist_ok=True)
                with open(
                    str(labeled_whole_dir.joinpath(filepath.name).with_suffix(".pkl")),
                    "wb",
                ) as pkl:
                    pickle.dump(labeled_data, pkl)

                # Divide the labeled dataset into trainset and testset, then save them
                labeled_train_dir = raw_dir.parent.joinpath("labeled", "train")
                labeled_train_dir.mkdir(parents=True, exist_ok=True)
                labeled_test_dir = raw_dir.parent.joinpath("labeled", "test")
                labeled_test_dir.mkdir(parents=True, exist_ok=True)
                if filepath.name == "chfdb_chf13_45590.txt":
                    disc_pkl(labeled_train_dir, filepath, labeled_data[:2439])
                    disc_pkl(labeled_test_dir, filepath, labeled_data[2439:3726])
                elif filepath.name == "chfdb_chf01_275.txt":
                    disc_pkl(labeled_train_dir, filepath, labeled_data[:1833])
                    disc_pkl(labeled_test_dir, filepath, labeled_data[1833:3674])
                elif filepath.name == "chfdbchf15.txt":
                    disc_pkl(labeled_train_dir, filepath, labeled_data[3381:14244])
                    disc_pkl(labeled_test_dir, filepath, labeled_data[33:3381])
                elif filepath.name == "qtdbsel102.txt":
                    disc_pkl(labeled_train_dir, filepath, labeled_data[10093:44828])
                    disc_pkl(labeled_test_dir, filepath, labeled_data[211:10093])
                elif filepath.name == "mitdb__100_180.txt":
                    disc_pkl(labeled_train_dir, filepath, labeled_data[2328:5271])
                    disc_pkl(labeled_test_dir, filepath, labeled_data[73:2328])
                elif filepath.name == "stdb_308_0.txt":
                    disc_pkl(labeled_train_dir, filepath, labeled_data[2986:5359])
                    disc_pkl(labeled_test_dir, filepath, labeled_data[265:2986])
                elif filepath.name == "ltstdb_20321_240.txt":
                    disc_pkl(labeled_train_dir, filepath, labeled_data[1520:3531])
                    disc_pkl(labeled_test_dir, filepath, labeled_data[73:1520])
                elif filepath.name == "xmitdb_x108_0.txt":
                    disc_pkl(labeled_train_dir, filepath, labeled_data[424:3576])
                    disc_pkl(labeled_test_dir, filepath, labeled_data[3576:5332])
                elif filepath.name == "ltstdb_20221_43.txt":
                    disc_pkl(labeled_train_dir, filepath, labeled_data[1121:3731])
                    disc_pkl(labeled_test_dir, filepath, labeled_data[0:1121])
                elif filepath.name == "ann_gun_CentroidA.txt":
                    disc_pkl(labeled_train_dir, filepath, labeled_data[3000:])
                    disc_pkl(labeled_test_dir, filepath, labeled_data[:3000])
                elif filepath.name == "nprs44.txt":
                    disc_pkl(labeled_train_dir, filepath, labeled_data[363:12955])
                    disc_pkl(labeled_test_dir, filepath, labeled_data[12955:24082])
                elif filepath.name == "nprs43.txt":
                    disc_pkl(labeled_train_dir, filepath, labeled_data[4285:10498])
                    disc_pkl(labeled_test_dir, filepath, labeled_data[10498:17909])
                elif filepath.name == "power_data.txt":
                    disc_pkl(labeled_train_dir, filepath, labeled_data[15287:33432])
                    disc_pkl(labeled_test_dir, filepath, labeled_data[501:15287])
                elif filepath.name == "TEK17.txt":
                    disc_pkl(labeled_train_dir, filepath, labeled_data[2469:4588])
                    disc_pkl(labeled_test_dir, filepath, labeled_data[1543:2469])
                elif filepath.name == "TEK16.txt":
                    disc_pkl(labeled_train_dir, filepath, labeled_data[521:3588])
                    disc_pkl(labeled_test_dir, filepath, labeled_data[3588:4539])
                elif filepath.name == "TEK14.txt":
                    disc_pkl(labeled_train_dir, filepath, labeled_data[2089:4098])
                    disc_pkl(labeled_test_dir, filepath, labeled_data[97:2089])
