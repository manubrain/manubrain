import os
import shutil
import zipfile

import wget

def download_nab(nab_path = "./data/nab/", nab_url = "https://github.com/numenta/NAB/archive/master.zip"):
    if os.path.exists(nab_path):
        print("nab data folder exists already .")
    else:
        os.makedirs(nab_path)
        print("Downloading nab data set.")
        file_name = wget.download(nab_url)
        os.replace(file_name, nab_path + file_name)
        with zipfile.ZipFile(nab_path + file_name, "r") as zipf:
            zipf.extractall(nab_path)

        # extract the nab specific processed zip file.
        shutil.move("./data/nab/NAB-master/data", "./data/nab/data")
        shutil.move("./data/nab/NAB-master/labels", "./data/nab/labels")
        shutil.rmtree("./data/nab/NAB-master/")
        os.remove("./data/nab/NAB-master.zip")
        print("download successful")