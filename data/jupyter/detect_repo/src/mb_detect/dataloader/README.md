## Dataset reader / download

iter_reader.py supports reading and iterating through standard datasets for anomaly detection. As of now, it supports the Yahoo and Numenta datasets. Get these datasets at https://webscope.sandbox.yahoo.com/catalog.php?datatype=s&did=70 and https://github.com/numenta/NAB/archive/master.zip respectively. Yahoo requires to first actively request the dataset before enabling the download. 

For downloading and unpacking the nab dataset, you can also run nab_downloader.py. From the detect folder run: python ./src/data_sets/nab_downloader.py