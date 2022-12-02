# ManuBrain
## Setup
1. Install [Docker Compose](https://docs.docker.com/compose/install/).
2. Clone the repository and navigate into its root directory.
3. Run: `docker compose up`

## Tools
- InfluxDB: [http://localhost:8000](http://localhost:8000)
- Jupyter: [http://localhost:8200](http://localhost:8200)
- NodeRed: [http://localhost:1880](http://localhost:1880)
- Grafana: [http://localhost:3000](http://localhost:3000)

## Grafana
- User:     `admin`
- Password: `admin`

## Influxdb
- Username: `root`
- Password: `verysecret`
- Token:    `hfO13-DJp8_gMihghVzheI-azZAOGm57UFgwHwa3zioWnsE_z_31_85nVxWr6t9RrA--LCGWDudIAW6ZAal2Cw==`

The token is also stored as an environment variable called `INFLUXDB_TOKEN` in the Jupyter Notebook. You can access it in a python script as follows:
```
import os
print(os.environ['INFLUXDB_TOKEN'])
```
## Anomaly Detection
The jupyter notebook under data/jupyter/anomaly_detection.ipynb can be used for anomaly detection on data from the InfluxDB. The notebook is applied on the sine dataset. In order to apply it to new data from InfluxDB, in the first cell of the notebook replace the necessary information of organisation, bucket, measurement and field names. The genereated anomaly score for the datastream can then be found under bucket-->anomalies-->tcn-->anomaly_score.

If desired, one may also update the time_window and the epochs, which are used for training.


## Jupyter Notebook
To establish a connection to the InfluxDB from python use the following stub:
```
from influxdb_client import InfluxDBClient
client = InfluxDBClient(url="http://influxdb:8083", token=os.environ['INFLUXDB_TOKEN'], org="manubrain")
```

An end-to-end script that establishes a connection, writes some data and reads it back from the InfluxDB is as follows:
```
# CONFIGURATION
import influxdb_client, os, time
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS

token = os.environ.get("INFLUXDB_TOKEN")
org = "manubrain"
url = "http://influxdb:8086" # we must use the port 8086 here as we are inside the container (8000 is only from external)

client = influxdb_client.InfluxDBClient(url=url, token=token, org=org)
```
```
# WRITE DATA
bucket="test"  # create buckets at http://localhost:8000/

write_api = client.write_api(write_options=SYNCHRONOUS)
   
for value in range(5):
  point = (
    Point("measurement1")
    .tag("tagname1", "tagvalue1")
    .field("field1", value)
  )
  write_api.write(bucket=bucket, org="manubrain", record=point)
  time.sleep(1) # separate points by 1 second
```
```
# READ DATA
query_api = client.query_api()

query = """from(bucket: "test")
 |> range(start: -10m)
 |> filter(fn: (r) => r._measurement == "measurement1")"""
tables = query_api.query(query, org="manubrain")

for table in tables:
  for record in table.records:
    print(record)
```
