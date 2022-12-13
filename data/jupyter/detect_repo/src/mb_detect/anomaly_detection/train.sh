#!/bin/bash
#SBATCH --job-name=Lorenz_DeepAnt_TCN
#SBATCH --output=deepant_output.txt
#SBATCH --error=deepant_error.txt
#SBATCH -p gpu

module load Anaconda3
source /home/lveeramacheneni/.bashrc
conda activate /home/lveeramacheneni/lconda_env

/home/lveeramacheneni/lconda_env/bin/python /home/lveeramacheneni/ManuBrain/detect/src/detector.py --datasetname='shuttle' --model='tcn' --historywindow=1800 --predictwindow=200 --batch-size=10 --modelpath='./Model_tcn_lorenz_10_epochs_1.pt'
#/home/lveeramacheneni/lconda_env/bin/python /home/lveeramacheneni/ManuBrain/detect/src/train.py --datasetname='lorenz' --model='tcn' --historywindow=1800 --predictwindow=200 --batch-size=10 --epochs=1
