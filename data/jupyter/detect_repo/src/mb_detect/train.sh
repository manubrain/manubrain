#!/bin/bash
#SBATCH --job-name=Shuttle_DeepAnt
#SBATCH --output=deepant_output.txt
#SBATCH --error=deepant_error.txt
#SBATCH -p gpu

module load Anaconda3
source /home/lveeramacheneni/.bashrc
conda activate /home/lveeramacheneni/lconda_env

/home/lveeramacheneni/lconda_env/bin/python /home/lveeramacheneni/ManuBrain/detect/src/mb_detect/cli.py --datasetname='nab' --model='rnn' --historywindow=100 --predictwindow=70 --batch-size=1 --epochs=150 --trainpath='./dataloader/data/nab/data/artificialNoAnomaly/art_daily_perfect_square_wave.csv' --testpath='./dataloader/data/nab/data/artificialWithAnomaly/art_daily_jumpsdown.csv'
