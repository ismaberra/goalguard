#!/bin/bash
#SBATCH --chdir /home/tdeclety
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 16
#SBATCH --mem 32G
#SBATCH --time 7:00:00
#SBATCH --qos serial # Change this according to your needs

echo "Starting job at $(date)"

cd ST_Trans
source trans/bin/activate

python preprocess.py
python training.py

deactivate

echo "Job finished at $(date)"
