#!/bin/bash

#SBATCH -A kdlin_cs7322_1247_000_0001

# Name for the job that will be visible in the job queue and accounting tools.
#SBATCH --job-name NewJob

# Name of the SLURM partition that this job should run on.
#SBATCH -p gpu-dev       # partition (queue)
# Number of nodes required to run this job
#SBATCH -N 1

#SBATCH --mem 380928

# Time limit for the job in the format Days-H:M:S
# A job that reaches its time limit will be cancelled.
# Specify an accurate time limit for efficient scheduling so your job runs promptly.
#SBATCH -t 0-1:0:0

# The standard output and errors from commands will be written to these files.
# %j in the filename will be replace with the job number when it is submitted.
#SBATCH -o job_%j.out
#SBATCH -e job_%j.err

# Send an email when the job status changes, to the specfied address.
#SBATCH --mail-type ALL
#SBATCH --mail-user ataychameekiatchai@smu.edu


module load conda 
conda activate py39

# Log common failure points.
pwd
python --version 

cd /lustre/work/client/users/ataychameekiatchai/CS-7322-NLP-Final-Project
python scripts/classifer-model-training.py

# END OF SCRIPT
