#!/usr/bin/env bash

#SBATCH -n 1
#SBATCH -t 04:00:00
#SBATCH --mem-per-cpu=2500M

#SBATCH
srun python all_to_all.py to_db