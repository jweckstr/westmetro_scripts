#!/usr/bin/env bash

#SBATCH -n 1
#SBATCH -t 04:00:00
#SBATCH --mem-per-cpu=2500M
#SBATCH --array=0-63

#SBATCH
srun python all_to_all.py run_routing $SLURM_ARRAY_TASK_ID 64