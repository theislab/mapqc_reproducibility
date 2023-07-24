#! /bin/bash

#SBATCH -o mapping_%j.out
#SBATCH -e mapping_%j.out
#SBATCH --partition=cpu_p
#SBATCH -J r_map
#SBATCH --mem=100G
#SBATCH --cpus-per-task=8
#SBATCH -t 08:00:00
#SBATCH --nice=10000
#SBATCH --qos=cpu_normal

export R_LIBS=$CONDA_PREFIX/lib/R/library/
Rscript /home/icb/jan.engelmann/atlas_extension/notebooks/hypomap/mapping_job.r $1