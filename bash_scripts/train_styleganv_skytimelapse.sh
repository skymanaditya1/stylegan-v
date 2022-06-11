#!/bin/bash 

#SBATCH --job-name=sky_stylv
#SBATCH --mem-per-cpu=2048
#SBATCH --partition long
#SBATCH --account cvit_bhaasha
#SBATCH --gres=gpu:4
#SBATCH --mincpus=38
#SBATCH --nodes=1
#SBATCH --time 10-00:00:00
#SBATCH --signal=B:HUP@600
#SBATCH -w gnode045


cd /ssd_scratch/cvit/aditya1/stylegan-v

source /home2/aditya1/miniconda3/bin/activate base
conda activate ./env

CUDA_VISIBLE_DEVICES=2,3 HYDRA_FULL_ERROR=1 python src/infra/launch.py hydra.run.dir=. exp_suffix=skytimelapse_augsenabled env=local dataset=sky_timelapse dataset.resolution=128 num_gpus=2