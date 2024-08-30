#!/bin/bash
#SBATCH -A MLMI-wa285-SL2-GPU
#SBATCH -J TRAIN_CHAT_3D_V2
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --time=30:00:00
#SBATCH --mail-type=ALL
#! Uncomment this to prevent the job from being requeued (e.g. if
#! interrupted by node failure or system downtime):
##SBATCH --no-requeue
#SBATCH -p ampere
#! ############################################################

. /etc/profile.d/modules.sh                # Leave this line (enables the module command)
module purge                               # Removes all modules still loaded
module load rhel8/default-amp
module load cuda/11.8

module load anaconda/3.2019-10
eval "$(conda shell.bash hook)"

conda activate py3d

cd /home/wa285/rds/hpc-work/Thesis/Data-Generation

python data_generation.py --segmentor=mask3d --version=""
