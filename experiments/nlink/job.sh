#!/bin/sh
#$-cwd
#$-j y
#$-l rt_F=1
#$-l h_rt=72:00:00

source /etc/profile.d/modules.sh
#module load cuda/9.0/9.0.176.4 
#module load cudnn/7.1/7.1.4 

source ~/.bashrc
source ~/kojima/new_env.sh
conda activate ddm2
conda info -e
pwd
nvidia-smi
cd ~/kojima/DeepDissipativModel/experiments/nlink/
#sh run_all.sh 1
parallel-gpu ./run_all.1.list

ls

