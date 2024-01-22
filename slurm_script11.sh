#!/bin/bash
#SBATCH --job-name=david11
#SBATCH --gres=gpu:1
#SBATCH -c 2
#SBATCH --mem=10G
#SBATCH --output=./slurm_outputs/slurm-%j.out
#SBATCH --error=./slurm_outputs/slurm-%j.err

echo Running on $(hostname)
date

module load pytorch2.0-cuda11.8-python3.9
module load cuda-11.8
source activate llm-pe

# (while true; do nvidia-smi; top -b -n 1 | head -20; sleep 10; done) &
# srun accelerate launch /h/davaus80/McIntoshLab-MedBind/david_efficiency_test.py
/h/davaus80/.conda/envs/llm-pe/bin/python3 /h/davaus80/llm-pe/experiment_manager.py -exp_dir=/h/davaus80/llm-pe/experiments/jan_21_recipes_100_users_noise0.5_greedy
# wait
