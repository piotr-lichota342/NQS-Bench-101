#!/bin/bash
#SBATCH --job-name=test_python_simple
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-user="p.w.lichota@umail.leidenuniv.nl"
#SBATCH --mail-type="ALL"
#SBATCH --mem=6G
#SBATCH --time=7-00:00:00
#SBATCH --partition=gpu-l4-24g
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

# load modules (assuming you start from the default environment)
# we explicitly call the modules to improve reproducibility
# in case the default settings change
#module load Python/3.11.5-GCCcore-13.2.0

# Source the Python virtual environment
cd /home/s4072006/Desktop/nqs-bench-101/NQS-Bench-101/
source ~/miniconda3/etc/profile.d/conda.sh
conda activate nqs


echo "[$SHELL] #### Starting Python test"
echo "[$SHELL] ## This is $SLURM_JOB_USER on $HOSTNAME and this job has the ID $SLURM_JOB_ID"
# get the current working directory
export CWD=$(pwd)
echo "[$SHELL] ## current working directory: "$CWD

# Run the file
echo "[$SHELL] ## Run script"
python3 code/all_architectures_control.py
echo "[$SHELL] ## Script finished"

echo "[$SHELL] #### Finished Python test. Have a nice day"
