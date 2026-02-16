#!/bin/bash
#
# SLURM script to kick off many jobs on the svante cluster. Each job 
# will process one layer
#
# Name of submitted job:
#SBATCH -J NoLCPS
# ** THE FOLDER HAS TO BE CREATED BEFOREHAND ** Otherwise unexplained JobLaunchFailure occurs.
#SBATCH -o NS_sounder_NoLCPS_HyperMatrix_Layer_Dynamic_numL900/logs/task-%a.sh.log # outdir directory
#SBATCH --array 0-30 # Number of total tasks
#SBATCH --cpus-per-task 5 # number of workers. 
#SBATCH -p pi_seager # partition; edr doesn't allow 10-day time limit
#SBATCH -t 5-00:00:00

echo "hello"
echo 'Job running on node(s): '$SLURM_JOB_NODELIST
echo 'CPUs(x cores) per task: '$SLURM_TASKS_PER_NODE
echo 'Task id: '$SLURM_ARRAY_TASK_ID
echo 'Num tasks '$SLURM_ARRAY_TASK_COUNT
echo 'Path:' 
pwd

source /etc/profile # supercloud: source /etc/profile
# eval "$(conda shell.bash hook)"
# conda deactivate
module load anaconda3/2023.07 # supercloud: anaconda/2023a-pytorch; eofe: anaconda3/2020.11
conda activate pioneer # supercloud: conda activate pioneer

# Use these lines if wanting to test run a single job in a specific task folder
#   export SLURM_ARRAY_TASK_ID=6
#   export SLURM_ARRAY_TASK_COUNT=6

# Run the script
python NS_NoLCPS_Layer_Matrix_Retrieval.py \
--task_id $SLURM_ARRAY_TASK_ID \
--num_tasks $SLURM_ARRAY_TASK_COUNT
