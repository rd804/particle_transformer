#!/bin/bash
#SBATCH --job-name=ddp-torch     # create a short name for your job
#SBATCH --requeue
#SBATCH --nodes=2                # node count
#SBATCH --ntasks-per-node=2      # total number of tasks per node
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=30G                # total memory per node (4 GB per cpu-core is default)
#SBATCH --gres=gpu:2             # number of allocated gpus per node
#SBATCH --time=03:00:00          # total run time limit (HH:MM:SS)
#SBATCH --nodelist=gpu[020-027]
#SBATCH --partition=gpu
#SBATCH --output=slurm/top_tagging.out        


export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "WORLD_SIZE="$WORLD_SIZE
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

module purge
module load singularity/3.1.0

nvidia-smi

extra_opt="--optimizer-option lr_mult (\"fc.*\",50)"

srun singularity exec --nv ~/weaver.sif python -u top_tagging.py \
    --batch_size=128 --epochs=20 \
    --optimizer-option weight_decay 0.01 \
    --start_lr 1e-4 --optimizer ranger --use_amp \
    ${extra_opt}