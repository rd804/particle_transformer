#!/bin/bash
#SBATCH --job-name=ddp-torch     # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks-per-node=1      # total number of tasks per node
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=5G                # total memory per node (4 GB per cpu-core is default)
####SBATCH --gres=gpu:1             # number of allocated gpus per node
#SBATCH --time=00:10:00          # total run time limit (HH:MM:SS)
########SBATCH --nodelist=gpu[015-026]
#SBATCH --partition=main
#SBATCH --output=slurm/top_tagging_convert.log        


module purge
module load singularity/3.1.0

singularity exec ~/weaver.sif python scripts/convert_dataset.py --source='data/test.h5' --destination='data/converted/tops' --type='test'