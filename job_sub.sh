#!/bin/bash
#SBATCH --nodes=1               # number of nodes
#SBATCH --gpus-per-node=4            # number of gpus per node
#SBATCH --cpus-per-task=8       # number of threads per task
#SBATCH --tasks-per-node=4 # This is the number of model replicas we will place on the GPU.
#SBATCH --mem=128G
#SBATCH --job-name="test-multi-gpu"
#SBATCH --time=00:05:00        # Specify run time 
#SBATCH --output=%N-%j.out    # Specify output file format generated by python script
#SBATCH --error=%N-%j.error
#SBATCH --mail-user=yuwei.cao@ubc.ca    # Request email notifications
#SBATCH --mail-type=ALL

# code transfer
cd $SLURM_TMPDIR
mkdir work
cd work
git clone git@github.com:yuwei-cao-git/trspes_dl.git
cd trspes_dl
echo "codes finished cloned"

# data transfer
mkdir -p data
# extract an archive to a different directory, the ‘-C’ option is followed by the destination path
tar -xf $project/data/rmf_laz.tar -C ./data
ls $SLURM_TMPDIR/work/trspes_dl
echo "data transfered"

# Load python module, and additional required modules
module purge 
module load python/3.10 scipy-stack
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r requirements.txt
pip install laspy[laszip]

# Set environment variables
export NCCL_BLOCKING_WAIT=1  #Set this environment variable if you wish to use the NCCL backend for inter-GPU communication.
export MASTER_ADDR=$(hostname) #Store the master node’s IP address in the MASTER_ADDR environment variable.

#Print (echo) info to output file
echo "r$SLURM_NODEID master: $MASTER_ADDR"
echo "r$SLURM_NODEID Launching test dl cc script"
# 3mins so far

# Log experiment variables
wandb offline

#Run python script
# The $((SLURM_NTASKS_PER_NODE * SLURM_JOB_NUM_NODES)) variable tells the script how many processes are available for this execution. “srun” executes the script <tasks-per-node * nodes> times
cd Pytorch/models/PointAugment
srun python main_cc.py --init_method tcp://$MASTER_ADDR:3456

cd $SLURM_TMPDIR
tar -cf ~/scratch/output/checkpoints.tar work/trspes_dl/Pytorch/models/PointAugment/checkpoints/*
tar -cf ~/scratch/output/wandblogs.tar work/trspes_dl/Pytorch/models/PointAugment/wandb/*

echo "end"