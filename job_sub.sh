#!/bin/bash
#SBATCH --nodes=1               # number of nodes
#SBATCH --gpus-per-node=4            # number of gpus per node
#SBATCH --cpus-per-task=8       # number of threads per task
#SBATCH --mem=32G
#SBATCH --job-name="test-multi-gpu"
#SBATCH --time=00:30:00        # Specify run time 
#SBATCH --output=%N-%j.out    # Specify output file format generated by python script
#SBATCH --error=%N-%j.error
#SBATCH --mail-user=yuwei.cao@ubc.ca    # Request email notifications
#SBATCH --mail-type=ALL

# Load python module, and additional required modules
module purge 
module load python/3.10 scipy-stack cuda cudnn

srun --tasks-per-node=1 bash << EOF
virtualenv --no-download $SLURM_TMPDIR/venv
source $SLURM_TMPDIR/venv/bin/activate
pip install --no-index --upgrade pip
pip install --no-index torch torchvision torchtext torchaudio
pip install --no-index -r ~/code/trspes_dl/requirements.txt
pip install laspy[laszip]
EOF

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Set environment variables
export NCCL_BLOCKING_WAIT=1  #Set this environment variable if you wish to use the NCCL backend for inter-GPU communication.
export MASTER_ADDR=$(hostname) #Store the master node’s IP address in the MASTER_ADDR environment variable.

#Get the current datetime and export to python script
#export DATETIME=`date +%Y_%m_%d_%H_%M_%S`

# Start a single node ray cluster before calling the Python script
#ray start --head --node-ip-address="$MASTER_ADDR" --port=34567 --num-cpus=$NUM_GPU --num-gpus=$NUM_GPU --block &

# Wait 10 seconds for ray setup
sleep 10
  
#Print (echo) info to output file
echo "r$SLURM_NODEID master: $MASTER_ADDR"
echo "r$SLURM_NODEID Launching test dl cc script"
# 3mins so far

# code transfer
cd $SLURM_TMPDIR
rm -r work
mkdir work
cd work
git clone git@github.com:yuwei-cao-git/trspes_dl.git
cd ./trspes_dl

# data transfer
mkdir -p data
# extract an archive to a different directory, the ‘-C’ option is followed by the destination path
tar -xf $project/data/rmf_laz.tar -C ./data
ls $SLURM_TMPDIR/work/trspes_dl

# Log experiment variables
wandb offline

#Run python script
# The $((SLURM_NTASKS_PER_NODE * SLURM_JOB_NUM_NODES)) variable tells the script how many processes are available for this execution. “srun” executes the script <tasks-per-node * nodes> times
python ./Pytorch/models/PointAugment/main_cc.py

cd $SLURM_TMPDIR
tar -cf ~/scratch/output/checkpoints.tar work/trspes_dl/Pytorch/models/PointAugment/checkpoints/*
tar -cf ~/scratch/output/wandblogs.tar work/trspes_dl/Pytorch/models/PointAugment/wandb/*

echo "end"