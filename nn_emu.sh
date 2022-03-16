#!/bin/bash

#SBATCH --job-name=nn_emu
#SBATCH --mem-per-cpu=4GB
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --account=timeifler
#SBATCH --partition=standard
#SBATCH --qos=user_qos_timeifler
#SBATCH --time=16:00:00

source ~/.bash_profile
cd /home/u7/ssarmabo/cocoa_emulator/cocoa/Cocoa
source start_cocoa

CONFIG_FILE=configs/bahamas_nn.yaml

mpirun -np 1 python3 lsst_y1_3x2pt_lhs.py $CONFIG_FILE
export OMP_NUM_THREADS=1
for N in {0..5}
do
    mpirun -np 48 python3 lsst_y1_3x2pt_calculate_data_vector.py $CONFIG_FILE $N
    srun python3 lsst_y1_3x2pt_train.py $CONFIG_FILE $N
    srun python3 lsst_y1_3x2pt_sampling.py $CONFIG_FILE $N 1 1500
done
