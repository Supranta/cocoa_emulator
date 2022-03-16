#!/bin/bash

#SBATCH --job-name=imp_sampl
#SBATCH --output=imp_sampl.out
#SBATCH --mem-per-cpu=4GB
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --account=timeifler
#SBATCH --partition=standard
#SBATCH --qos=user_qos_timeifler
#SBATCH --time=20:00:00

source ~/.bash_profile
cd /home/u7/ssarmabo/cocoa_emulator/cocoa/Cocoa
source start_cocoa

CONFIG_FILE=cocoa_emu/configs/imp_sampl.yaml

export OMP_NUM_THREADS=1
#srun python3 lsst_y1_3x2pt_sampling.py $CONFIG_FILE 3 1
mpirun -np 48 python3 lsst_y1_3x2pt_calculate_data_vector.py $CONFIG_FILE 4
