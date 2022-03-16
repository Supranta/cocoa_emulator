from multiprocessing import Pool
import torch 
import numpy as np
import time
from cocoa_emu import *
from cocoa_emu.emulator import NNEmulator, GPEmulator
from cocoa_emu.data_model import LSST_3x2
import time
import os
import sys

configfile = sys.argv[1]
ITERATION = int(sys.argv[2])

config = Config(configfile)

config_args     = config.config_args
config_args_io  = config_args['io']
config_args_emu = config_args['emulator']

# ============================
emu_type = config_args_emu['emu_type']
# ============================

# cocoa_model     = CocoaModel(config_args_io)

N_DIM         = 17
data_model    = LSST_3x2(N_DIM, config_args_io)

savedir = config_args_io['savedir']

if(emu_type=='nn'):
    if(ITERATION==0):
        train_samples = np.load(savedir + 'iteration_0/train_params.npy')
        train_data_vectors = np.load(savedir + '/iteration_0/train_data_vectors.npy')
    else:
        train_samples_list = []
        train_data_vectors_list = []

        for i in range(max(ITERATION-2, 1), ITERATION+1):
            train_samples_i      = np.load(savedir + 'iteration_%d/train_params.npy'%(i))
            train_data_vectors_i = np.load(savedir + 'iteration_%d/train_data_vectors.npy'%(i))
            train_samples_list.append(train_samples_i)
            train_data_vectors_list.append(train_data_vectors_i) 

        train_samples      = np.vstack(train_samples_list)
        train_data_vectors = np.vstack(train_data_vectors_list)
elif(emu_type=='gp'):
    train_samples = np.load(savedir + 'iteration_%d/train_params.npy'%(ITERATION))
    train_data_vectors = np.load(savedir + '/iteration_%d/train_data_vectors.npy'%(ITERATION))

CHI_SQ_CUT = float(config_args_emu['CHI_SQ_CUT'])

train_data_vectors_list = []
train_samples_list = []

for sample, data_vector in zip(train_samples, train_data_vectors):
    delta_dv = (data_vector  - data_model.dv_obs)[data_model.mask_3x2]
    chi_sq = delta_dv @ data_model.inv_cov @ delta_dv
    if(chi_sq < CHI_SQ_CUT):
        train_data_vectors_list.append(data_vector)
        train_samples_list.append(sample)

print("Training sample size: "+str(len(train_data_vectors_list)))
train_data_vectors = np.array(train_data_vectors_list)
train_samples      = np.array(train_samples_list)
print("train_samples.shape: " + str(train_samples.shape))
print("train_data_vectors.shape: " + str(train_data_vectors.shape))

start_time = time.time()
#==============================================================
#==============================================================
OUTPUT_DIM = len(data_model.dv_fid)
if(emu_type=='nn'):
    emu = NNEmulator(N_DIM, OUTPUT_DIM, data_model.dv_fid, data_model.dv_std)
    batch_size = int(config_args_emu['batch_size'])
    n_epochs   = int(config_args_emu['n_epochs'])

    emu.train(torch.Tensor(train_samples), torch.Tensor(train_data_vectors), batch_size=batch_size, n_epochs=n_epochs)
    emu.save(savedir + 'iteration_%d/model'%(ITERATION))
elif(emu_type=='gp'):
    emu = GPEmulator(N_DIM, OUTPUT_DIM, data_model.dv_fid, data_model.dv_std)
    emu.train(train_samples, train_data_vectors)
    emu.save(savedir + 'iteration_%d/model.h5'%(ITERATION))
#==============================================================
end_time = time.time()
with open(savedir + '/iteration_%d/timing.txt'%(ITERATION), 'a') as f:
    f.write("Time taken for emulator training: %2.2f s\n"%(end_time - start_time))