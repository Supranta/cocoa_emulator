from mpi4py import MPI
import numpy as np
import time
from cocoa_emu import *
import sys

configfile = sys.argv[1]
iteration = int(sys.argv[2])

comm = MPI.COMM_WORLD
size = comm.Get_size()
my_rank = comm.Get_rank()

config = Config(configfile)

config_args    = config.config_args
config_args_io = config_args['io']

cocoa_model = CocoaModel(config_args_io)
savedir = config_args_io['savedir'] + '/iteration_%d/'%(iteration)

params_arr = np.load(savedir + '/params.npy')
params_list = get_params_list(params_arr, cocoa_model.labels)

N_total = len(params_list)
print("Total data vectors to be evaluated: %d"%(N_total))
n_local = N_total // size

def get_data_vector_list(my_rank):
    train_data_vector_list = []
    train_params_list      = []
    for i in range(my_rank * n_local, (my_rank + 1) * n_local):
        data_vector = cocoa_model.calculate_data_vector(params_list[i])
        train_data_vector_list.append(data_vector)
        train_params_list.append(params_arr[i])
    return train_params_list, train_data_vector_list

start_time = time.time()
local_params_list, local_data_vector_list = get_data_vector_list(my_rank)

if my_rank!=0:
    comm.send([local_params_list, local_data_vector_list], dest=0)
else:
    data_vector_list = local_data_vector_list
    params_list      = local_params_list
    for source in range(1,size):
        new_params_list, new_data_vector_list = comm.recv(source=source)
        data_vector_list = data_vector_list + new_data_vector_list
        params_list      = params_list + new_params_list
    train_data_vectors = np.vstack(data_vector_list)
    train_params       = np.vstack(params_list)    
    
    np.save(savedir + '/train_data_vectors.npy', train_data_vectors)
    np.save(savedir + '/train_params.npy', train_params)
    end_time = time.time()
    with open(savedir + '/timing.txt', 'w') as f:
        f.write("===============================\n")
        f.write("Iteration %d\n"%(iteration))
        f.write("===============================\n")
        f.write("Time taken for data vector calculation: %2.2f s\n"%(end_time - start_time))
MPI.Finalize