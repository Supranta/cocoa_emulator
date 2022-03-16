import numpy as np
import time
from cocoa_emu import *
import sys

configfile = sys.argv[1]

start_time = time.time()
config = Config(configfile)

config_args    = config.config_args
config_args_io = config_args['io']

savedir = config_args_io['savedir']

cocoa_model = CocoaModel(config_args_io)

params_list = cocoa_model.get_lhs_params(config_args['lhs']['N_samples'])
params_arr = cocoa_model.get_params_array(params_list)  

np.save(savedir + '/iteration_0/params.npy', params_arr)
end_time = time.time()

with open(savedir + '/timing.txt', 'w') as f:
    f.write("Time taken for LHS sampling: %2.2f s\n"%(end_time - start_time))