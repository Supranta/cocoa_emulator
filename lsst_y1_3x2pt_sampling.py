import numpy as np
import sys
import torch
import time
from cocoa_emu import *
from cocoa_emu.emulator import NNEmulator, GPEmulator
from cocoa_emu.data_model import LSST_3x2

configfile = sys.argv[1]
ITERATION = int(sys.argv[2])
tempered_run = int(sys.argv[3])
#=======================================
try:
    scale_cut_scenario = int(sys.argv[4])
except:
    scale_cut_scenario = None
#=======================================
try:
    planck_prior = bool(int(sys.argv[5])==1)
    if(planck_prior):
        print("Using planck prior...")
except:
    planck_prior = False
#=======================================
try:
    baryon_scenario = sys.argv[6]
    print("Using baryon scenario: "+baryon_scenario)
except:
    baryon_scenario = None
#=======================================

if(tempered_run==1):
    temper_schedule = [0.01, 0.05, 0.1, 0.3, 0.3]
    if(ITERATION < 5):
        temper = temper_schedule[ITERATION]
    else:
        temper = 0.5
else:
    temper = 1


config = Config(configfile)

config_args     = config.config_args
config_args_io  = config_args['io']
config_args_emu = config_args['emulator']
config_args_sampling = config_args['sampling']

try:
    save_chain = bool(config_args_sampling['save_chain'] == 'true')
except:
    save_chain = False
    
savedir = config_args_io['savedir']

N_DIM         = 17
data_model    = LSST_3x2(N_DIM, config_args_io, scale_cut_scenario, baryon_scenario=baryon_scenario)
data_model.emu_type = config_args_emu['emu_type']
### =========================================================
### =========================================================
try:
    imp_sampling = bool(int(config_args_sampling['is'])==1)
    print("Doing importance sampling...")
    temper = 0.9
    data_model.dv_obs = data_model.dv_fid
except:    
    imp_sampling = False

N_DIM      = 17
OUTPUT_DIM = 1560

if(data_model.emu_type=='nn'):
    emu = NNEmulator(N_DIM, OUTPUT_DIM, data_model.dv_fid, data_model.dv_std)    
    emu.load(savedir + 'iteration_%d/model'%(ITERATION))
elif(data_model.emu_type=='gp'):  
    emu = GPEmulator(N_DIM, OUTPUT_DIM, data_model.dv_fid, data_model.dv_std)
    emu.load(savedir + 'iteration_%d/model.h5'%(ITERATION))

# ======================================================

data_model.emu = emu

bias_fid = np.array([1.24, 1.36, 1.47, 1.60, 1.76])
bias_mask        = np.load('/home/u7/ssarmabo/cocoa_emulator/cocoa/Cocoa/lsst_y1_data/cosmo_ia_dz/dv_grad/bias_mask.npy')
shear_calib_mask = np.load('/home/u7/ssarmabo/cocoa_emulator/cocoa/Cocoa/lsst_y1_data/cosmo_ia_dz/dv_grad/shear_calib_mask.npy')
baryon_pca = np.loadtxt('/home/u7/ssarmabo/cocoa_emulator/cocoa/Cocoa/projects/lsst_y1/data/pca.txt')

cosmo_prior_lim = np.array([[1.61, 3.91],
                       [0.87, 1.07],
                       [55, 91],
                       [0.01, 0.04],
                       [0.001, 0.99]])

ia_prior_lim = np.array([[-5., 5.],
                       [-5., 5.]])

bias_prior_lim = np.array([[0.8, 3.],
                       [0.8, 3.],
                       [0.8, 3.],
                       [0.8, 3.],
                       [0.8, 3.]])

baryon_prior_lim = np.array([[-3., 12.],
                             [-2.5, 2.5]])
#                             [-2.5, 2.5],
#                             [-2.5, 2.5]])
baryon_prior_lim = 3. * baryon_prior_lim 

dz_source_std   = 0.002 * np.ones(5)
dz_lens_std     = 0.005 * np.ones(5)
shear_calib_std = 0.005 * np.ones(5)

def add_bias(bias_theta, datavector):
    for i in range(5):
        factor = (bias_theta[i] / bias_fid[i])**bias_mask[i]
        datavector = factor * datavector
    return datavector

def add_shear_calib(m, datavector):
    for i in range(5):
        factor = (1 + m[i])**shear_calib_mask[i]
        datavector = factor * datavector
    return datavector

def add_baryon_q(Q, datavector):
    for i in range(2):
        datavector = datavector + Q[i] * baryon_pca[:,i]
    return datavector

def get_data_vector_emu(theta):
    cosmo_ia_dz_theta = theta[:17]
    bias        = theta[17:22]
    shear_calib = theta[22:27]
    baryon_q    = theta[27:]
    datavector = data_model.compute_datavector(cosmo_ia_dz_theta)
    datavector = np.array(datavector)
    datavector = add_bias(bias, datavector)
    datavector = add_shear_calib(shear_calib, datavector)
    datavector = add_baryon_q(baryon_q, datavector)
    return datavector

def hard_prior(theta, params_prior):
    is_lower_than_min = bool(np.sum(theta < params_prior[:,0]))
    is_higher_than_max = bool(np.sum(theta > params_prior[:,1]))
    if is_lower_than_min or is_higher_than_max:
        return -np.inf
    else:
        return 0.
    
ns_planck = 0.97
sigma_ns  = 3. * 0.004
def lnprior(theta):
    cosmo_theta = theta[:5]
    ns          = cosmo_theta[1]
    if(planck_prior):
        ns_prior    = -0.5 * (ns - ns_planck)**2 / sigma_ns**2
    else:
        ns_prior    = 0.
    dz_source   = theta[5:10]
    ia_theta    = theta[10:12]
    dz_lens     = theta[12:17]
    bias        = theta[17:22]
    shear_calib = theta[22:27]
    baryon_q    = theta[27:]
    
    cosmo_prior = hard_prior(cosmo_theta, cosmo_prior_lim) + ns_prior
    ia_prior    = hard_prior(ia_theta, ia_prior_lim)
    bias_prior  = hard_prior(bias, bias_prior_lim)
    baryon_prior = hard_prior(baryon_q, baryon_prior_lim)
    
    dz_source_lnprior   = -0.5 * np.sum((dz_source / dz_source_std)**2)
    dz_lens_lnprior     = -0.5 * np.sum((dz_lens / dz_lens_std)**2)
    shear_calib_lnprior = -0.5 * np.sum((shear_calib / shear_calib_std)**2)
    
    return cosmo_prior + ia_prior + dz_source_lnprior + dz_lens_lnprior + \
            shear_calib_lnprior + bias_prior + baryon_prior
    
def ln_lkl(theta):
    model_datavector = get_data_vector_emu(theta)
    delta_dv = (model_datavector - data_model.dv_obs)[data_model.mask_3x2]
    return -0.5 * delta_dv @ data_model.masked_inv_cov @ delta_dv

def lnprob(theta):
    return lnprior(theta) + temper * ln_lkl(theta)

import emcee

ndim = 29

theta0    = np.array([3.0675, 0.97, 69.0, 0.0228528, 0.1199772, 
                      0., 0., 0., 0., 0.,
                      0.5, 0.,
                      0., 0., 0., 0., 0.,
                      1.24, 1.36, 1.47, 1.60, 1.76,
                      0., 0., 0., 0., 0.,
                      0., 0.])

theta_std = np.array([0.01, 0.001, 0.1, 0.001, 0.002, 
                      0.002, 0.002, 0.002, 0.002, 0.002, 
                      0.1, 0.1,
                      0.005, 0.005, 0.005, 0.005, 0.005, 
                      0.1, 0.1, 0.1, 0.1, 0.1,
                      0.005, 0.005, 0.005, 0.005, 0.005, 
                      0.1, 0.1]) 

import os

os.environ["OMP_NUM_THREADS"] = "1"

from multiprocessing import Pool

try:
    N_MCMC    = int(config_args_sampling['N_MCMC'])
    N_BURN_IN = int(config_args_sampling['N_BURNIN'])
    N_WALKERS = int(config_args_sampling['N_WALKERS'])
except:
    N_MCMC    = 10000
    N_BURN_IN = 6000
    N_WALKERS = 120
N_THIN = 10

pos0 = theta0[np.newaxis] + 3. * theta_std[np.newaxis] * np.random.normal(size=(N_WALKERS, ndim))

start_time = time.time()
with Pool() as pool:
    emu_sampler = emcee.EnsembleSampler(N_WALKERS, ndim, lnprob, pool=pool)
    emu_sampler.run_mcmc(pos0, N_MCMC, progress=True)

emu_samples = emu_sampler.chain[:,N_BURN_IN::N_THIN].reshape((-1, ndim))

if(tempered_run==1):
    N_resample = int(config_args_emu['N_resample'])
    select_indices = np.random.choice(np.arange(len(emu_samples)), replace=False, size=N_resample)
    next_training_samples = emu_samples[select_indices]
    if(save_chain):
        np.save(savedir + 'iteration_%d/tempered_chain.npy'%(ITERATION+1), emu_sampler.chain)
    np.save(savedir + 'iteration_%d/params.npy'%(ITERATION+1), next_training_samples[:,:17])
    if(imp_sampling):
        np.save(savedir + 'iteration_%d/full_params.npy'%(ITERATION+1), next_training_samples[:,:])
        log_p = emu_sampler.lnprobability[:,N_BURN_IN::N_THIN].reshape((-1))
        np.save(savedir + 'iteration_%d/logp_emu.npy'%(ITERATION+1), log_p[select_indices])
else:
    if(planck_prior):
        np.save(savedir + 'iteration_%d/chain_2pcas_ns_prior.npy'%(ITERATION), emu_sampler.chain[:,N_BURN_IN::N_THIN])
    else:
        np.save(savedir + 'iteration_%d/chain_2pcas.npy'%(ITERATION), emu_sampler.chain[:,N_BURN_IN::N_THIN])
