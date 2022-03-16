from pyDOE import lhs
from .cobaya_utils import *
from tqdm import tqdm
import time
from schwimmbad import JoblibPool
import copy
import sys

class CocoaModel:
    def __init__(self, config_args_io):
        self.cosmology_yaml_file = config_args_io['cosmology_yaml_file']
        self.cocoa_yaml_file     = config_args_io['cocoa_yaml_file']
        self.labels, self.lhs_prior = get_lhs_priors(self.cosmology_yaml_file, self.cocoa_yaml_file)
        self.N_DIM = len(self.labels)
        self.model = get_model(self.cosmology_yaml_file)

    def get_lhs_params(self, N_SAMPLES):
        lhs_samples = lhs(self.N_DIM, N_SAMPLES)
        return get_params_list(lhs_samples, self.labels, self.lhs_prior)

    def calculate_data_vector(self, params_values):        
        likelihood   = self.model.likelihood['lsst_y1.lsst_3x2pt']
        input_params = self.model.parameterization.to_input(params_values)
        self.model.provider.set_current_input_params(input_params)
        for (component, index), param_dep in zip(self.model._component_order.items(), 
                                                 self.model._params_of_dependencies):
            depend_list = [input_params[p] for p in param_dep]
            params = {p: input_params[p] for p in component.input_params}
            compute_success = component.check_cache_and_compute(want_derived=False,
                                         dependency_params=depend_list, cached=False, **params)
        data_vector = likelihood.get_datavector(**input_params)
        return np.array(data_vector)

    def get_params_array(self, params_list):
        params_arr_list = []
        for params in params_list:
            _, params_arr = params_dict2array(params)
            params_arr_list.append(params_arr)
        return np.array(params_arr_list)
        
    def get_data_vectors(self, params_list):
        print("Calculating data vectors...") 
        """
        input_args = [(params, self.model) for params in params_list]
        with JoblibPool(4) as pool:
            data_vector_list = pool.map(self.calculate_data_vector, input_args)
        """
        # SERIAL CODE
        data_vector_list = []
        for params in params_list:
            data_vector = self.calculate_data_vector(params)
            data_vector_list.append(data_vector)        
#         """
        return np.array(data_vector_list)