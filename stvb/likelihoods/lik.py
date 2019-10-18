import numpy as np
import tensorflow as tf
import scipy

from . import likelihood
from stvb.util.util import *


# Implementation of the likelihood - not actually needed for STVB

class Lik(likelihood.Likelihood):
    def __init__(self, num_missing_data, offset_type = 'task', num_tasks = 1, 
        point_estimate = 'mean', trainable_offset = False): 
        self.point_estimate = point_estimate
        self.num_tasks = num_tasks
   


    def log_cond_prob(self, model, means, covars, inducing_inputs, kernel_chol, train_inputs, train_outputs, weights_optim, 
                        region_bounds, eta, events_locations, events_counts, beta, alpha, num_samples_ell):
        
        ##### Discrete model
        to_print = 'not yet defined'
        return (to_print)

    def get_params(self):
        return []
    

    def predict(self, latent_means, latent_vars, weights_optim):
        
        to_print = 'not yet defined'
        return (to_print)