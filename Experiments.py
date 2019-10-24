import sys
sys.path.append("..")

import matplotlib.pyplot as plt
from multiprocessing import Pool
from methods import *
from stvb.util.generate_data import *
from stvb.util.process_results import *

np.random.seed(1500)

##### Parameters to set 
# N_all = total number of observations.
# n_tasks = number of tasks. This is set to one cause we are not doing multi-outputs
# num_latent = number of latent GPs. This is set to one cause we are not doing multi-outputs
# sparsity = sparsity in the inputs considering M training points
# inducing_on_inputs = inducing inputs must concide with some training points or not
# num_samples = num of samples to use for the GP  
# var_steps = variational steps
# epochs= total number of epochs to be optimized for. Epochs are complete passes over the data.
# display_step_nelbo = how frequently we want to print elbo information
# debug = if we want to print additional info during training 
# n_cores = number of cores to use in multiprocessing 
# function_type = experiment we want to run
# num_realisation = number of point process realisations we want to run the algorithm for. These are run in parallel.
# kernel_type = kernel for the GP
# input_scaling = ARD kernel or not

######### SETTINGS
N_all = 200
n_tasks = 1
num_latent = 1
sparsity = False
sparsity_level = 1.0
inducing_on_inputs = True
optim_ind = False
num_samples = 1000

# var set need to be at least one with epochs > 0!!!
epochs=10
var_steps=1 
display_step_nelbo = 1

n_cores = 5
intra_op_parallelism_threads = 1
inter_op_parallelism_threads = 1

debug = False

## This can be 1, 2 or 3 and 4 (neurons), 5 (taxi) and 6 (spatio temporal taxi) for real world experiments.
function_type = 5
num_realisation = 1

# Kernels for latent GPs. This should be RadialBasis, Matern_3_2, Matern_5_2, Exponential and Periodic
kernel_type = "RadialBasis"
input_scaling = True


######### DATA GENERATION
# Import datasets
if function_type == 1:
	inputs_dimension = 1
	max_tau = 50.
	min_tau = 0.
	data = np.load('Data/synthetic_experiments/original_data_Adams/dataAdams1.npy')
	region_bounds = np.array([[min_tau, max_tau]], dtype = np.float32)

if function_type == 2:
	inputs_dimension = 1
	max_tau = 5.
	min_tau = 0.
	data = np.load('Data/synthetic_experiments/original_data_Adams/dataAdams2.npy')
	region_bounds = np.array([[min_tau, max_tau]], dtype = np.float32)

if function_type == 3:
	inputs_dimension = 1
	max_tau = 100.
	min_tau = 0.
	data = np.load('Data/synthetic_experiments/original_data_Adams/dataAdams3.npy')
	region_bounds = np.array([[min_tau, max_tau]], dtype = np.float32)

if function_type == 4:
	### Neurons data
	inputs_dimension = 2
	max_tau = 100.
	min_tau = 0.
	data = np.load('Data/neurons_dataset/xtrain.npy')
	num_realisation = 1
	region_bounds = np.array([[min_tau, max_tau],
                         		[min_tau, max_tau]], dtype = np.float32)

if function_type == 5:
	### Taxi data
	inputs_dimension = 2
	max_tau = np.array([-1.7305944541986333, 1.6697934877173213], dtype = np.float32)
	min_tau = np.array([-1.2231723175869935, 2.4109413344285886], dtype = np.float32)
	data = np.load('Data/taxi_data/xtrain.npy')
	num_realisation = 1
	region_bounds = np.array([[max_tau[0], max_tau[1]],
                         [min_tau[0], min_tau[1]]], dtype = np.float32)

if function_type == 6:
	### Taxi data time
	inputs_dimension = 3
	data = np.load('Data/taxi_data_time/xtrain_time.npy')
	num_realisation = 1
	region_bounds = np.array([[min(data[:,0]), max(data[:,0])],
                         [min(data[:,1]), max(data[:,1])],
                         [min(data[:,2]), max(data[:,2])]], dtype = np.float32)


if input_scaling == True:
	num_kernel_hyperpar = 1 + inputs_dimension
else:
	num_kernel_hyperpar = 2



def MTSM_single_realisation(num_realisation):
	print('This is the realisation number', num_realisation)
	num_realisation = 0
	if num_realisation == 0:
		subset_events = data
	else:
		(inputs, sample_intensity, constant_intensity) = generate_synthetic_data(N_all, n_tasks, num_latent, function_type, max_tau, min_tau)
		events, rejected_events = generate_events_location(inputs, constant_intensity, sample_intensity, n_tasks, function_type, num_realisation)
		subset_events = events[0][:,np.newaxis]


	## Get the inputs on a regular grid
	if function_type < 4:
		inputs = np.linspace(region_bounds[0,0], region_bounds[0,1], num=N_all)[:, np.newaxis]
	else:
		if function_type == 6:
			x_data = np.linspace(region_bounds[0,0], region_bounds[0,1], num=10)[:, np.newaxis]
			y_data = np.linspace(region_bounds[1,0], region_bounds[1,1], num=10)[:, np.newaxis]
			z_data = np.linspace(region_bounds[2,0], region_bounds[2,1], num=10)[:, np.newaxis]
			X_mesh, Y_mesh, Z_mesh = np.meshgrid(x_data,y_data, z_data)
			inputs = np.vstack([X_mesh.flatten(),Y_mesh.flatten(), Z_mesh.flatten()]).T
		else:		
			x_data = np.linspace(region_bounds[0,0], region_bounds[0,1], num=10)[:, np.newaxis]
			y_data = np.linspace(region_bounds[1,0], region_bounds[1,1], num=10)[:, np.newaxis]
			X_mesh, Y_mesh = np.meshgrid(x_data,y_data)
			inputs = np.vstack([X_mesh.flatten(),Y_mesh.flatten()]).T



	events_concatenated = np.asarray(subset_events)

	# compute the number of events 
	events_counts = np.asarray(subset_events.shape[0])[np.newaxis]

	# Get the measure of tau
	tau_region_size = np.prod(region_bounds[:,1] - region_bounds[:,0])[np.newaxis]

	# As training points we pass the events locations so that we choose the inducing points among them
	xtrain = (np.sort(events_concatenated, axis = 0))

	if function_type < 4:
		xtest = np.linspace(min_tau, max_tau, num=50)[:, np.newaxis]
	else:
		if function_type == 6:
			x_data = np.linspace(region_bounds[0,0], region_bounds[0,1], num=20)[:, np.newaxis]
			y_data = np.linspace(region_bounds[1,0], region_bounds[1,1], num=20)[:, np.newaxis]
			z_data = np.linspace(region_bounds[2,0], region_bounds[2,1], num=20)[:, np.newaxis]
			X_mesh, Y_mesh, Z_mesh = np.meshgrid(x_data,y_data, z_data)
			xtest = np.vstack([X_mesh.flatten(),Y_mesh.flatten(), Z_mesh.flatten()]).T
		else:
			x_data = np.linspace(region_bounds[0,0], region_bounds[0,1], num=50)[:, np.newaxis]
			y_data = np.linspace(region_bounds[1,0], region_bounds[1,1], num=50)[:, np.newaxis]
			X_mesh, Y_mesh = np.meshgrid(x_data,y_data)
			xtest = np.vstack([X_mesh.flatten(),Y_mesh.flatten()]).T

	# Determine the number of testing points and training points. 
	# In the synthetic experiment they are both equal to N_all. 
	num_train = xtrain.shape[0]
	num_test = xtest.shape[0]


	if debug == True:
		print('this is xtrain', xtrain.shape)
		print('this is xtest', xtest.shape)
		print('num train', num_train)
		print('num test', num_test)
		print('xtrain', xtrain.shape)


	events_locations = [subset_events]


	######### INITIALISATION
	# Initialise kernel hyperpars and lik pars
	if function_type == 1:
		lengthscale_initial = np.float32(10.0)
		sigma_initial = np.float32(1.)

	if function_type == 2:
		lengthscale_initial = np.float32(0.25)
		sigma_initial = np.float32(1.)

	if function_type == 3:
		lengthscale_initial = np.float32(15.0)
		sigma_initial = np.float32(1.)

	if function_type == 4:
		lengthscale_initial = np.float32(10.)
		sigma_initial = np.float32(1.)

	if function_type == 5:
		lengthscale_initial = np.float32(0.3)
		sigma_initial = np.float32(1.)

	if function_type == 6:
		lengthscale_initial = np.float32(0.3)
		sigma_initial = np.float32(1.)

	# Set the white noise needed for the inversion of the kernel
	white_noise = 0.01

	######### TRAINING
	return STVB_learning(xtrain, xtest, tau_region_size, region_bounds, kernel_type, 
																		events_locations, events_counts, sparsity, sparsity_level, 
																		inducing_on_inputs, optim_ind, n_tasks, num_latent, 
																		lengthscale_initial, sigma_initial, white_noise, input_scaling, num_samples, epochs, var_steps,  
																		display_step_nelbo, intra_op_parallelism_threads, inter_op_parallelism_threads, 
																		debug, num_realisation)






realisation_list = list(range(0,num_realisation,1))
pool = Pool(processes = n_cores)
results_single_realisation = pool.map(MTSM_single_realisation, realisation_list)	
	


### Process results
### Extract results from the multiprocessing output
### This function create tensors where to store objects
### It extracts results from the multiprocessing output assigning them to the corresponding tensors
(num_realisation, pred_mean, pred_var, latent_means, latent_vars,
      nelbo_values, time_iterations, 
      crossent_vector, ent_vector, ell_vector, ent_M_vector, 
      ent_x_m_vector, value_for_events_locations_vector, 
      value_for_thinned_events_vector, samples_latent_function, probabilities_mixture_vector, 
      means_mixture_vector, variances_mixture_vector,
      value_expectation_vector, 
      first_part_events_location_vector, kl_lambda_max_vector, 
      alpha_vector, beta_vector, alpha_final, beta_final, 
      time_to_train_list) = post_process_results_MTSM(results_single_realisation)




######### SAVING RESULTS
print('Saving results...')


# String to save objects
method = "STVB"
suffix = method + "_" + str(function_type)

if function_type == 1 or function_type ==2 or function_type ==3:
	folder = 'Data/synthetic_experiments/'
if function_type == 4:
	folder = 'Data/neurons_dataset/'
if function_type == 5:
	folder = 'Data/taxi_data/'
if function_type == 6:
	folder = 'Data/taxi_data_time/'

np.save(folder + 'pred_mean_' + suffix, pred_mean)
np.save(folder + 'pred_var_' + suffix, pred_var)



# Save nelbo values, time iterations and variables' values over epochs
np.save(folder + 'nelbo_values_' + suffix, nelbo_values)
np.save(folder + 'time_iterations_' + suffix, time_iterations)
np.save(folder + 'crossent_vector_' + suffix, crossent_vector)
np.save(folder + 'ent_vector_' + suffix, ent_vector)
np.save(folder + 'ell_vector_' + suffix, ell_vector)
np.save(folder + 'ent_x_m_vector_' + suffix, ent_x_m_vector)
np.save(folder + 'value_for_events_locations_vector_' + suffix, value_for_events_locations_vector)
np.save(folder + 'value_for_thinned_events_vector_' + suffix, value_for_thinned_events_vector)
np.save(folder + 'probabilities_mixture_vector_' + suffix, probabilities_mixture_vector)
np.save(folder + 'means_mixture_vector_' + suffix, means_mixture_vector)
np.save(folder + 'variances_mixture_vector_' + suffix, variances_mixture_vector)
np.save(folder + 'value_expectation_vector_' + suffix, value_expectation_vector)
np.save(folder + 'first_part_events_location_vector_' + suffix, first_part_events_location_vector)
np.save(folder + 'kl_lambda_max_vector_' + suffix, kl_lambda_max_vector)
np.save(folder + 'time_to_train_list_' + suffix, time_to_train_list)



# Save latent functions and weights info
np.save(folder + 'latent_means_' + suffix, latent_means)
np.save(folder + 'latent_variances_' + suffix, latent_vars)
np.save(folder + 'samples_latent_function_' + suffix, samples_latent_function)
np.save(folder + 'alpha_final_' + suffix, alpha_final)
np.save(folder + 'beta_final_' + suffix, beta_final)



print('Results saved')
print('Function', str(function_type))




