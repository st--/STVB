import time
import numpy as np
import tensorflow as tf
import stvb
from stvb import util


def STVB_learning(xtrain, xtest, tau_region_size, region_bounds, kernel_type, events_locations, events_counts,
				  sparsity, sparsity_level, inducing_on_inputs, optim_ind, n_tasks, num_latent, lengthscale_initial, 
				  sigma_initial, white_noise, input_scaling, num_samples, epochs, var_steps, display_step_nelbo, 
				  intra_op_parallelism_threads, inter_op_parallelism_threads, debug, num_realisation = 0):

	data = stvb.datasets.DataSet(xtrain)

	N_all = xtrain.shape[0]

	# Initialize the likelihood function.
	likelihood = stvb.likelihoods.Lik(num_missing_data = 0)
    

	# Get the dimension of the input
	dim_inputs = xtrain.shape[1]
	num_train = xtrain.shape[0]
	num_test = xtest.shape[0]


	# Initiliaze the kernels
	if kernel_type == "RadialBasis":
		kernel = [stvb.kernels.RadialBasis(dim_inputs, lengthscale = lengthscale_initial, std_dev = sigma_initial, white = white_noise, input_scaling = input_scaling) for i in range(num_latent)] 
	if kernel_type == "Matern_5_2":
		kernel = [stvb.kernels.Matern_5_2(dim_inputs, lengthscale = lengthscale_initial, std_dev = sigma_initial, white = white_noise, input_scaling = input_scaling) for i in range(num_latent)] 
	if kernel_type == "Matern_3_2":
		kernel = [stvb.kernels.Matern_3_2(dim_inputs, lengthscale = lengthscale_initial, std_dev = sigma_initial, white = white_noise, input_scaling = input_scaling) for i in range(num_latent)] 
	if kernel_type == "Exponential":
		kernel = [stvb.kernels.Exponential(dim_inputs, lengthscale = lengthscale_initial, std_dev = sigma_initial, white = white_noise, input_scaling = input_scaling) for i in range(num_latent)] 
	if kernel_type == 'Linear':
		kernel = [stvb.kernels.Linear(dim_inputs, variance = sigma_initial) for i in range(num_latent)] 
	if kernel_type == 'Periodic':
		kernel = [stvb.kernels.Periodic(period = 1.0, variance = sigma_initial, lengthscale = lengthscale_initial, white = white_noise) for i in range(num_latent)] 


	# Define the levels of sparsity we want to train the model and initialise the inducing inputs
	if sparsity == False:
		# In this case we use as inducing inputs all of the events locations 
		sparsity_vector = np.array([1.0])
		inducing_inputs = xtrain 
	else:
		# In this case we choose, among the events locations, a subset of points
		# Optimisation of the inducing inputs is only introduced when using sparsity.
		sparsity_vector = np.array([sparsity_level])
		inducing_number = int(sparsity_level*N_all)
		inducing_inputs, _ = initialize_inducing_points(xtrain, ytrain, inducing_on_inputs, num_latent, inducing_number, N_all, dim_inputs) 
		
	
	if dim_inputs == 1:
		inducing_inputs = np.linspace(region_bounds[0,0], region_bounds[0,1], num=10)[:, np.newaxis]
	else:
		if dim_inputs == 2:
			x_data = np.linspace(region_bounds[0,0], region_bounds[0,1], num=10)[:, np.newaxis]
			y_data = np.linspace(region_bounds[1,0], region_bounds[1,1], num=10)[:, np.newaxis]
			X_mesh, Y_mesh = np.meshgrid(x_data,y_data)
			inducing_inputs = np.vstack([X_mesh.flatten(),Y_mesh.flatten()]).T	
		
		if dim_inputs == 3:
			x_data = np.linspace(region_bounds[0,0], region_bounds[0,1], num=10)[:, np.newaxis]
			y_data = np.linspace(region_bounds[1,0], region_bounds[1,1], num=10)[:, np.newaxis]
			z_data = np.linspace(region_bounds[2,0], region_bounds[2,1], num=10)[:, np.newaxis]
			X_mesh, Y_mesh, Z_mesh = np.meshgrid(x_data,y_data, z_data)
			inducing_inputs = np.vstack([X_mesh.flatten(),Y_mesh.flatten(), Z_mesh.flatten()]).T	


	## Distribution MOG
	distribution_x_m = stvb.q_xm.truncated_mog(input_dim = dim_inputs)

	# Define the model
	print('Defining the model...')
	model = stvb.stvb(likelihood, kernel, inducing_inputs, tau_region_size, region_bounds, distribution_x_m, num_training_obs = num_train, 
							num_testing_obs = num_test, events_locations = events_locations,
							events_counts = events_counts, 
							num_tasks = n_tasks,  optimization_inducing = optim_ind, num_samples = num_samples, 
							debug = debug,
							intra_op_parallelism_threads = intra_op_parallelism_threads, inter_op_parallelism_threads = inter_op_parallelism_threads)

  

  


	# Define the tf  optimizer
	optimizer = tf.train.AdamOptimizer(0.005)


	# Start the training of the model
	start = time.time()

	# Train
	print('Training the model...')
	(nelbo_values, time_iterations, crossent_vector, ent_vector, ell_vector,
		ent_M_vector, ent_x_m_vector, value_for_events_locations_vector, 
	    value_for_thinned_events_vector, weights_optim,
	    probabilities_mixture_vector, means_mixture_vector, variances_mixture_vector, 
	    value_expectation_vector, 
	    first_part_events_location_vector, kl_lambda_max_vector, 
	    alpha_vector, beta_vector)  = model.fit(data, optimizer, var_steps=var_steps, epochs=epochs, display_step=1, display_step_nelbo = display_step_nelbo)
        


	end = time.time()
	time_elapsed = end-start

	print("Total training finished in seconds", time_elapsed)

	# Predictions
	print('Predicting...')
	pred_mean, pred_var, latent_means, latent_vars, samples_latent_function, alpha_final, beta_final = model.predict(xtest)


	return (num_realisation, pred_mean, pred_var, latent_means, latent_vars, nelbo_values, time_iterations,  crossent_vector, ent_vector, ell_vector,
			ent_M_vector, ent_x_m_vector, value_for_events_locations_vector, 
    		value_for_thinned_events_vector, samples_latent_function, probabilities_mixture_vector, means_mixture_vector, variances_mixture_vector,
    		value_expectation_vector, first_part_events_location_vector, kl_lambda_max_vector, alpha_vector, beta_vector, alpha_final, beta_final, 
    		time_elapsed)



