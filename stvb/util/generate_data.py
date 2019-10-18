import numpy as np
import scipy.integrate as integrate
from stvb.util.util import init_list


def generate_synthetic_data(N_all, n_tasks, num_latent, function_type, max_tau, min_tau):
	np.random.seed(15)
	# Initialise required variables
	sample_intensity = init_list(0.0, [n_tasks])

	# Initiliaze the inputs and stardardize them 
	inputs = np.linspace(min_tau, max_tau, num=N_all)[:, np.newaxis]

	if function_type == 1:
		sample_intensity[0] = 2.*np.exp(-inputs/15.) + np.exp(-((inputs-25.)/10.)**2) 
	if function_type == 2:
		sample_intensity[0] = 5.*np.sin(inputs**2) + 6
	if function_type == 3:
		x = np.array((0, 25, 50, 75, 100))
		y = np.array((2, 3, 1, 2.5, 3))
		sample_intensity[0] = np.interp(inputs, x, y)
		# To add 

    # Define the constant intensities
	constant_intensity = np.max(sample_intensity[0])[np.newaxis]

	return (inputs, sample_intensity, constant_intensity)




def generate_events_location(inputs, constant_intensity, sample_intensity, n_tasks, function_type, s):
	np.random.seed(s)

	subset_events_task = [None]*n_tasks
	subset_events_rejected = [None]*n_tasks
	N_all = sample_intensity[0].shape[0]

	range_inputs = np.max(inputs) - np.min(inputs)

	for t in range(n_tasks):
		max_intensity = constant_intensity[t]
		volume = max_intensity*range_inputs
		number_events = np.random.poisson(volume)
		uniform_events = np.sort(np.random.uniform(np.min(inputs), np.max(inputs), number_events)[:,np.newaxis], axis =0)
	
		
		if function_type == 1:
			function = 2.*np.exp(-uniform_events/15.) + np.exp(-((uniform_events-25.)/10.)**2) 
			result = integrate.quad(lambda x: 2.*np.exp(-x/15.) + np.exp(-((x-25.)/10.)**2), 0, 50)
			difference = volume - result
		if function_type == 2:
			function = 5.*np.sin(uniform_events**2) + 6
			result = integrate.quad(lambda x: 5.*np.sin(x**2) + 6, 0, 5)
			difference = volume - result
		if function_type == 3:
			x_interpolate = np.array((0, 25, 50, 75, 100))
			y_interpolate = np.array((2, 3, 1, 2.5, 3))
			result = integrate.quad(lambda x: np.interp(x, x_interpolate, y_interpolate), 0, 100)
			function = np.interp(uniform_events, x_interpolate, y_interpolate)
			difference = volume - result
			# To add 

		# Assign some events to the rejection class
		uniform = np.random.uniform(0.,1., uniform_events.shape[0])
		
		accepted_events = uniform_events[uniform[:,np.newaxis] < function/constant_intensity[0]]
		rejected_events = uniform_events[uniform[:,np.newaxis] > function/constant_intensity[0]]

		subset_events_task[t] = accepted_events
		subset_events_rejected[t] = rejected_events

	return subset_events_task, rejected_events





