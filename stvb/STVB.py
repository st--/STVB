from __future__ import print_function

import time

import tensorflow_probability as tfp

from stvb.util import *

tf.set_random_seed(2)

class stvb(object):
    """
    The class representing the STVB model.

    Parameters
    ----------
    likelihood_func : subclass of likelihoods.Likelihood
        An object representing the likelihood function p(y|f, w).
    kernel_funcs : list of subclasses of kernels.Kernel
        A list of one kernel per latent function.
    inducing_inputs : ndarray   
        An array of initial inducing input locations. Dimensions: num_inducing * input_dim.
    num_testing_obs:
        Number of testing obs
    optimization_inducing:
        Bool. If False the location of the inducing inputs is not optimized. 
    num_samples : int
        The number of samples to approximate the expected log likelihood of the posterior.
    intra_op_parallelism_threads, inter_op_parallelism_threads: 
        Number of threads to use in tensorflow computations. 0 leaves tensorflow free to optimise
        the number of threads to use. 
    """

    def __init__(self,
                 likelihood_func,
                 kernel_funcs,
                 inducing_inputs,
                 tau_region_size,
                 region_bounds, 
                 distribution_x_m,
                 num_training_obs,
                 num_testing_obs, 
                 events_locations = None,  
                 events_counts = None,   
                 convergence_check = 0.001,
                 num_tasks = 1,
                 optimization_inducing = False,
                 num_samples = 100, 
                 debug = False,
                 intra_op_parallelism_threads = 0, 
                 inter_op_parallelism_threads = 0):

        
        # Get the likelihood function 
        self.likelihood = likelihood_func

        self.distribution_x_m = distribution_x_m

        # Get the list of kernel functions for each q
        self.kernels = kernel_funcs 

        # Get the max and min values for the region tau
        self.tau_region_size = tau_region_size
        self.region_bounds = region_bounds

        # Define the number of MonteCarlo samples
        # This is used to generated from q(M_p) and from q(x_m p)
        self.num_samples = num_samples

        # Whether we want to optimize the inducing inputs or not. Optimise only when using a 
        # level of sparsity (num inducing inputs/number of training observations) smaller than 1.
        self.optimization_inducing = optimization_inducing

        # Repeat the inducing inputs for all latent processes if we haven't been given individually (dimension QxMxD)
        # specified inputs per process.
        if inducing_inputs.ndim == 2:
            inducing_inputs = np.tile(inducing_inputs[np.newaxis, :, :], [len(self.kernels), 1, 1])

        # Initialize all model dimension constants.
        self.num_tasks = num_tasks # P
        self.num_training_obs = num_training_obs
        self.num_latent = len(self.kernels) # Q
        self.num_inducing = inducing_inputs.shape[1] # M
        self.input_dim = inducing_inputs.shape[2] # D
        self.num_testing_obs = num_testing_obs

        # Define all parameters that get optimized directly in raw form. Some parameters get
        # transformed internally to maintain certain pre-conditions (eg variance needs to be positive).

        #### Variational parameters for the latent functions
        # Means m_j for the latent functions, dimension is Q*M
        self.raw_means = tf.Variable(tf.zeros([self.num_latent, self.num_inducing]), trainable = True)

        # Covariances S_j for the latent functions, dimension Q*M*M. We initialise the lower triangular matrix of the Cholesky decomposition. 
        init_vec = np.zeros([self.num_latent] +  tri_vec_shape(self.num_inducing), dtype=np.float32) 
        self.raw_covars = tf.Variable(init_vec, trainable = True)

        # Inducing inputs. They can be optimized or not depending on optimization_inducing var. 
        self.raw_inducing_inputs = tf.Variable(inducing_inputs, trainable = False, dtype=tf.float32)  

        # Get the kernel pars from the chosen cov matrices
        self.raw_kernel_params = sum([k.get_params() for k in self.kernels], [])
        
        # Define weighths to be optimized 
        self.weights_optim = tf.Variable(tf.ones([self.num_latent, self.num_tasks]), dtype=np.float32, trainable = False)



        #self.events_locations = tf.cast(events_locations, dtype = tf.float32)
        self.events_locations = events_locations
        self.events_counts = tf.cast(events_counts, dtype = tf.float32)

        ## Define variational pars for q(lambda^*)
        self.raw_a = tf.Variable(4.*tf.ones(self.num_tasks,  dtype=np.float32), trainable = False)
        self.raw_b = tf.Variable((2./(self.events_counts[0]/(self.tau_region_size[0])))*tf.ones(self.num_tasks,  dtype=np.float32), trainable = False)

        self.raw_alpha = tf.Variable(tf.ones(self.num_tasks,  dtype=np.float32), trainable = True)
        self.raw_beta = tf.Variable(tf.ones(self.num_tasks,  dtype=np.float32), trainable = True)

        ## data we pass for training
        self.num_train = tf.placeholder(tf.float32, shape=[], name="num_train")
        self.train_inputs = tf.placeholder(tf.float32, shape=[self.num_training_obs, self.input_dim], name="train_inputs")
        self.test_inputs = tf.placeholder(tf.float32, shape=[self.num_testing_obs, self.input_dim], name="test_inputs")

        ## Initialisation of the MOG parameters
        if self.input_dim ==1:
            self.raw_probabilities = tf.Variable(tf.constant(([0.,0., 0., 0., 0.,0.,0.])), dtype = tf.float32, trainable = True)
            self.raw_means_mixture = tf.Variable(tf.constant(([0.,self.tau_region_size[0]/5., self.tau_region_size[0]/5.*2, 
                                                                self.tau_region_size[0]/5.*3, self.tau_region_size[0]/5.*4, 
                                                                self.tau_region_size[0], self.tau_region_size[0]])), dtype = tf.float32, trainable = True)
            self.raw_variances_mixture =  tf.Variable(tf.constant(([0.,0., 0.,0., 0., 0., 0.])), dtype = tf.float32, trainable = True)
        else:
            if self.input_dim ==2:
                if self.tau_region_size >15:
                    self.raw_probabilities = tf.Variable(tf.constant(([0.,0., 0., 0., 0., 0.,0., 0., 0., 0.])), dtype = tf.float32, trainable = True)
                    self.raw_means_mixture = tf.Variable(tf.constant(([20.,95., 80., 65., 20., 20., 20., 60., 90., 85.])), dtype = tf.float32, trainable = True)
                    self.raw_variances_mixture =  tf.Variable(tf.constant(([0.,0., 0., 0., 0., 0.,0., 0., 0., 0.])), dtype = tf.float32, trainable = True)
                else:
                    self.raw_probabilities = tf.Variable(tf.constant(([0.,0., 0., 0., 0., 0.,0., 0., 0., 0.])), dtype = tf.float32, trainable = True)
                    self.raw_means_mixture = tf.Variable(tf.constant(([1. , 0., 1., 1., -1., -1., 0., 1.0, -1.0, 1.0])), dtype = tf.float32, trainable = True)
                    self.raw_variances_mixture =  tf.Variable(tf.constant(([1.,1., 1., 1., 1., 1.,1., 1., 1., 1.])), dtype = tf.float32, trainable = True)

            if self.input_dim ==3:
                self.raw_probabilities = tf.Variable(tf.constant(([0.,0., 0., 0., 0., 0.,0., 0., 0., 0.,0.,0., 0., 0., 0.])), dtype = tf.float32, trainable = True)
                self.raw_means_mixture = tf.Variable(tf.constant(([1. , 0., 1., 1., -1., -1., 0., 1.0, -1.0, 1.0, 1.0, 0.0, 1.0, -1.0, 1.0])), dtype = tf.float32, trainable = True)
                self.raw_variances_mixture =  tf.Variable(tf.constant(([0.,0., 0., 0., 0., 0.,0., 0., 0., 0.,0.,0., 0., 0., 0.])), dtype = tf.float32, trainable = True)
            

        self.debug = debug

        # Do all the tensorflow bookkeeping. intra_op_parallelism_threads gives the number of cores to be used for one single operation (tf parallelises 
        # single steps within an op). inter_op_parallelism_threads gives the number of cores to be used across different operations within a single session. 
        # This is different from the multicore parallelisation. Multicore executes the code #n times in parallel on #n core. On each #n_i core tf split #m operations
        # on #m cores according to the inter_* parameter. Within each of the #m operations, tf splits the steps on #q cores. 
        # Need to pay attention to this when running on servers. Extreme parallelisation might slow down the algorithm because of cores trying to 
        # do different things at the same time. 

        session_conf = tf.ConfigProto(intra_op_parallelism_threads = intra_op_parallelism_threads, 
                                      inter_op_parallelism_threads = inter_op_parallelism_threads)
        self.session = tf.Session(config=session_conf)
        self.optimizer = tf.train.AdamOptimizer(0.005)



        ### Build our computational graph. 
        (self.nelbo, self.entropy, self.cross_ent, self.entropy_M, self.entropy_x_m,
        self.ell, self.predictions, self.kernel_mat, 
        self.value_for_events_locations, self.value_for_thinned_events, 
        self.probabilities, 
        self.means_mixture, self.variances_mixture, 
        self.value_expectation, 
        self.first_part_events_location, self.kl_lambda_max, self.samples_eta) = self._build_graph(self.raw_means,self.raw_covars, 
                                                                self.raw_inducing_inputs,
                                                                self.train_inputs,
                                                                self.num_train,
                                                                self.test_inputs, 
                                                                self.events_locations, 
                                                                self.tau_region_size, 
                                                                self.region_bounds,
                                                                self.events_counts, 
                                                                self.weights_optim, 
                                                                self.raw_probabilities, 
                                                                self.raw_means_mixture, 
                                                                self.raw_variances_mixture,
                                                                self.raw_a, 
                                                                self.raw_b,
                                                                self.raw_alpha, 
                                                                self.raw_beta)

        print('GRAPH DONE')


    def fit(self, data, optimizer, var_steps=10, epochs=200, batch_size=None, display_step=1, display_step_nelbo = 100):
        """
        Fit the STVB model to the given data.
        This function is returning the nelbo values over iterations and the itaration for which convergence is achieved.

        Parameters
        ----------
        data : subclass of datasets.DataSet  ############################ The dataset will have the events locations now
            The train inputs and outputs.
        optimizer : TensorFlow optimizer
            The optimizer to use in the fitting process.
        var_steps : int
            Number of steps to update variational parameters using variational objective (elbo).
            Set this to 1 when doing batch (all data used in optimisation once) optmisation. 
        epochs : int
            The number of epochs to optimize the model for. These give the number of complete pass through the data.
        batch_size : int
            The number of datapoints to use per mini-batch when training. If batch_size is None,
            then we perform batch gradient descent. Use all data together. 
        display_step : int
            The frequency at which the current iteration number is printed out.
        display_step_nelbo:
            The frequency at which current values are printed out and stored.
        

        Returns
        ----------
        nelbo_vector: np.array
            Values of the objective function over epochs
        
        Terms composiing the elbo that we might want to track
        
        Values of parameters over epochs 
        """

        num_train = data.num_examples
        if batch_size is None: 
            batch_size = num_train

        # Initialise counters of epochs
        old_epoch = 0
        initial_epoch = 1

        # Define tensor to store the values of different objects over epochs
        nelbo_vector = []
        crossent_vector = []
        ent_vector = []
        ell_vector = []
        ent_M_vector = []
        ent_x_m_vector = []
        value_for_events_locations_vector = []
        value_for_thinned_events_vector = []
        time_tensor = []
        value_expectation_vector  = []
        first_part_events_location_vector = []
        kl_lambda_max_vector = []

        probabilities_mixture_vector = []
        means_mixture_vector = []
        variances_mixture_vector = []
        alpha_vector = []
        beta_vector = []
        locations_vector = [] 



        ## Initialising the file where to store graph info and training info
        #merged_summary = tf.summary.merge_all()
        #train_writer = tf.summary.FileWriter('tmp/', self.session.graph)

        ######### Optimise parameters 
        # This is defining the training step that tf should execute and INITIALISE (assign values once the graph is created) all the variables.
        print('############### Construct the graph... ###############')

        self.train_step = optimizer.minimize(self.nelbo,  var_list = [self.raw_means, 
                                                                      self.raw_covars, 
                                                                      self.raw_alpha, 
                                                                      self.raw_beta, 
                                                                      self.raw_means_mixture, 
                                                                      self.raw_probabilities, 
                                                                      self.raw_variances_mixture])


        print('############### Initialise the variables ###############')
        self.session.run(tf.global_variables_initializer())
        

        # Start training phase over epochs
        while data.epochs_completed < epochs:
            # Notice that wen var steps > 1 this step is repeated for epochs + var steps. 
            # This means that for each epoch we pass throught the data var_steps times shuffling the training mini-batch. 
            num_epochs = data.epochs_completed + var_steps
            while data.epochs_completed < num_epochs:
                # Shuffling the training data and get a new batch. Next_batch is giving a list with the x of the batch as first element (need to be a tensor)
                # and with the y of the batch as second element  (a matrix)
                batch = data.next_batch(batch_size)

                # Time training step
                start = time.time()

                #print('############### About to train... ###############')
                self.session.run(self.train_step,feed_dict={self.train_inputs: batch, 
                                                             self.num_train: num_train})
                # Time training step
                end = time.time()
                time_elapsed = end - start
                #print("Execution time per training epoch", time_elapsed)
                
                ## Printing the kernel hyper parameters at each iteration
                if data.epochs_completed % display_step == 0 and data.epochs_completed != old_epoch:
                    print('Epoch' + ' ' + 'i=' + repr(data.epochs_completed) + ' ')
                    old_epoch = data.epochs_completed
                    

                    ## Print current state of the variables
                    (nelbo_value, crossent, ent, ell, ent_M, 
                    ent_x_m, value_for_events_locations, 
                    value_for_thinned_events, weights_optim, 
                    probabilities_mixture, means_mixture, variances_mixture,
                    value_expectation,  first_part_events_location,   
                    kl_lambda_max, alpha_par, beta_par) = self._print_current_state(data, num_train)



                    # Append values to save them
                    nelbo_vector.append(nelbo_value)
                    crossent_vector.append(crossent)
                    ent_vector.append(ent)
                    ell_vector.append(ell)
                    ent_M_vector.append(ent_M)
                    ent_x_m_vector.append(ent_x_m)
                    value_for_events_locations_vector.append(value_for_events_locations)
                    value_for_thinned_events_vector.append(value_for_thinned_events)
                    time_tensor.append(time_elapsed)
                    value_expectation_vector.append(value_expectation)
                    first_part_events_location_vector.append(first_part_events_location)
                    kl_lambda_max_vector.append(kl_lambda_max)


                    probabilities_mixture_vector.append(probabilities_mixture)
                    means_mixture_vector.append(means_mixture)
                    variances_mixture_vector.append(variances_mixture)
                    alpha_vector.append(alpha_par)
                    beta_vector.append(beta_par)



        # Once the optimisation is finished, convert objects to be saved in arrays
        nelbo_vector = np.asarray(nelbo_vector)
        time_tensor = np.asarray(time_tensor)
        crossent_vector = np.asarray(crossent_vector)
        ent_vector = np.asarray(ent_vector)
        ell_vector = np.asarray(ell_vector)
        ent_M_vector = np.asarray(ent_M_vector)
        ent_x_m_vector = np.asarray(ent_x_m_vector)
        value_for_events_locations_vector = np.asarray(value_for_events_locations_vector)
        value_for_thinned_events_vector = np.asarray(value_for_thinned_events_vector)
        probabilities_mixture_vector = np.asarray(probabilities_mixture_vector)
        means_mixture_vector = np.asarray(means_mixture_vector)
        variances_mixture_vector = np.asarray(variances_mixture_vector)
        value_expectation_vector = np.asarray(value_expectation_vector)
        first_part_events_location_vector = np.asarray(first_part_events_location_vector)
        kl_lambda_max_vector = np.asarray(kl_lambda_max_vector)
        alpha_vector = np.asarray(alpha_vector)
        beta_vector = np.asarray(beta_vector)


        

        return (nelbo_vector, time_tensor, crossent_vector, ent_vector, ell_vector, ent_M_vector, ent_x_m_vector, 
                value_for_events_locations_vector, value_for_thinned_events_vector, weights_optim, 
                probabilities_mixture_vector, means_mixture_vector, variances_mixture_vector,
                value_expectation_vector, first_part_events_location_vector, kl_lambda_max_vector, 
                alpha_vector, beta_vector)



    def predict(self, test_inputs, batch_size=None):
        """
        After training, predict outputs given testing inputs.

        Parameters
        ----------
        test_inputs : ndarray
            Points on which we wish to make predictions. Dimensions: num_test * D.
        batch_size : int
            The size of the batches we make predictions on. If batch_size is None, predict on the entire test set at once.

        Returns
        -------
        pred_means: np.array
            Predicted intensity mean of the test inputs. 
        pred_vars: ndarray
            Predicted intensity variance of the test inputs. 
        latent_means: ndarray
            Approximate posterior means of the GPs. 
        latent_vars: ndarray
            Approximate posterior vars of the GPs. 
        alpha: ndarray
            optimised parameter of the distribution of lambdamax
        beta: ndarray
            optimised parameter of the distribution of lambdamax
        """

        # If batch size is None, we are prediction on the all test inputs at once. 
        if batch_size is None:
            num_batches = 1
        else:
            # If instedad we consider batches, the number of batches is equal to the number of testing points 
            # divided by the number of points we want to consider for each batch.
            num_batches = util.ceil_divide(test_inputs.shape[0], batch_size)

        # Define lists for the objects we want to evaluate. We split each list depending on the number of batches. 
        test_inputs = np.array_split(test_inputs, num_batches)
        #Predicted mean and var of the intensity for each task at each test point
        pred_means = util.init_list(0.0, [num_batches])  
        pred_vars = util.init_list(0.0, [num_batches])  

        # Posterior mean and var for the GPs and the mixing weights. These are combined internally to compute the predicted intensity.
        # However, they are needed separately in order to evaluate alternative performance metrics (MC NLPL RMSE). 
        latent_means = util.init_list(0.0, [num_batches])  
        latent_vars = util.init_list(0.0, [num_batches])

        # Optimised parameters. Keep weights_optim to allow for linear combination
        samples_latent_function = util.init_list(0.0, [num_batches])

        alpha = util.init_list(0.0, [num_batches])
        beta = util.init_list(0.0, [num_batches])

        for i in range(num_batches):
            (pred_means[i], pred_vars[i], latent_means[i], 
            latent_vars[i], samples_latent_function[i], 
            alpha[i], beta[i]) = self.session.run(self.predictions, feed_dict={self.test_inputs: test_inputs[i]})

        return (np.concatenate(pred_means, axis=0), np.concatenate(pred_vars, axis=0), 
                np.concatenate(latent_means, axis=0), np.concatenate(latent_vars, axis=0), 
                np.concatenate(samples_latent_function, axis=0), 
                np.concatenate(alpha, axis=0), np.concatenate(beta, axis=0))
        
 

    def _print_current_state(self, data, num_train):
        # This is evaluating the value of the elbo with the overall dataset. 
        # It is different from the one computed during optimisation only when using mini-batches.
        # All the variables that we want to evaluate at each epoch need to be added here 
        # if debug is true these objects are also printed

        (nelbo_value, crossent, ent, ent_M, ent_x_m, 
        ell, kernel_pars, 
        value_for_events_locations, value_for_thinned_events, weights_optim,
        probabilities_mixture, means_mixture, variances_mixture,
        value_expectation,  first_part_events_location, 
        kl_lambda_max, alpha_par, beta_par) = self.session.run([self.nelbo, self.cross_ent, self.entropy, self.entropy_M, self.entropy_x_m, 
                                                                self.ell, 
                                                                self.raw_kernel_params, 
                                                                self.value_for_events_locations, self.value_for_thinned_events, self.weights_optim, 
                                                                          self.probabilities, self.means_mixture, self.variances_mixture,
                                                                          self.value_expectation,
                                                                          self.first_part_events_location, self.kl_lambda_max,
                                                                          self.raw_alpha, self.raw_beta],
                                                                          feed_dict={self.train_inputs: data.X,
                                                                          self.num_train: num_train})

        if self.debug == True:
            print('nelbo_value: ' + str(nelbo_value))
            print('cross_entropy: '+ str(crossent))
            print('entropy:' + ' ' + str(ent))    
            print('ent_M: ' + str(ent_M))
            print('ent_x_m: ' + str(ent_x_m))
            print('ell: ' + str(ell))
            print('kl_lambda_max: ' + str(kl_lambda_max))
            print('value_expectation: ' + str(value_expectation))
            print('kernel_pars:' + ' ' + str(kernel_pars)) 
            print('probabilities_mixture:' + ' ' + str(probabilities_mixture)) 
            print('means_mixture:' + ' ' + str(means_mixture)) 
            print('variances_mixture:' + ' ' + str(variances_mixture)) 

        return (nelbo_value,  crossent, ent, ell, ent_M, ent_x_m, value_for_events_locations, 
                value_for_thinned_events, weights_optim, probabilities_mixture, means_mixture, 
                variances_mixture, value_expectation,  first_part_events_location, kl_lambda_max, alpha_par, beta_par)


    def _build_graph(self, raw_means, raw_covars, raw_inducing_inputs,
                     train_inputs, num_train, test_inputs, events_locations, tau_region_size, region_bounds, events_counts, 
                     weights_optim, raw_probabilities, raw_means_mixture, raw_variances_mixture, raw_a, raw_b, raw_alpha, raw_beta):

        # This function is building the computational graph that will be then evaluated and optimized.

        # First transform all raw variables into their internal form. The optimisation is realized on the unconstrained variables. 
        # Variables are then brought back to the acceptable regions. (eg positive values for variances)
        # The values that are restricted to be positive are :
        # -- the variances for the latent functions (diagonal values of the covar mat)
        # -- the alpha and beta values of the Gamma dist on lambda_stars
        # -- the eta values of the Poisson distribution on the number of thinned events M (intensity of a Poisson distribution)

        ### Variables for the latent GPs
        # The cholesky has positive DIAGONAL entries thus we substitute the diagonal element of the chelesky 
        # with their exponential in order to garantee the positive definitness.
        # We use vec_to_tri(raw_covars) to go from one vector to a lower triangular matrix. 
        # We only optimize over the lower triangular portion of the Cholesky.
        # NB. We note that we will always operate over the cholesky space internally!!!

        # Variational covariances
        mat = util.forward_tensor(raw_covars, self.num_inducing)

        diag_mat = tf.matrix_diag(tf.matrix_diag_part(mat))
        exp_diag_mat = tf.matrix_diag(tf.exp(tf.matrix_diag_part(mat)))
        covars = mat - diag_mat + exp_diag_mat

        
        # These can varies freely
        means = raw_means
        inducing_inputs = raw_inducing_inputs
 
        kernel_mat = [self.kernels[i].kernel(inducing_inputs[i, :, :]) for i in range(self.num_latent)] 
        kernel_chol = tf.stack([tf.cholesky(k) for k in kernel_mat], 0)


        #### Placing a distribution on lambda^* and leaving eta to be free parameter of the poisson distrib. on M
        ### a, b are the prior parameters, alpha and beta are the posterior parameters for q(lambda_max)
        a = raw_a
        b = raw_b
        alpha = tf.exp(raw_alpha)
        beta = tf.exp(raw_beta)
        

        if self.debug == True:
            raw_alpha = tf.Print(raw_alpha, [self.raw_alpha], 'this is the current value of raw_alpha:')
            raw_beta = tf.Print(raw_beta, [self.raw_beta], 'this is the current value of raw_beta:')    
            alpha = tf.Print(alpha, [alpha], 'this is the current value of alpha:')
            beta = tf.Print(beta, [beta], 'this is the current value of beta:')
            a = tf.Print(a, [a], 'this is the current value of a:')
            b = tf.Print(b, [b], 'this is the current value of b:')



        ################ Sample from the full joint distributions
        ## Define number of samples 
        num_samples_lambda_max = 100
        num_samples_M = 100

        ########### Sample from the full joint distribution
        sample_lambda_max = tf.distributions.Gamma(alpha, beta).sample(num_samples_lambda_max)
        
        ### This is computing the integral for the distribution on M
        ### done differently depending on the dimension of the input space
        if self.input_dim ==1:
            full_covar = tf.matmul(covars[0, :, :], tf.transpose(covars[0, :, :])) 
            sample_u = sigmoidal_tf(tfp.distributions.MultivariateNormalFullCovariance(loc=tf.transpose(means)[:,0], covariance_matrix=full_covar).sample(num_samples_lambda_max))
            average_function_values = tf.reduce_mean(sample_u, axis = 1)
            integral_approximation = tau_region_size[0]*tf.expand_dims(average_function_values, axis =1)
        else:
            if self.input_dim ==2:
                x_fake_points = tf.expand_dims(tfp.distributions.Uniform(self.region_bounds[0, 0],self.region_bounds[0, 1]).sample(100), axis = 1)
                y_fake_points = tf.expand_dims(tfp.distributions.Uniform(self.region_bounds[1, 0],self.region_bounds[1, 1]).sample(100), axis = 1)
                fake_points = tf.concat((x_fake_points, y_fake_points), axis =1)
                kern_prods_inducing, kern_sums_inducing = self._build_interim_vals(kernel_chol, inducing_inputs, fake_points)

                samples_latent_function = self._build_samples_GP(kern_prods_inducing, kern_sums_inducing, means, covars, num_samples_lambda_max)
                samples_f_x_n = sigmoidal_tf(samples_latent_function)
                integral_approximation = tau_region_size[0]*tf.reduce_mean(samples_f_x_n, axis = 1)
            
            if self.input_dim ==3:
                x_fake_points = tf.expand_dims(tfp.distributions.Uniform(self.region_bounds[0, 0],self.region_bounds[0, 1]).sample(10), axis = 1)
                y_fake_points = tf.expand_dims(tfp.distributions.Uniform(self.region_bounds[1, 0],self.region_bounds[1, 1]).sample(10), axis = 1)
                z_fake_points = tf.expand_dims(tfp.distributions.Uniform(self.region_bounds[2, 0],self.region_bounds[2, 1]).sample(10), axis = 1)
                fake_points = tf.concat((x_fake_points, y_fake_points,z_fake_points), axis =1)
                kern_prods_inducing, kern_sums_inducing = self._build_interim_vals(kernel_chol, inducing_inputs, fake_points)

                samples_latent_function = self._build_samples_GP(kern_prods_inducing, kern_sums_inducing, means, covars, num_samples_lambda_max)
                samples_f_x_n = sigmoidal_tf(samples_latent_function)
                integral_approximation = tau_region_size[0]*tf.reduce_mean(samples_f_x_n, axis = 1)


        ## Generate eta = lambda_max*integral_approximation
        samples_eta = sample_lambda_max*(self.tau_region_size[0] - integral_approximation)
        samples_M = tfp.distributions.Poisson(samples_eta).sample(num_samples_M)

        ## Sample from q(x_m) given the M, this is again done differently depending on the dimension of the input space and thus of the y_m
        if self.input_dim ==1:
            ## These are the parameters for q(x_m)
            variances_mixture = tf.clip_by_value(tf.exp(raw_variances_mixture),0.,self.tau_region_size[0]/2)
            probabilities = tf.exp(raw_probabilities)/(tf.reduce_sum(tf.exp(raw_probabilities)))
            means_mixture = raw_means_mixture
            sample_x_m = self.distribution_x_m.sample(means_mixture, variances_mixture, probabilities, 100, self.region_bounds)
        else:
            if self.input_dim ==2:
                ## These are the parameters for q(x_m)
                variances_mixture = tf.exp(raw_variances_mixture)
                means_mixture = raw_means_mixture
                ## The probabilities need to be normalised in the 2 groups
                probabilities_1 = tf.exp(raw_probabilities[:5])/(tf.reduce_sum(tf.exp(raw_probabilities[:5])))
                probabilities_2 = tf.exp(raw_probabilities[5:])/(tf.reduce_sum(tf.exp(raw_probabilities[5:])))
                probabilities = tf.concat((probabilities_1, probabilities_2), axis =0)
                sample_x_m = self.distribution_x_m.sample(means_mixture, variances_mixture, probabilities, 100, self.region_bounds)
            
            if self.input_dim ==3:
                ## These are the parameters for q(x_m)
                variances_mixture = tf.exp(raw_variances_mixture)
                means_mixture = raw_means_mixture
                ## The probabilities need to be normalised in the 2 groups
                probabilities_1 = tf.exp(raw_probabilities[:5])/(tf.reduce_sum(tf.exp(raw_probabilities[:5])))
                probabilities_2 = tf.exp(raw_probabilities[5:10])/(tf.reduce_sum(tf.exp(raw_probabilities[5:10])))
                probabilities_3 = tf.exp(raw_probabilities[10:])/(tf.reduce_sum(tf.exp(raw_probabilities[10:])))
                probabilities = tf.concat((probabilities_1, probabilities_2, probabilities_3), axis =0)
                sample_x_m = self.distribution_x_m.sample(means_mixture, variances_mixture, probabilities, 100, self.region_bounds)


        ####### Build objective function by computing the terms composing the nelbo
        print('I am building the entropy and cross entropy terms')
        entropy = self._build_entropy(means, covars)
        cross_ent = self._build_cross_ent(means, covars, kernel_chol)
        entropy_M = self._build_entropy_M(tau_region_size, means, alpha, beta, events_counts, samples_eta, samples_M, integral_approximation, sample_lambda_max)

        entropy_x_m = self._build_entropy_x_m(means, covars, inducing_inputs, kernel_chol, weights_optim,
                                                means_mixture, variances_mixture, probabilities, alpha, beta, 
                                                tau_region_size, events_counts, integral_approximation, 
                                                sample_x_m, samples_eta, samples_M)
        kl_lambda_max = self._build_kl_lambda_star(a, b, alpha, beta)

         # Build the ell term.
        (ell, value_for_events_locations, mc_value, value_expectation, 
        first_part_events_location, second_part_events_location) = self._build_ell(means, covars, inducing_inputs, kernel_chol, 
                                                                                    train_inputs, weights_optim, region_bounds, events_locations, 
                                                                                    events_counts, tau_region_size,
                                                                                    probabilities, means_mixture, variances_mixture, 
                                                                                    alpha, beta,integral_approximation, 
                                                                                    samples_eta, samples_M, sample_lambda_max, sample_x_m)
        print('Build the terms composing the ELBO - DONE')


        nelbo = - (ell + cross_ent + entropy  + entropy_x_m + kl_lambda_max + entropy_M)
        
        # Finally, build the prediction function.
        predictions = self._build_predict(means, covars, inducing_inputs, kernel_chol, test_inputs, weights_optim, alpha, beta)
        print('Build the prediction function - DONE')

        return (nelbo, entropy, cross_ent, entropy_M, entropy_x_m, ell, predictions, kernel_mat, 
                value_for_events_locations, mc_value, probabilities, means_mixture, variances_mixture,
                value_expectation, first_part_events_location, kl_lambda_max, samples_eta)


    def _build_predict(self, means, covars, inducing_inputs,
                       kernel_chol, test_inputs, weights_optim, alpha, beta):

        ## This is giving the prior parameters for the inducing processes giving the optimised hypermerameters 
        kern_prods, kern_sums = self._build_interim_vals(kernel_chol, inducing_inputs, test_inputs)
        
        ## Given kern_prods and kern_sums this is reconstructing the parameters of the variational distribution on F to construct the predictions
        sample_means, sample_vars = self._build_sample_GP_info(kern_prods, kern_sums, means, covars)
        
        ## Sampling the latent functions F at the test inputs
        samples_latent_function = self._build_samples_GP(kern_prods, kern_sums, means, covars, 5000)

        samples_lambda_max = tfp.distributions.Gamma(alpha,beta).sample(10)

        MC_mean_f, MC_var_f =  tf.nn.moments(sigmoidal_tf(samples_latent_function), axes = 0)
        MC_mean_lambda_max, MC_var_lambda_max =  tf.nn.moments(tfp.distributions.Gamma(alpha,beta).sample(1000), axes = 0)

        pred_means = MC_mean_lambda_max * MC_mean_f
        pred_vars = MC_var_lambda_max*MC_var_f + MC_var_lambda_max*(MC_mean_f**2) + (MC_mean_lambda_max**2)*MC_var_f

        return (pred_means, pred_vars, sample_means, sample_vars, samples_latent_function, alpha, beta)


    def _build_entropy(self, means, covars): 
        # This function is building the entropy for the latent functions Eq(U)[logq(U)]
        sum_val = 0.0
        for i in range(self.num_latent):
            # Recostruct the full covars S starting from its cholesky
            full_covar = tf.matmul(covars[i, :, :], tf.transpose(covars[i, :, :]))
            trace = tf.reduce_sum(tf.matrix_diag_part(tf.cholesky_solve(covars[i, :, :],full_covar)))

            sum_val -= (CholNormal(means[i,:], covars[i, :, :]).log_prob(means[i,:]) - 0.5 * trace) 
        return sum_val
        


    def _build_entropy_M(self, tau_region_size, means, alpha, beta, events_counts, samples_eta, samples_M, integral_approximation, sample_lambda_max):
        # This function is building the entropy for M Eq(M)[-logq(M)] where M is a Poisson distribution with parameter eta

        def function_for_M_samples():
            def internal_function(samples_M):
                if self.debug == True:
                    samples_M = tf.Print(samples_M, [tf.reduce_mean(samples_M)], 'this is the current AVERAGE value of samples_M:')     
                value = samples_M*tf.log(samples_eta) - samples_eta - tf.lgamma(samples_M + 1)
                return value
            return internal_function

        q_M = tfp.distributions.Poisson(samples_eta)

        mc_values_Poisson = tfp.monte_carlo.expectation(f = function_for_M_samples(), samples = samples_M, 
                                 log_prob = q_M.log_prob, use_reparametrization = False, axis = 0)
    

        entropy = - tf.reduce_mean(mc_values_Poisson)
        return entropy
    


    def _build_kl_lambda_star(self, a, b, alpha, beta, name = 'kl_divergence_lambda_max'):
        with tf.name_scope(name):
            p = tfp.distributions.Gamma(a, b)
            q = tfp.distributions.Gamma(alpha, beta)
            negative_kl_divergence = - tfp.distributions.kl_divergence(q,p)
            return negative_kl_divergence




    def _build_entropy_x_m(self, means, covars, inducing_inputs, kernel_chol, 
                        weights_optim, means_mixture, variances_mixture, probabilities, alpha, beta, 
                        tau_region_size, events_counts, integral_approximation, sample_x_m, samples_eta, samples_M, num_components = 3): 
        ## This function is building the entropy for the thinned events locations Eq(x_m)[-logq(x_m)]. 
        ## We sample from lambda_max, then sample from M and then sample x_m
        ## Expectations wrt Poisson are computed with score gradien trick
        ## Expectation wrt Gamma are computed with reparametrisation

        q_M = tfp.distributions.Poisson(samples_eta)

        def function_x_m_samples():
            def internal_function(samples_M):
                ######### Faster way
                evaluate_pdf_trunc_MOG = self.distribution_x_m.evaluate_logprob(sample_x_m, means_mixture, variances_mixture, probabilities, self.region_bounds)

                entropy_vector = samples_M*tf.reduce_mean(evaluate_pdf_trunc_MOG)
                return entropy_vector 
            return internal_function

        mc_value = tfp.monte_carlo.expectation(f = function_x_m_samples(), samples = samples_M, log_prob = q_M.log_prob, 
                                                    use_reparametrization = False, axis = 0)
        entropy = - tf.reduce_mean(mc_value)
        return entropy




    def _build_cross_ent(self, means, covars, kernel_chol):
        # This function is building the NEGATIVE cross entropy for the latent functions 
        sum_val = 0.0
        for i in range(self.num_latent):
            full_covar = tf.matmul(covars[i, :, :], tf.transpose(covars[i, :, :]))

            trace = tf.reduce_sum(tf.matrix_diag_part(tf.cholesky_solve(kernel_chol[i, :, :],full_covar)))
            sum_val += (CholNormal(means[i, :], kernel_chol[i, :, :]).log_prob(0.0) - 0.5 * trace)
        return sum_val




    def _build_ell(self, means, covars, inducing_inputs,
                   kernel_chol, train_inputs, weights_optim, region_bounds, events_locations, 
                   events_counts, tau_region_size, probabilities, means_mixture, variances_mixture,
                   alpha, beta, integral_approximation, samples_eta, samples_M, sample_lambda_max, sample_x_m):

        ### Compute E_q(lambda_max)q(M|lambda_max)[Mlog(lambda_max)] using the score gradient trick in a nested way        
        ### Compute E_q(lambda_max)q(M|lambda_max)q(F)q({x_m})[sum_m log(sigma(-F(x_m)))] using the score gradient trick in a nested way
        ### Define MC function computed in the thinned events
        if self.debug == True:
            means_mixture = tf.Print(means_mixture, [means_mixture], 'means_mixture in build_ell')
            variances_mixture = tf.Print(variances_mixture, [variances_mixture], 'variances_mixture in build_ell')
            probabilities = tf.Print(probabilities, [probabilities], 'probabilities in build_ell')
                

        def function_M_samples():
            def internal_function(samples_M):
                M_log_lambda_max = samples_M*tf.log(sample_lambda_max) - tf.lgamma(samples_M + 1.)
                return M_log_lambda_max
            return internal_function
        

        def function_x_m_samples():
            def internal_function(samples_M):
                elements_to_sample = tf.reshape(tf.cast(samples_M, dtype = tf.int32),[-1])
                elements_to_sample_reduced = tf.cast(tf.reduce_sum(samples_M), dtype = tf.int32)
                num_samples_lambda_star = 100
                num_samples_M = 10

                ######### Faster way _ Monte Carlo
                kern_prods_thinned_locations, kern_sums_thinned_locations = self._build_interim_vals(kernel_chol, inducing_inputs, sample_x_m)
                sample_means_events_thinned, sample_vars_events_thinned = self._build_sample_GP_info(kern_prods_thinned_locations, kern_sums_thinned_locations, means, covars)
                samples_f_thinned = self._build_samples_GP(kern_prods_thinned_locations, kern_sums_thinned_locations, means, covars, self.num_samples)
                function_value = tf.reduce_mean(tf.log(sigmoidal_tf(-samples_f_thinned)))
                function_value_list = samples_M*function_value
                return function_value_list
            return internal_function



        q_M = tfp.distributions.Poisson(samples_eta)
      
        ## The Monte Carlo here for samples of x_m
        mc_value_M = tfp.monte_carlo.expectation(f = function_M_samples(), samples = samples_M, 
                        log_prob = q_M.log_prob, use_reparametrization = False, axis = 0)
        
        ## The Monte Carlo here for samples of M
        mc_value_x_m = tfp.monte_carlo.expectation(f = function_x_m_samples(), samples = samples_M, 
            log_prob = q_M.log_prob, use_reparametrization = False, axis = 0)
                     

        mc_value_part1 = tf.reduce_mean(mc_value_M)

        expectation_log_lambda_max = (tf.digamma(alpha) - tf.log(beta))[0]
        
        value_expectation = (events_counts[0]*expectation_log_lambda_max + mc_value_part1 
                             - (alpha[0]/beta[0])*tau_region_size[0] - tf.lgamma(events_counts[0] + 1.))


        ### Evaluate ELL
        ### Terms in x_n
        task_inputs = tf.cast(events_locations[0], dtype = tf.float32)
        kern_prods_events_locations, kern_sums_events_locations = self._build_interim_vals(kernel_chol, inducing_inputs, task_inputs)
        sample_means_events_locations, sample_vars_events_locations = self._build_sample_GP_info(kern_prods_events_locations, kern_sums_events_locations, means, covars)
        samples_f = self._build_samples_GP(kern_prods_events_locations, kern_sums_events_locations, means, covars, self.num_samples)
        value_for_events_locations = tf.reduce_sum(sample_means_events_locations) - tf.reduce_mean(tf.reduce_sum(tf.log(1.+tf.exp(samples_f)), axis = 1))
            
        ### Terms in x_m
        mc_value_part2 = tf.reduce_mean(mc_value_x_m)

        ell = value_expectation + value_for_events_locations + mc_value_part2

        return (ell, value_for_events_locations, mc_value_part1, value_expectation, mc_value_part2, value_for_events_locations)



    ###################################################### GP stuff - Do not change this ##########################################################

    def _build_interim_vals(self, kernel_chol, inducing_inputs, inputs):
        # Starting from the values of means and vars for the inducing process we compute intermediate values given by 
        # product of kernels that are then needed to get the parameter values for q(F).
        # Create list to save the intermediate values
        kern_prods = util.init_list(0.0, [self.num_latent])
        kern_sums = util.init_list(0.0, [self.num_latent])
        kern_prods_location = util.init_list(0.0, [self.num_latent])

        for i in range(self.num_latent):
            # Compute the term kzx and Kzx_location
            ind_train_kern = self.kernels[i].kernel(inducing_inputs[i, :, :], inputs)

            # ind_train_kern = tf.Print(ind_train_kern, [ind_train_kern], 'this is the ind train kernel', summarize = 100)
            # Compute A = Kxz.Kzz^(-1) = (Kzz^(-1).Kzx)^T. for x and x_location
            # Note that kernel_chol is the cholesky of kzz. full_covar = K_zz
            kern_prods[i] = tf.transpose(tf.cholesky_solve(kernel_chol[i, :, :], ind_train_kern))

            # Diagonal components of kxx - AKzx 
            kern_sums[i] = (self.kernels[i].diag_kernel(inputs) - util.diag_mul(kern_prods[i], ind_train_kern))
            
        # For each latent function q, this gives the Aq (NxN)
        kern_prods = tf.stack(kern_prods, 0) 
        kern_prods_location = tf.stack(kern_prods_location, 0)
        # For each latent function q, this gives the diagonal elements of k^q_xx - AqK^qzx (Nx1)
        kern_sums = tf.stack(kern_sums, 0)

        return kern_prods, kern_sums


    def _build_samples_GP(self, kern_prods, kern_sums, means, covars, n_samples):
        # This function creates the samples from the latent functions that are used in the computation of the ell term
        sample_means, sample_vars = self._build_sample_GP_info(kern_prods, kern_sums, means, covars)
        batch_size = tf.shape(sample_means)[0]
        return (sample_means + tf.sqrt(sample_vars) * tf.random_normal([n_samples, batch_size, self.num_latent], seed=1))


    def _build_sample_GP_info(self, kern_prods, kern_sums, means, covars): 
        # This function is used to get the means and the cov matrices for the latent functions
        # starting from the values of the means and the cov matrices of the inducing processes.
        # This are then used in building the samples for the latent functions and thus in evaluating ell
        # sample means and sample vars are the parameters of q(F)
        sample_means = util.init_list(0.0, [self.num_latent])
        sample_vars = util.init_list(0.0, [self.num_latent])

        for i in range(self.num_latent):
            # From the cholesky, we get back the full covariance for the inducing processes. This gives S.
            full_covar = tf.matmul(covars[i, :, :], tf.transpose(covars[i, :, :]))
            # quad form is giving the terms in (23), second formula, second term
            quad_form = util.diag_mul(tf.matmul(kern_prods[i, :, :], full_covar), tf.transpose(kern_prods[i, :, :]))

            # (23), first formula
            sample_means[i] = tf.matmul(kern_prods[i, :, :], tf.expand_dims(means[i, :], 1))
            
            # (23), second formula
            sample_vars[i] = tf.expand_dims(kern_sums[i, :] + quad_form, 1)

        # The means for each process and the diagonal terms of the covariance matrices for each process
        sample_means = tf.concat(sample_means, 1)
        sample_vars = tf.concat(sample_vars, 1)

        return sample_means, sample_vars
