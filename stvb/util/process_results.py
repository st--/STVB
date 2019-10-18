from scipy.spatial import distance 
import tensorflow as tf 
import numpy as np
import scipy 
from scipy.integrate import quad



def post_process_results_MTSM(multi_processing_results):
    lenght = len(multi_processing_results)
    num_realisation_list = [None]*lenght
    pred_mean_list = [None]*lenght
    pred_var_list = [None]*lenght
    latent_means_list = [None]*lenght
    latent_vars_list = [None]*lenght
    nelbo_values_list = [None]*lenght
    time_iterations_list = [None]*lenght
    crossent_vector_list = [None]*lenght
    ent_vector_list = [None]*lenght
    ell_vector_list = [None]*lenght
    ent_M_vector_list = [None]*lenght
    ent_x_m_vector_list = [None]*lenght
    value_for_events_locations_vector_list = [None]*lenght 
    value_for_thinned_events_vector_list = [None]*lenght
    samples_latent_function_list = [None]*lenght
    probabilities_mixture_vector_list = [None]*lenght
    means_mixture_vector_list = [None]*lenght
    variances_mixture_vector_list = [None]*lenght
    value_expectation_vector_list = [None]*lenght
    first_part_events_location_vector_list = [None]*lenght
    kl_lambda_max_vector_list = [None]*lenght
    alpha_vector_list = [None]*lenght
    beta_vector_list = [None]*lenght
    alpha_final_list = [None]*lenght
    beta_final_list = [None]*lenght
    time_to_train_list  = [None]*lenght


    for i in range(len(multi_processing_results)):
        single_result = multi_processing_results[i]

        num_realisation_list[i] = single_result[0]
        pred_mean_list[i]  = single_result[1]
        pred_var_list[i]  = single_result[2]
        latent_means_list[i]  = single_result[3]
        latent_vars_list[i]  = single_result[4]
        nelbo_values_list[i]  = single_result[5]
        time_iterations_list[i]  = single_result[6]
        crossent_vector_list[i]  = single_result[7]
        ent_vector_list[i]  = single_result[8]
        ell_vector_list[i]  = single_result[9]
        ent_M_vector_list[i]  = single_result[10]
        ent_x_m_vector_list[i]  = single_result[11]
        value_for_events_locations_vector_list[i]  = single_result[12]
        value_for_thinned_events_vector_list[i]  = single_result[13]
        samples_latent_function_list[i]   = single_result[14]
        probabilities_mixture_vector_list[i]   = single_result[15]
        means_mixture_vector_list[i]   = single_result[16]
        variances_mixture_vector_list[i]   = single_result[17]
        value_expectation_vector_list[i]   = single_result[18]
        first_part_events_location_vector_list[i]   = single_result[19]
        kl_lambda_max_vector_list[i]   = single_result[20]
        alpha_vector_list[i]   = single_result[21]
        beta_vector_list[i]  = single_result[22]
        alpha_final_list[i]  = single_result[23]
        beta_final_list[i]  = single_result[24]
        time_to_train_list[i] = single_result[25]

    return (num_realisation_list, pred_mean_list, pred_var_list, latent_means_list, latent_vars_list,
      nelbo_values_list, time_iterations_list, 
      crossent_vector_list, ent_vector_list, ell_vector_list, ent_M_vector_list, 
      ent_x_m_vector_list, value_for_events_locations_vector_list, 
      value_for_thinned_events_vector_list, samples_latent_function_list, probabilities_mixture_vector_list, 
      means_mixture_vector_list, variances_mixture_vector_list,
      value_expectation_vector_list, 
      first_part_events_location_vector_list, kl_lambda_max_vector_list, 
      alpha_vector_list, beta_vector_list, alpha_final_list, beta_final_list, time_to_train_list)
  
