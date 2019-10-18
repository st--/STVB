import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from . import q_xm
from stvb.util.util import *

class truncated_mog(q_xm.Q_xm):

	def __init__(self, input_dim):
		self.input_dim = input_dim

	def sample(self, means, variances, probabilities, num_samples, region_bounds, K=3):
		tfd = tfp.distributions
		if self.input_dim == 1:
			bimix_gauss = tfd.Mixture(cat=tfd.Categorical(probs=probabilities),
			components=[
				tfd.TruncatedNormal(loc=means[0], scale=variances[0], low = 0., high= region_bounds[0, 1]),
				tfd.TruncatedNormal(loc=means[1], scale=variances[1], low = 0., high= region_bounds[0, 1]),
				tfd.TruncatedNormal(loc=means[2], scale=variances[2], low = 0., high= region_bounds[0, 1]),
				tfd.TruncatedNormal(loc=means[3], scale=variances[3], low = 0., high= region_bounds[0, 1]), 
				tfd.TruncatedNormal(loc=means[4], scale=variances[4], low = 0., high= region_bounds[0, 1]), 
				tfd.TruncatedNormal(loc=means[5], scale=variances[5], low = 0., high= region_bounds[0, 1]), 
				tfd.TruncatedNormal(loc=means[6], scale=variances[6], low = 0., high= region_bounds[0, 1])
				])
			sample = tf.expand_dims(bimix_gauss.sample(num_samples), axis = 1)
		else:
			if self.input_dim == 2:
				means_1 = means[:5]
				means_2 = means[5:]	    
				variances_1 = variances[:5]
				variances_2 = variances[5:]
				probabilities_1 = probabilities[:5]
				probabilities_2 = probabilities[5:]

				bimix_gauss1 = tfd.Mixture(
				cat=tfd.Categorical(probs=probabilities_1),
				components=[
				tfd.TruncatedNormal(loc=means_1[0], scale=variances_1[0], low = region_bounds[0, 0], high= region_bounds[0, 1]),
				tfd.TruncatedNormal(loc=means_1[1], scale=variances_1[1], low = region_bounds[0, 0], high= region_bounds[0, 1]),
				tfd.TruncatedNormal(loc=means_1[2], scale=variances_1[2], low = region_bounds[0, 0], high= region_bounds[0, 1]),
				tfd.TruncatedNormal(loc=means_1[3], scale=variances_1[3], low = region_bounds[0, 0], high= region_bounds[0, 1]),
				tfd.TruncatedNormal(loc=means_1[4], scale=variances_1[4], low = region_bounds[0, 0], high= region_bounds[0, 1])])

				bimix_gauss2 = tfd.Mixture(
				cat=tfd.Categorical(probs=probabilities_2),
				components=[
				tfd.TruncatedNormal(loc=means_2[0], scale=variances_2[0], low = region_bounds[1, 0], high= region_bounds[1, 1]),
				tfd.TruncatedNormal(loc=means_2[1], scale=variances_2[1], low = region_bounds[1, 0], high= region_bounds[1, 1]),
				tfd.TruncatedNormal(loc=means_2[2], scale=variances_2[2], low = region_bounds[1, 0], high= region_bounds[1, 1]),
				tfd.TruncatedNormal(loc=means_2[3], scale=variances_2[3], low = region_bounds[1, 0], high= region_bounds[1, 1]),
				tfd.TruncatedNormal(loc=means_2[4], scale=variances_2[4], low = region_bounds[1, 0], high= region_bounds[1, 1])])

				sample_1 = tf.expand_dims(bimix_gauss1.sample(num_samples), axis = 1)
				sample_2 = tf.expand_dims(bimix_gauss2.sample(num_samples), axis = 1)
				sample = tf.concat((sample_1, sample_2), axis =1)
			

			if self.input_dim == 3:
				## TO DO 
				means_1 = means[:5]
				means_2 = means[5:10]	  
				means_3 = means[10:]	    
				variances_1 = variances[:5]
				variances_2 = variances[5:10]
				variances_3 = variances[10:]
				probabilities_1 = probabilities[:5]
				probabilities_2 = probabilities[5:10]
				probabilities_3 = probabilities[10:]

				bimix_gauss1 = tfd.Mixture(
				cat=tfd.Categorical(probs=probabilities_1),
				components=[
				tfd.TruncatedNormal(loc=means_1[0], scale=variances_1[0], low = region_bounds[0, 0], high= region_bounds[0, 1]),
				tfd.TruncatedNormal(loc=means_1[1], scale=variances_1[1], low = region_bounds[0, 0], high= region_bounds[0, 1]),
				tfd.TruncatedNormal(loc=means_1[2], scale=variances_1[2], low = region_bounds[0, 0], high= region_bounds[0, 1]),
				tfd.TruncatedNormal(loc=means_1[3], scale=variances_1[3], low = region_bounds[0, 0], high= region_bounds[0, 1]),
				tfd.TruncatedNormal(loc=means_1[4], scale=variances_1[4], low = region_bounds[0, 0], high= region_bounds[0, 1])])

				bimix_gauss2 = tfd.Mixture(
				cat=tfd.Categorical(probs=probabilities_2),
				components=[
				tfd.TruncatedNormal(loc=means_2[0], scale=variances_2[0], low = region_bounds[1, 0], high= region_bounds[1, 1]),
				tfd.TruncatedNormal(loc=means_2[1], scale=variances_2[1], low = region_bounds[1, 0], high= region_bounds[1, 1]),
				tfd.TruncatedNormal(loc=means_2[2], scale=variances_2[2], low = region_bounds[1, 0], high= region_bounds[1, 1]),
				tfd.TruncatedNormal(loc=means_2[3], scale=variances_2[3], low = region_bounds[1, 0], high= region_bounds[1, 1]),
				tfd.TruncatedNormal(loc=means_2[4], scale=variances_2[4], low = region_bounds[1, 0], high= region_bounds[1, 1])])

				bimix_gauss3 = tfd.Mixture(
				cat=tfd.Categorical(probs=probabilities_3),
				components=[
				tfd.TruncatedNormal(loc=means_2[0], scale=variances_3[0], low = region_bounds[2, 0], high= region_bounds[2, 1]),
				tfd.TruncatedNormal(loc=means_2[1], scale=variances_3[1], low = region_bounds[2, 0], high= region_bounds[2, 1]),
				tfd.TruncatedNormal(loc=means_2[2], scale=variances_3[2], low = region_bounds[2, 0], high= region_bounds[2, 1]),
				tfd.TruncatedNormal(loc=means_2[3], scale=variances_3[3], low = region_bounds[2, 0], high= region_bounds[2, 1]),
				tfd.TruncatedNormal(loc=means_2[4], scale=variances_3[4], low = region_bounds[2, 0], high= region_bounds[2, 1])])

				sample_1 = tf.expand_dims(bimix_gauss1.sample(num_samples), axis = 1)
				sample_2 = tf.expand_dims(bimix_gauss2.sample(num_samples), axis = 1)
				sample_3 = tf.expand_dims(bimix_gauss3.sample(num_samples), axis = 1)
				sample = tf.concat((sample_1, sample_2, sample_3), axis =1)


		return sample


	def evaluate_logprob(self, x_m, means, variances, probabilities, region_bounds, K=3):
		tfd = tfp.distributions
		if self.input_dim ==1:
			bimix_gauss = tfd.Mixture(
			cat=tfd.Categorical(probs=probabilities),
			components=[
			tfd.TruncatedNormal(loc=means[0], scale=variances[0], low = 0., high= region_bounds[0, 1]),
			tfd.TruncatedNormal(loc=means[1], scale=variances[1], low = 0., high= region_bounds[0, 1]),
			tfd.TruncatedNormal(loc=means[2], scale=variances[2], low = 0., high= region_bounds[0, 1]),
			tfd.TruncatedNormal(loc=means[3], scale=variances[3], low = 0., high= region_bounds[0, 1]),
			tfd.TruncatedNormal(loc=means[4], scale=variances[4], low = 0., high= region_bounds[0, 1]),
			tfd.TruncatedNormal(loc=means[5], scale=variances[5], low = 0., high= region_bounds[0, 1]), 
			tfd.TruncatedNormal(loc=means[6], scale=variances[6], low = 0., high= region_bounds[0, 1])
			])
			prob_MOG = bimix_gauss.log_prob(x_m)
		else:
			if self.input_dim ==2:
				means_1 = means[:5]
				means_2 = means[5:]
	    
				variances_1 = variances[:5]
				variances_2 = variances[5:]

				probabilities_1 = probabilities[:5]
				probabilities_2 = probabilities[5:]

				bimix_gauss1 = tfd.Mixture(
				cat=tfd.Categorical(probs=probabilities_1),
				components=[
				tfd.TruncatedNormal(loc=means_1[0], scale=variances_1[0], low = region_bounds[0, 0], high= region_bounds[0, 1]),
				tfd.TruncatedNormal(loc=means_1[1], scale=variances_1[1], low = region_bounds[0, 0], high= region_bounds[0, 1]),
				tfd.TruncatedNormal(loc=means_1[2], scale=variances_1[2], low = region_bounds[0, 0], high= region_bounds[0, 1]),
				tfd.TruncatedNormal(loc=means_1[3], scale=variances_1[3], low = region_bounds[0, 0], high= region_bounds[0, 1]),
				tfd.TruncatedNormal(loc=means_1[4], scale=variances_1[4], low = region_bounds[0, 0], high= region_bounds[0, 1])
				])
			        

				bimix_gauss2 = tfd.Mixture(
				cat=tfd.Categorical(probs=probabilities_2),
				components=[
				tfd.TruncatedNormal(loc=means_2[0], scale=variances_2[0], low = region_bounds[1, 0], high= region_bounds[1, 1]),
				tfd.TruncatedNormal(loc=means_2[1], scale=variances_2[1], low = region_bounds[1, 0], high= region_bounds[1, 1]),
				tfd.TruncatedNormal(loc=means_2[2], scale=variances_2[2], low = region_bounds[1, 0], high= region_bounds[1, 1]),
				tfd.TruncatedNormal(loc=means_2[3], scale=variances_2[3], low = region_bounds[1, 0], high= region_bounds[1, 1]),
				tfd.TruncatedNormal(loc=means_2[4], scale=variances_2[4], low = region_bounds[1, 0], high= region_bounds[1, 1])
				])

				prob_MOG_1 = bimix_gauss1.log_prob(x_m[:,0])
				prob_MOG_2 = bimix_gauss2.log_prob(x_m[:,1])

				prob_MOG = prob_MOG_1 + prob_MOG_2
			
			if self.input_dim ==3:
				## TO DO 
				means_1 = means[:5]
				means_2 = means[5:10]	  
				means_3 = means[10:]	    
				variances_1 = variances[:5]
				variances_2 = variances[5:10]
				variances_3 = variances[10:]
				probabilities_1 = probabilities[:5]
				probabilities_2 = probabilities[5:10]
				probabilities_3 = probabilities[10:]

				bimix_gauss1 = tfd.Mixture(
				cat=tfd.Categorical(probs=probabilities_1),
				components=[
				tfd.TruncatedNormal(loc=means_1[0], scale=variances_1[0], low = region_bounds[0, 0], high= region_bounds[0, 1]),
				tfd.TruncatedNormal(loc=means_1[1], scale=variances_1[1], low = region_bounds[0, 0], high= region_bounds[0, 1]),
				tfd.TruncatedNormal(loc=means_1[2], scale=variances_1[2], low = region_bounds[0, 0], high= region_bounds[0, 1]),
				tfd.TruncatedNormal(loc=means_1[3], scale=variances_1[3], low = region_bounds[0, 0], high= region_bounds[0, 1]),
				tfd.TruncatedNormal(loc=means_1[4], scale=variances_1[4], low = region_bounds[0, 0], high= region_bounds[0, 1])
				])
			        

				bimix_gauss2 = tfd.Mixture(
				cat=tfd.Categorical(probs=probabilities_2),
				components=[
				tfd.TruncatedNormal(loc=means_2[0], scale=variances_2[0], low = region_bounds[1, 0], high= region_bounds[1, 1]),
				tfd.TruncatedNormal(loc=means_2[1], scale=variances_2[1], low = region_bounds[1, 0], high= region_bounds[1, 1]),
				tfd.TruncatedNormal(loc=means_2[2], scale=variances_2[2], low = region_bounds[1, 0], high= region_bounds[1, 1]),
				tfd.TruncatedNormal(loc=means_2[3], scale=variances_2[3], low = region_bounds[1, 0], high= region_bounds[1, 1]),
				tfd.TruncatedNormal(loc=means_2[4], scale=variances_2[4], low = region_bounds[1, 0], high= region_bounds[1, 1])
				])

				bimix_gauss3 = tfd.Mixture(
				cat=tfd.Categorical(probs=probabilities_3),
				components=[
				tfd.TruncatedNormal(loc=means_2[0], scale=variances_3[0], low = region_bounds[2, 0], high= region_bounds[2, 1]),
				tfd.TruncatedNormal(loc=means_2[1], scale=variances_3[1], low = region_bounds[2, 0], high= region_bounds[2, 1]),
				tfd.TruncatedNormal(loc=means_2[2], scale=variances_3[2], low = region_bounds[2, 0], high= region_bounds[2, 1]),
				tfd.TruncatedNormal(loc=means_2[3], scale=variances_3[3], low = region_bounds[2, 0], high= region_bounds[2, 1]),
				tfd.TruncatedNormal(loc=means_2[4], scale=variances_3[4], low = region_bounds[2, 0], high= region_bounds[2, 1])])


				prob_MOG_1 = bimix_gauss1.log_prob(x_m[:,0])
				prob_MOG_2 = bimix_gauss2.log_prob(x_m[:,1])
				prob_MOG_3 = bimix_gauss3.log_prob(x_m[:,2])

				prob_MOG = prob_MOG_1 + prob_MOG_2 + prob_MOG_3				

        
		return prob_MOG






