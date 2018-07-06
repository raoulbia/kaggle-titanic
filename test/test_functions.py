#!/usr/bin/env python

import numpy as np
from src.logistic_regression import regularizedCostLogReg

# test passenger
# test_passenger_survived = np.matrix([0, 3, 0, 26.0, 7.9250, 2, 9, 0])
# test_passenger_survived_not = np.matrix([0, 3, 1, 22.0, 7.2500, 2, 12, 1])
# prob = predict(X=test_passenger_survived, theta=theta_optimized)
# print('test passenger expected survived (1):', prob)
# prob = predictLogReg(X=test_passenger_survived_not, theta=theta_optimized)
# print('test passenger expected did not survive (0):', prob)


# x = -102
# x = np.arange(-1., 1., 0.5) # vector
# x = np.matrix('-1 0; 1 4')
# print(x)
# sig = expit(x)
# print(sig)
#

feature_weights =  np.array([0,0,0])
feature_weights = feature_weights.reshape(3,1) # make it a column vector

feature_matrix = np.matrix([[1, 10, 10], [1, 10, 10], [1, 10, 10]]) # note [1, ..., ...] intercept term

ground_truth = np.array([0,0,0])
ground_truth = ground_truth.reshape(3,1) # make it a column vector


print('\ninit feature_weights\n', feature_weights)
print('\nfeature_matrix\n', feature_matrix)
print('\nground_truth\n', ground_truth, '\n--------------\n')

J, grad = regularizedCostLogReg(linear, X=feature_matrix, theta=feature_weights, y=ground_truth, learning_rate=0.1,
                                reg_param=4)

print(grad)