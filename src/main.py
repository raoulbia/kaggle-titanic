import pandas as pd
import numpy as np
from scipy.special import expit
from sklearn.preprocessing import normalize

from src.cost_function import costFunction
from src.predict import predict

pd.set_option('display.width', 320)

train=pd.read_csv("../local-data/train-clean.csv", sep=",", header=0, index_col=0)
print(train.head())
X = np.matrix(normalize(train.ix[:, 1:])) # IMPORTANT: normalize!
y = np.matrix(train.ix[:, :1])
# print(X[:5,:])
# print(y[:5,:])
print('shape X', X.shape)
# print('shape y', y.shape)

# add intercept terms
a = np.ones((X.shape[0], 1))
X = np.append(a, X, axis=1)
# print(X[:5,:])
print('shape X after adding ones', X.shape)


# Initialize fitting parameters
"""
The theta values are the parameters.
Theta's are actually the weights assigned to a particular feature.
Not all parameters affect the hypothesis function equally.
Different features contribute differnetly.
The purpose of writing the ML algorithm is to come-up with values of θ that will make
the hypothesis function fit the training set well.
"""

# initial_theta
initial_theta = np.zeros((X.shape[1], 1))
# print('initial theta:', initial_theta)
# print('shape initial theta', initial_theta.shape)

cost, theta_optimized = costFunction(theta=initial_theta, X=X, y=y, learning_rate=0.01, num_iters=1500)
# print('J shape', cost.shape, ' | grad shape', grad.shape)
# print('J:', cost, '\nGradient\n', theta_optimized, '\n=============')

# test passenger
# test_passenger_survived = np.matrix([0, 3, 0, 26.0, 7.9250, 2, 9, 0])
# test_passenger_survived_not = np.matrix([0, 3, 1, 22.0, 7.2500, 2, 12, 1])
# prob = predict(X=test_passenger_survived, theta=theta_optimized)
# print('test passenger expected survived (1):', prob)
# prob = predict(X=test_passenger_survived_not, theta=theta_optimized)
# print('test passenger expected did not survive (0):', prob)

# Compute accuracy on our training set
p = predict(X=X, theta=theta_optimized);

print('Train Accuracy:', round(np.mean(p == y),2) * 100)
print('Expected Train Accuracy: 38%')


# use test data
test=pd.read_csv("../local-data/test-clean.csv", sep=",", header=0, index_col=0)
print(test.head())

X = np.matrix(normalize(test)) # IMPORTANT: normalize!
# print(X[:5,:])
print('shape X', X.shape)

# add intercept terms
a = np.ones((X.shape[0], 1))
X = np.append(a, X, axis=1)
# print(X[:5,:])
print('shape X after adding ones', X.shape)

p = predict(X=X, theta=theta_optimized)
print(type(pd.DataFrame(p)))
print('Predicted % Survived:', round(np.mean(p == 1),2) * 100)

X_ids = test.index.values
print(X_ids)

res = pd.concat([pd.DataFrame(X_ids), pd.DataFrame(p)], axis=1)
res.to_csv("../local-data/result.csv", index=False, header=False)
