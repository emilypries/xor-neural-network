'''
This neural network learns the correct output of the XOR function.

It is modified from the implementation from https://iamtrask.github.io/2015/07/12/basic-python-network/
'''

import numpy as np

def sigmoid(x, deriv=False):
	if(deriv):
		return (x*(1-x))
	return (1/(1+np.exp(-x)))

# Initialize our input and output arrays
inputs = np.array([[0,0,1],
					[0,1,1],
					[1,0,1],
					[1,1,1]])
outputs = np.array([[0],
					[1],
					[1],
					[0]])

# Randomly generate initial synaptic weights
np.random.seed(1)
syn0 = np.random.random((3,4))*2-1
syn1 = np.random.random((4,1))*2-1

layer0 = inputs
for i in range(60000):
	# Calculate output based on input and synaptic weights
	layer1 = sigmoid(np.dot(layer0, syn0))
	layer2 = sigmoid(np.dot(layer1, syn1))

	# Determine correctness
	layer2_error = outputs - layer2

	# View error as it decreases
	if (i % 5000) == 1:
		print(np.average(layer2_error))

	# Calculate how much we need to update syn1
	layer2_delta = layer2_error * sigmoid(layer2, deriv=True)

	# Calculate error and delta for layer1/syn0
	layer1_error = layer2_delta.dot(syn1.T)
	layer1_delta = layer1_error * sigmoid(layer1, deriv=True)

	# Update the synaptic weights
	syn1 += np.dot(layer1.T, layer2_delta)
	syn0 += np.dot(layer0.T, layer1_delta)

print("Final output:")
print(layer2)