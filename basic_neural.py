'''
This neural network learns that the output is based on the first column of input.

It is modified from the implementation from https://iamtrask.github.io/2015/07/12/basic-python-network/
'''

import numpy as np # You will need to install a package called numpy - see pip installation instructions

# The sigmoid function is chosen due to its convenient derivative
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
					[0],
					[1],
					[1]])

# np.random.seed(1) # Why is it useful to seed the random number generator?

# Randomly generate initial synaptic weights
synapses = np.random.random((3,1))*2-1

layer0 = inputs
for i in range(60000):
	# Calculate output based on input and synaptic weights
	layer1 = sigmoid(np.dot(layer0, synapses))

	# Determine correctness
	error = outputs - layer1

	# View error as it decreases
	if (i % 5000) == 1:
		print(np.average(error))

	# Calculate the how much we need to update
	delta = error * sigmoid(layer1, deriv=True)

	# Update the synaptic weights
	synapses += np.dot(layer0.T, delta)

print("Final output:")
print(layer1)