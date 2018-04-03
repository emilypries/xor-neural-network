'''
Starter code for developing an XOR neural network.
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

for i in range(60000):
	# Calculate output based on input and synaptic weights
	
	# Determine correctness
	
	# Calculate the how much we need to update syn1
	
	# Calculate error and delta for layer1/syn0
	
	# Update the synaptic weights
	
# Display the final output