import numpy as np

def _generate_initial_parameters_for_one_layer(layer_size_1, layer_size_2, seed=42):
    """
    The weights are taken from a random normal distribution with a variance of 0.01
    """
    np.random.seed(seed)
    weights = np.random.normal(size=layer_size_1*layer_size_2, scale=0.01)
    biases = np.random.normal(size=layer_size_2, scale=0.01) # This does not mean there are similar values in the biases as in the weights. 
    return weights, biases

def generate_parameters(layer_sizes):

    weights = []
    biases = []
    for i,(l1,l2) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        w,b=_generate_initial_parameters_for_one_layer(l1,l2, seed=i)
        weights.append(w)
        biases.append(b)
    return weights, biases 

def relu(x):
    return np.maximum(x,0)

def sigmoid(x):
    return 1/(1+np.exp(-x))
 
    
