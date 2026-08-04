import numpy as np

def _generate_initial_parameters_for_one_layer(layer_size_1, layer_size_2, seed=42):
    """
    The weights are taken from a random normal distribution with a variance of 0.01
    """
    np.random.seed(seed)
    weights = np.random.normal(size=layer_size_1*layer_size_2, scale=0.01)
    biases = np.random.normal(size=layer_size_1, scale=0.01) # This does not mean there are similar values in the biases as in the weights. 
    return weights, biases

def generate_parameters(layer_sizes, seed=42):

    np.random.seed(seed)
    weights = []
    biases = []
    for l1,l2 in zip(layer_sizes[:-1], layer_sizes[1:]):
        w,b=_generate_initial_parameters_for_one_layer(l1,l2)
        weights.append(w)
        biases.append(b)
    return weights, biases 

layer_sizes = [2,3,4,6]
w,b=generate_parameters(layer_sizes)


