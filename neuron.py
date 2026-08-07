import numpy as np

def _generate_initial_parameters_for_one_layer(layer_size_1, layer_size_2, seed=42):
    """
    The weights are taken from a random normal distribution with a variance of 0.01
    """
    np.random.seed(seed)
    weights = np.random.normal(size=(layer_size_2, layer_size_1), scale=0.01)
    biases = np.random.normal(size=(layer_size_2,), scale=0.01) # This does not mean there are similar values in the biases as in the weights. 
    return weights, biases

def generate_parameters(layer_sizes):

    weights = []
    biases = []
    for i,(l1,l2) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        w,b=_generate_initial_parameters_for_one_layer(l1,l2, seed=i)
        weights.append(w)
        biases.append(b)
    
    params = {
        'w': weights,
        'b': biases
    }

    return params 

def relu(x):
    return np.maximum(x,0)

def sigmoid(x):
    return 1/(1+np.exp(-x))
 
def forward_pass(input_value : np.ndarray, params: dict) -> np.ndarray:
    x = input_value
    for i in range(len(params['w'])):
        x = relu(np.dot(params['w'][i], x)+params['b'][i])
    return sigmoid(x)

def main() -> None:
    layer_structure = [2,3,3]
    params=generate_parameters(layer_structure)
    input_value = np.zeros(2)
    x = input_value
    print(type(x))
    print(f"x: {np.shape(x)}")
    print(f"w: {np.shape(params['w'][0])}")
    print(f"b: {np.shape(params['b'][0])}")
    x = relu(np.dot(params['w'][0], x) + params['b'][0])
    x = sigmoid(x)
    print(type(x))

if __name__ == "__main__":
    main()