import unittest
import neuron
import numpy as np

class Test_ParameterInitialization(unittest.TestCase):
    def setUp(self):
        self.layer_structure = [2, 3, 4, 5, 6]
        params = neuron.generate_parameters(self.layer_structure)
        self.flat_weights = np.concatenate(params['w'])
        self.flat_biases = np.concatenate(params['b'])
        self.flat_params = np.concatenate([self.flat_weights, self.flat_biases])

    def test_flat_weights_and_biases_have_expected_length(self):
        expected_weights = sum(self.layer_structure[i] * self.layer_structure[i+1] for i in range(len(self.layer_structure)-1))
        expected_biases = sum(self.layer_structure[1:]) 
        self.assertEqual(len(self.flat_weights), expected_weights)
        self.assertEqual(len(self.flat_biases), expected_biases)

    def test_if_all_elements_are_unique(self):
        # Sort the parameters and then take the differences between neighbouring elements 
        # This is a computationally cheaper method of checking for non-unique elements than when going through all the elements
        sorted_params = np.sort(self.flat_params)
        diffs = np.diff(sorted_params)
        no_of_duplicates = np.sum(diffs <= 1e-6)

        self.assertEqual(
            no_of_duplicates, 0,
            f"There have been {no_of_duplicates} non-unique elements found"
        )

class Test_ActivationFunctions(unittest.TestCase):
    def test_relu_for_negative_value(self):
        self.assertEqual(neuron.relu(-100),0)

    def test_sigmoid_at_zero(self):
        self.assertEqual(neuron.sigmoid(0),(1/2))

    def test_sigmoid_at_large_value(self):
        self.assertAlmostEqual(neuron.sigmoid(1e10),1,8)

    def test_sigmoid_at_small_value(self):
        self.assertAlmostEqual(neuron.sigmoid(-1e10),0,8)

class Test_ForwardPropagation(unittest.TestCase):
    def setUp(self):
        self.layer_structure = [2, 3, 4]
        self.params = neuron.generate_parameters(self.layer_structure)

    def test_forward_output_for_simple_architecture(self):
        input_value = np.zeros(self.layer_structure[0])
        out = neuron.forward_pass(input_value, self.params)
        self.assertEqual(np.shape(out), (self.layer_structure[-1],))
        

# if __name__ == "__main__":
#     unittest.main()