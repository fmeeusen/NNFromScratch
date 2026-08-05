import unittest
import neuron
import numpy as np

class Test_InitialParameters(unittest.TestCase):
    def setUp(self):
        self.layer_structure = [2, 3, 4, 5, 6]
        weights, biases = neuron.generate_parameters(self.layer_structure)
        self.flat_weights = np.concatenate(weights)
        self.flat_biases = np.concatenate(biases)
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
