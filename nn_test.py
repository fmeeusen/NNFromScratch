import unittest
import neuron
import numpy as np

class Test_InitialParameters(unittest.TestCase):
    def test_number_of_parameters(self):
        layer_structure = [2,3,4,5,6]
        weights,biases=neuron.generate_parameters(layer_structure)
        flat_weights = np.concatenate(weights)
        flat_biases = np.concatenate(biases)

        self.assertEqual(len(flat_weights), 2*3+3*4+4*5+5*6)
        self.assertEqual(len(flat_biases), 3+4+5+6)

    def test_if_all_unique_elements(self):
        layer_structure = [2,2,3,3]
        weights,biases=neuron.generate_parameters(layer_structure)
        flat_weights = np.concatenate(weights)
        flat_biases = np.concatenate(biases)
        flat_params = np.append(flat_weights, flat_biases)

        # Sort the parameters and then take the differences between neighbouring elements 
        # This is a computationally cheaper method of checking for non-unique elements

        sorted_params = np.sort(flat_params)
        diffs = np.diff(sorted_params)
        no_of_duplicates = np.sum(diffs <= 1e-5)

        self.assertEqual(
            no_of_duplicates, 0,
            f"There have been {no_of_duplicates} non-unique elements found"
        )
