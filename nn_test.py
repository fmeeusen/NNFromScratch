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
