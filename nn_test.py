import unittest
import neuron
import numpy as np

class Test_ParameterInitialization(unittest.TestCase):
    def setUp(self):
        self.layer_structure = [2, 3, 4, 5, 6]
        params = neuron.generate_parameters(self.layer_structure)
        self.flat_weights = np.concatenate([w.ravel() for w in params['w']])
        self.flat_biases = np.concatenate([b.ravel() for b in params['b']])
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

class Test_ForwardPropagationWithDummyValues(unittest.TestCase):
    def setUp(self):
        self.layer_structure = [2, 3, 4]
        self.dummy_input = np.zeros(self.layer_structure[0])
        self.dummy_params = {
            'w': [np.array(np.ones((self.layer_structure[1], self.layer_structure[0]))),
                  np.array(np.ones((self.layer_structure[2], self.layer_structure[1]))),
                ],
            'b': [
                np.array([1, -1, 2]),
                np.array([0, 0, 0, 0])
                ]     
                  }

    def test_output_for_dummy_input_and_params(self):
        # --- Manual computation (computed using Claude)---
        # Layer 0: input is all zeros, so w0 @ x = 0 regardless of weights.
        #   relu(0 + b0) = relu([1, -1, 2]) = [1, 0, 2]
        # Layer 1: w1 @ [1, 0, 2] = row sums since all weights are 1 -> [3, 3, 3, 3]
        #   relu([3, 3, 3, 3] + [0, 0, 0, 0]) = [3, 3, 3, 3]
        # Final sigmoid: sigmoid(3) ≈ 0.9525741268224334, applied to all 4 elements
        neuralnet_output = neuron.forward_pass(self.dummy_input, self.dummy_params)

        expected = np.full(4, 1 / (1 + np.exp(-3)))
        np.testing.assert_allclose(neuralnet_output, expected, rtol=1e-7)
        self.assertEqual(neuralnet_output.shape, (4,))

class Test_lossFunction(unittest.TestCase):
    def setUp(self):
        self.y_true = np.array([[1, 2, 3], 
                                [4, 5., 6.],
                                [7., 8., 9.]]) # Make up a set of 3 observations with 3 categories 
        self.y_pred = np.array([[10, 11, 12],
                               [13, 14, 15],
                               [16, 17, 18]]

                               ) # TO DO: now do it for a non-symmetric y_pred aswell. 
    # The correct output is 1/3 * (((0)^2 + (1/3)^2+ (1/3)^2)) + 2* ((1/3)^2 + (1/3)^2 + (2/3)^2)) = 1/3 * (2/9 + 2*(2/3)) = 14/27
    def test_loss_function_gives_correct_result(self):
        loss_value = neuron.mse_loss(self.y_true, self.y_pred)
        num_observations, num_categories = np.shape(self.y_true)
        error = 0 
        for i in range(num_observations):
            error+=np.sum((self.y_true[i]-self.y_pred[i])**2)
        error = error/num_observations
        self.assertAlmostEqual(neuron.mse_loss(self.y_true, self.y_pred), error, places=4)

# if __name__ == "__main__":
#     unittest.main()