import unittest
import os, sys
import numpy as np
import pandas as pd
current_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
root_path = os.path.dirname(current_path)
# Add 'WrittenAssignment' directory to sys.path
sys.path.insert(0, root_path)
print(root_path)

from Code.model import LinearRegressionModel, LinearRegressionError

class TestLinearRegressionModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.db_name = "test_data.db"
        cls.data_dict = {
            'train': os.path.join('Dataset', 'train.csv'),
            'test': os.path.join('Dataset', 'test.csv'),
            'ideal': os.path.join('Dataset', 'ideal.csv')
        }
        cls.model = LinearRegressionModel(db_name=cls.db_name, data_dict=cls.data_dict)

    def test_extract_x_y_train(self):
        x, y = self.model.extract_x_y(is_train=True, set_id=1)
        print("\nX (train):", x[:5])
        print("Y (train):", y[:5])
        self.assertEqual(len(x), len(y))
        self.assertIsInstance(x, np.ndarray)
        self.assertIsInstance(y, np.ndarray)

    def test_calculate_intercept_slope(self):
        slope, intercept = self.model.calculate_intercept_slope(set_id=1)
        print(f"\nSlope: {slope}, Intercept: {intercept}")
        self.assertIsInstance(slope, float)
        self.assertIsInstance(intercept, float)

    def test_get_yhat(self):
        y_hat = self.model.get_yhat(set_id=1)
        print("\nPredicted Y-hat:", y_hat[:5])
        self.assertIsInstance(y_hat, np.ndarray)
        self.assertFalse(np.isnan(y_hat).any())

    def test_sum_deviation_square(self):
        y_true = np.array([1, 2, 3])
        y_pred = np.array([1.1, 1.9, 3.2])
        error = self.model.sum_deviation_square(y_true, y_pred)
        print("\nSum of Squared Deviation:", error)
        self.assertGreater(error, 0)

    def test_get_max_deviation(self):
        y_train = np.array([1, 2, 3])
        y_ideal = np.array([1.1, 1.5, 3.1])
        max_dev = self.model.get_max_deviation(y_train, y_ideal)
        print("\nMax Deviation:", max_dev)
        self.assertIsInstance(max_dev, float)

    def test_choose_y_ideal(self):
        ideal_info = self.model.choose_y_ideal(set_id=1)
        print("\nChosen Ideal Function:", ideal_info)
        self.assertIn('set1', ideal_info)
        self.assertIn('ideal_set', ideal_info['set1'])

    def test_criteria_eval_test_ideal(self):
        # Use mock inputs
        y_test = 5.5
        y_ideal1 = 5.4
        y_ideal2 = 5.6
        y_ideal3 = 4.5
        y_ideal4 = 6.0

        ideal_dict = {
            'set1': {'max_deviation': 0.2, 'ideal_set': 'y1'},
            'set2': {'max_deviation': 0.3, 'ideal_set': 'y2'},
            'set3': {'max_deviation': 1.5, 'ideal_set': 'y3'},
            'set4': {'max_deviation': 0.4, 'ideal_set': 'y4'}
        }

        result = self.model.criteria_eval_test_ideal(
            y_test, y_ideal1, y_ideal2, y_ideal3, y_ideal4, ideal_dict, get_bool=False)

        print("\nTest Evaluation Result:", result)
        self.assertIn('diff_deviation', result)
        self.assertIn('ideal_set', result)

    def test_eval_test_set(self):
        df_eval = self.model.eval_test_set()
        print("\nEvaluated Test Set Sample:\n", df_eval.head())
        self.assertIn('x_test', df_eval.columns)
        self.assertIn('y_test', df_eval.columns)
        self.assertIn('diff_deviation', df_eval.columns)
        self.assertIn('ideal_func', df_eval.columns)

if __name__ == '__main__':
    unittest.main()
