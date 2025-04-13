import unittest
import os, sys
import pandas as pd
current_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
root_path = os.path.dirname(current_path)
# Add 'WrittenAssignment' directory to sys.path
sys.path.insert(0, root_path)
from Code.dataloader import DataLoader

class TestDataLoader(unittest.TestCase):
    """Unit tests for the DataLoader class using unittest."""

    @classmethod
    def setUpClass(cls):
        cls.db_name = "test_data.db"
        cls.data_dict = {
            'train': os.path.join('Dataset', 'train.csv'),
            'test': os.path.join('Dataset', 'test.csv'),
            'ideal': os.path.join('Dataset', 'ideal.csv')
        }
        cls.loader = DataLoader(db_name=cls.db_name)

    def test_store_three_dataset_into_db(self):
        """Test storing train, test, and ideal CSVs into database."""
        train_df, test_df, ideal_df = self.loader.store_three_dataset_into_db(self.data_dict)

        print("\nTrain Set Sample:\n", train_df.head())
        print("\nTest Set Sample:\n", test_df.head())
        print("\nIdeal Set Sample:\n", ideal_df.head())

        self.assertIsInstance(train_df, pd.DataFrame)
        self.assertIsInstance(test_df, pd.DataFrame)
        self.assertIsInstance(ideal_df, pd.DataFrame)
        self.assertFalse(train_df.empty)
        self.assertFalse(test_df.empty)
        self.assertFalse(ideal_df.empty)

    def test_table_existence(self):
        """Test if tables exist in SQLite after loading."""
        self.assertTrue(self.loader.check_tb_exists("training_tb"))
        self.assertTrue(self.loader.check_tb_exists("test_tb"))
        self.assertTrue(self.loader.check_tb_exists("ideal_tb"))

    def test_load_table_as_df(self):
        """Test loading tables back from the database as DataFrame."""
        df_train = self.loader.load_table_as_df("training_tb")
        df_test = self.loader.load_table_as_df("test_tb")
        df_ideal = self.loader.load_table_as_df("ideal_tb")

        print("\nReloaded Train Sample:\n", df_train.head())
        self.assertIsInstance(df_train, pd.DataFrame)
        self.assertFalse(df_train.empty)

    def test_save_df_into_db(self):
        """Test saving a small dummy DataFrame into DB and reloading it."""
        dummy_df = pd.DataFrame({"x": [0, 1, 2], "y": [10, 20, 30]})
        self.loader.save_df_into_db(dummy_df, "dummy_tb")

        loaded_df = self.loader.load_table_as_df("dummy_tb")
        print("\nDummy Table Reloaded:\n", loaded_df)
        self.assertTrue(loaded_df.equals(dummy_df))

    def test_invalid_table_name(self):
        """Test loading from an invalid table name."""
        with self.assertRaises(Exception) as context:
            self.loader.load_table_as_df("non_existent_table")
        print("\nExpected Error Raised:\n", context.exception)


if __name__ == '__main__':
    unittest.main()
