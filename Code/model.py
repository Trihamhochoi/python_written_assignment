from typing import Literal, Tuple, List, Dict
import math
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from dataloader import DataLoader


class LinearRegressionError(Exception):
    """Custom exception for linear regression operations."""
    pass


class LinearRegressionModel:
    """
    A simple linear regression model to compute slope, intercept,
    and mean deviation error between two sets of data.
    Initialization
    train_df: The train dataframe is loaded from database
    test_df: The test dataframe is loaded from database
    ideal_df: The ideal dataframe is loaded from database
    """

    def __init__(self, train_df, test_df, ideal_df):
        # Initialization
        self.train_df = train_df
        self.test_df = test_df
        self.ideal_df = ideal_df
        self.x = None
        self.slope = None
        self.intercept = None

    def extract_x_y(self, is_train: bool, set_id: Literal[1, 2, 3, 4] = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract X, Y from train and test dataframes.
        :param is_train: Choose between train and test dataframes.
        :param set_id: Choose from [1,2,3,4] as set ID
        :return: X and y values
        """
        if is_train:
            x, y = self.train_df['x'].values, self.train_df[f'y{set_id}'].values
            return x, y
        else:
            x, y = self.test_df['x'].values, self.test_df['y'].values
            return x, y

    def calculate_intercept_slope(self, set_id: Literal[1, 2, 3, 4] = 1):
        """
        Fit the model to the training data and compute slope and intercept.

        Parameters:
            set_id: Choose the setID, there are 4 sets

        Raises:
            LinearRegressionError: if x and y have different lengths or are too small
        """
        self.x, y = self.extract_x_y(is_train=True, set_id=set_id)
        if self.x.shape[0] != y.shape[0]:
            raise LinearRegressionError("x and y must be the same length.")
        if len(self.x) < 2:
            raise LinearRegressionError("At least two points are required for regression.")

        x_mean = np.mean(self.x)
        y_mean = np.mean(y)

        numerator = np.sum((self.x - x_mean) * (y - y_mean))
        denominator = np.sum((self.x - x_mean) ** 2)

        if denominator == 0:
            raise LinearRegressionError("Denominator for slope calculation is zero.")

        slope = numerator / denominator
        intercept = y_mean - slope * x_mean
        return slope, intercept

    def get_yhat(self, set_id: Literal[1, 2, 3, 4] = 1) -> np.ndarray:
        """
        Calculate y hat given train x-values based on slope and intercept was calculated before.

        Parameters:
            set_id: Choose the setID, there are 4 sets
        Returns:
            np.ndarray: Predicted y-values

        Raises:
            LinearRegressionError: if the model is not fitted
        """
        slope, intercept = self.calculate_intercept_slope(set_id=set_id)
        y_hat = slope * self.x + intercept
        return y_hat

    # compute the sum of squared differences between it and each ideal function:
    @staticmethod
    def sum_deviation_square(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate sum deviation squared between actual and predicted y-values.

        Parameters:
            y_true (np.ndarray): Actual y-values
            y_pred (np.ndarray): Predicted y-values

        Returns:
            float: Mean absolute deviation error
        """
        return np.sum(np.square(y_true - y_pred))

    # compute the max deviation between y_ideal and y_train
    @staticmethod
    def get_max_deviation(y_train: np.ndarray, y_ideal: np.ndarray) -> float:
        """
        Return the maximum deviation between train and ideal y-values.
        :param y_train: array of y train values
        :param y_ideal: array of y ideal values
        :return:
        """
        max_deviation = np.max(np.abs(y_train - y_ideal))
        return float(max_deviation)

    # According to the defined metrics, Choose the best ideal function for assigned set
    def choose_y_ideal(self,
                       set_id: Literal[1, 2, 3, 4] = 1) -> dict:
        """
        According to the defined metrics, Choose the best ideal function for assigned set
        :param set_id: set id we want to choose
        :return: the chosen ideal function gives the minimum of sum deviation error
        """
        # Calculate y_pred
        y_pred = self.get_yhat(set_id=set_id)

        # Get shape of y
        num_y = self.ideal_df.shape[1]
        error_dict = {}
        for i in tqdm(range(num_y - 1), f"Calculating ideal set"):
            y_ideal = self.ideal_df.iloc[:, i + 1].values
            square_error = self.sum_deviation_square(y_true=y_ideal, y_pred=y_pred)
            error_dict[f'y{i + 1}'] = float(square_error)

        # Select least square error
        value_arr = np.array(list(error_dict.values()))

        # Get min error and min ideal set
        min_value = np.min(value_arr)
        min_ideal_y = list(error_dict.keys())[np.argmin(value_arr)]
        print(f">> Best ideal set of set {set_id} is {min_ideal_y} with min values {min_value}")

        # Get max deviation between y_ideal and y_train
        x_train, y_train = self.extract_x_y(is_train=True, set_id=set_id)
        y_ideal = self.ideal_df[min_ideal_y].values
        max_deviation = self.get_max_deviation(y_train=y_train, y_ideal=y_ideal)

        ideal_dict = {f"set{set_id}": {'ideal_set': min_ideal_y,
                                       'min_squared_error': error_dict[min_ideal_y],
                                       'max_deviation': max_deviation, }}
        return ideal_dict

    def criteria_eval_test_ideal(self,
                                 y_test: float,
                                 y_ideal1: float,
                                 y_ideal2: float,
                                 y_ideal3: float,
                                 y_ideal4: float,
                                 ideal_dict: dict,get_bool:bool) -> Dict:
        # Calculate the difference between y_test and 4 y_ideal:
        diff_deviation1 = abs(y_test - y_ideal1)
        diff_deviation2 = abs(y_test - y_ideal2)
        diff_deviation3 = abs(y_test - y_ideal3)
        diff_deviation4 = abs(y_test - y_ideal4)

        # Extract 4 max_deviation
        max_dev1 = ideal_dict['set1']['max_deviation']
        max_dev2 = ideal_dict['set2']['max_deviation']
        max_dev3 = ideal_dict['set3']['max_deviation']
        max_dev4 = ideal_dict['set4']['max_deviation']

        a = np.array([[diff_deviation1, max_dev1],
                      [diff_deviation2, max_dev2],
                      [diff_deviation3, max_dev3],
                      [diff_deviation4, max_dev4]])
        arr_max_dev = a[:, 1]
        # sort value descending by diff value (first value of row)
        sort_a = a[a[:, 0].argsort()[::-1]]

        # Check condition if diff deviation (1st) is lower than the maximum deviation (2nd) by sqrt(2)
        cond = sort_a[:, 0] < sort_a[:, 1] * np.sqrt(2)

        # get the fist index of condition
        try:
            first_idx = np.where(cond)[0][0]
        except IndexError:
            re = np.array([np.nan, np.nan])
        else:
            re = sort_a[first_idx]

        # Get final result dictionary including 2 values diff_deviation and ideal set
        result_dict = {}
        if not np.isnan(re).all():
            result_dict['diff_deviation'] = round(float(re[0]),3)
            for idx, v in enumerate(arr_max_dev):
                if v == re[1]:
                    ideal_set = ideal_dict[f'set{idx + 1}']['ideal_set']
                    result_dict['ideal_set'] = ideal_set
        else:
            result_dict['diff_deviation'] = np.nan
            result_dict['ideal_set'] = np.nan

        # Compare between diff deviation to the maximum deviation of corresponding set by sqrt(2)
        if get_bool:
            result = {'set1': diff_deviation1 < (max_dev1 * math.sqrt(2)),
                      'set2': diff_deviation2 < (max_dev2 * math.sqrt(2)),
                      'set3': diff_deviation3 < (max_dev3 * math.sqrt(2)),
                      'set4': diff_deviation4 < (max_dev4 * math.sqrt(2))}
            return result
        else:
            return result_dict

    # Evaluate the ideal set with test set
    def eval_test_set(self):
        # Calculate the best four of ideal function
        ideal_dict = {}
        ideal_feat_rename_dict = {}
        for i in range(1, 5):
            best_ideal_dict = self.choose_y_ideal(set_id=int(i))

            # get ideal y
            feat = best_ideal_dict[f'set{i}']['ideal_set']

            # Extract to ideal_dict and ideal_feat
            ideal_dict.update(best_ideal_dict)
            ideal_feat_rename_dict[feat] = f"set{i}_{feat}"

        #print(ideal_dict)
        # merge test df and the best 4 ideal df
        feat_col = ['x']
        id_feat_ls = list(ideal_feat_rename_dict.keys())
        feat_col.extend(id_feat_ls)
        chosen_ideal_df = self.ideal_df[feat_col].rename(columns=ideal_feat_rename_dict)

        df_merge = (pd.merge(left=self.test_df, right=chosen_ideal_df, on='x', how='left')
                    .rename(columns={'y': 'y_test'}))
        # print(df_merge.info())
        # print(df_merge.describe())

        # Calculate the mapping between each pair x,y of test set with corresponding ideal function
        rename_ls = list(ideal_feat_rename_dict.values())
        mapping_criterion = df_merge.apply(lambda x: self.criteria_eval_test_ideal(y_test=x['y_test'],
                                                                                   y_ideal1=x[rename_ls[0]],
                                                                                   y_ideal2=x[rename_ls[1]],
                                                                                   y_ideal3=x[rename_ls[2]],
                                                                                   y_ideal4=x[rename_ls[3]],
                                                                                   ideal_dict=ideal_dict,get_bool=False
                                                                                   ), axis=1)
        # bool_criterion = df_merge.apply(lambda x: self.criteria_eval_test_ideal(y_test=x['y_test'],
        #                                                                            y_ideal1=x[rename_ls[0]],
        #                                                                            y_ideal2=x[rename_ls[1]],
        #                                                                            y_ideal3=x[rename_ls[2]],
        #                                                                            y_ideal4=x[rename_ls[3]],
        #                                                                            ideal_dict=ideal_dict, get_bool=True
        #                                                                            ), axis=1)

        # df_merge['result'] = mapping_criterion
        df_merge[['diff_deviation', 'ideal_func']] = mapping_criterion.apply(pd.Series)
        #df_merge['bool_criterion'] = bool_criterion
        return df_merge


if __name__ == '__main__':
    from dataloader import DataLoader
    import numpy as np
    import os

    # Initialization
    data_dict = {'train': os.path.join('Dataset', 'train.csv'),
                 'test': os.path.join('Dataset', 'test.csv'),
                 'ideal': os.path.join('Dataset', 'ideal.csv'), }
    loader = DataLoader(db_name="data.db")
    # get train, test and ideal set
    train_df, test_df, ideal_df = loader.store_data_into_db(dict_csv=data_dict)

    # Build model
    reg_model = LinearRegressionModel(train_df, test_df, ideal_df)
    # best_ideal_dict = reg_model.choose_y_ideal(set_id=1)
    # print(best_ideal_dict)

    # evaluate with test set
    eval_df = reg_model.eval_test_set()
    print(eval_df.info())
    print(eval_df.head(20).to_string())
