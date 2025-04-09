import os
import sqlite3
import pandas as pd
from sqlalchemy import create_engine, Column, Integer, Float, String, MetaData, Table, inspect
from sqlalchemy.orm import declarative_base, sessionmaker
from typing import List, Literal, Union, Optional, Dict, Any, Tuple
import sys
import traceback

Base = declarative_base()
current_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.dirname(current_path)
#print(root_path)


class DatabaseManager:
    """
    Base class to manage SQLite database connection and table creation.
    """

    def __init__(self, db_name="data.db"):
        self.engine = create_engine(f'sqlite:///{db_name}', echo=False)
        self.inspector = inspect(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        self.root_path = root_path
        self.metadata = MetaData()

    def create_tables(self):
        """
        Creates necessary tables in the SQLite database.
        """
        Base.metadata.create_all(self.engine)

    def get_session(self):
        """
        Returns a new database session.
        """
        return self.Session()

    def check_tb_exists(self, tb_name) -> bool:
        if tb_name in self.inspector.get_table_names():
            print(f"Table '{tb_name}' exists.")
            return True
        else:
            print(f"Table '{tb_name}' does not exist.")
            return False


class DataLoader(DatabaseManager):
    """
    Class to load training data into SQLite.
    """

    def __init__(self, db_name="data.db"):
        super().__init__(db_name)
        self.training_tb_name = "training_tb"
        self.test_tb_name = "test_tb"
        self.ideal_tb_name = "ideal_tb"

    # Save 3 dataset into database: train set, ideal set and test set
    def store_three_dataset_into_db(self, dict_csv: Dict[str, str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Reads multiple CSV files and stores them in the database.
        :param dict_csv: Dict of file paths for loading data into database.
        """
        for k, v in dict_csv.items():
            if k == 'train':
                if not self.check_tb_exists(self.training_tb_name):
                    train_csv_path = os.path.join(self.root_path, dict_csv['train'])
                    train_df = pd.read_csv(train_csv_path)
                    train_df.to_sql(name=self.training_tb_name,
                                    con=self.engine,
                                    if_exists='replace',
                                    index=False)
                else:
                    train_df = self.load_table_as_df(self.training_tb_name)
            elif k == 'test':
                if not self.check_tb_exists(self.test_tb_name):
                    test_csv_path = os.path.join(self.root_path, dict_csv['test'])
                    test_df = pd.read_csv(test_csv_path)
                    test_df.to_sql(name=self.test_tb_name,
                                   con=self.engine,
                                   if_exists='replace',
                                   index=False)
                else:
                    test_df = self.load_table_as_df(self.test_tb_name)
            elif k == 'ideal':
                if not self.check_tb_exists(self.ideal_tb_name):
                    ideal_csv_path = os.path.join(self.root_path, dict_csv['ideal'])
                    ideal_df = pd.read_csv(ideal_csv_path)
                    ideal_df.to_sql(name=self.ideal_tb_name,
                                    con=self.engine,
                                    if_exists='replace',
                                    index=False)
                else:
                    ideal_df = self.load_table_as_df(self.ideal_tb_name)
            else:
                raise ValueError("Invalid key '{}'".format(k))

        return train_df, test_df, ideal_df

    # function to save a pandas dataframe to sqlite database
    def save_df_into_db(self, df: pd.DataFrame, table_name: str):
        """
        Save a pandas DataFrame to an SQLite database using SQLAlchemy.

        Parameters:
        - df (pd.DataFrame): The DataFrame to save.
        - db_path (str): Path to the SQLite database file (e.g., 'data_analysis.db').
        - table_name (str): Name of the table to create or overwrite.
        - if_exists (str): What to do if the table already exists.
                           Options: 'fail', 'replace', 'append'. Default is 'replace'.
        """
        try:
            df.to_sql(table_name, con=self.engine, if_exists='replace', index=False)
            db_url = self.engine.url.database
            print(f"✅ DataFrame saved to table '{table_name}' in {db_url}")
        except Exception as e:
            print(f"❌ Error saving DataFrame to SQLite: {e}")

    # Load table from SQLite database. Then transform it into a pandas DataFrame.
    def load_table_as_df(self, tb_name) -> pd.DataFrame:
        """
        Load table from SQLite database. Then transform it into a pandas DataFrame.

        Parameters:
            - tb_name (str): Name of the table to load from database
        """
        try:
            query = f"SELECT * FROM {tb_name}"
            df = pd.read_sql_query(query, self.engine)
            return df
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            f_name = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            content = {'Title': '500_SERVER_ERROR',
                       'Type': str(exc_type),
                       'Detail': f'Error in loading table: {e}',
                       "Filename": f_name,
                       "Exception": str(exc_obj),
                       'line_error': exc_tb.tb_lineno,
                       'traceback': traceback.format_exc()}
            raise content

    # def load_all_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    #     train_df = self.load_table_as_df(self.training_tb_name)
    #     test_df = self.load_table_as_df(self.test_tb_name)
    #     ideal_df = self.load_table_as_df(self.ideal_tb_name)
    #     return train_df, test_df, ideal_df


if __name__ == '__main__':
    data_dict = {'train': os.path.join('Dataset', 'train.csv'),
                 'test': os.path.join('Dataset', 'test.csv'),
                 'ideal': os.path.join('Dataset', 'ideal.csv'), }
    loader = DataLoader(db_name="data.db")
    train_df, test_df, ideal_df = loader.store_three_dataset_into_db(data_dict)
    #train_df, test_df, ideal_df = loader.load_all_data()
    print("-" * 50, "Train DF", "-" * 50)
    print(train_df.head(10))
    print("-" * 50, "Test DF", "-" * 50)
    print(test_df.head(10))
    print("-" * 50, "Ideal DF", "-" * 50)
    print(ideal_df.head(10))
