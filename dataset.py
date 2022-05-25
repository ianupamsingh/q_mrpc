from typing import Optional

from config import DataConfig
import pandas as pd
import csv
from sklearn.model_selection import StratifiedShuffleSplit
import tensorflow as tf


class Dataset:
    """Reads pd.Dataframe to create tf.data.Dataset"""
    def __init__(self, config: DataConfig, max_len: Optional[int] = 128, shuffle: bool = True):
        self.config = config
        self.max_len = max_len
        self.shuffle = shuffle
        self.df = pd.read_csv(self.config.input_path, sep='\t', quoting=csv.QUOTE_NONE)
        if self.config.str1_column not in self.df.keys() or self.config.str2_column not in self.df.keys() or\
                self.config.label_column not in self.df.keys():
            raise ValueError(f'Make sure text and label columns are present in df')

    def generate(self):
        """Generate dataset from given `pd.Dataframe`"""
        # train_df, valid_df = self.split_data()

        self.df.rename(columns={self.config.str1_column: 'str1', self.config.str2_column: 'str2',
                                self.config.label_column: 'label'}, inplace=True)

        data = tf.data.Dataset.from_tensor_slices((dict(self.df[['str1', 'str2']]), self.df['label'].values))

        if self.shuffle:
            data = data.shuffle(10000)\
                .batch(self.config.batch_size, drop_remainder=False)\
                .prefetch(tf.data.experimental.AUTOTUNE)
        else:
            data = data.batch(self.config.batch_size, drop_remainder=False)\
                .prefetch(tf.data.experimental.AUTOTUNE)

        return data

    def split_data(self):
        """Stratified splitting to create `train` and `valid` Dataframes"""
        split = StratifiedShuffleSplit(n_splits=1, test_size=self.config.split_ratio, random_state=self.config.seed)
        train_df, valid_df = None, None
        for train_index, test_index in split.split(self.df, self.df[self.config.y_column]):
            train_df = self.df.loc[train_index]
            valid_df = self.df.loc[test_index]
        return train_df, valid_df
