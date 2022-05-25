import pytest
import os
import shutil
import pandas as pd

from config import DataConfig, ExperimentConfig


@pytest.fixture(scope='class')
def setup_model():
    if os.path.exists('test_model'):
        shutil.rmtree('test_model')
    os.mkdir('test_model')
    # os.mkdir('test_model/')
    yield
    shutil.rmtree('test_model')


@pytest.fixture(scope='class')
def config_model():
    config = ExperimentConfig(train_dataset=DataConfig(input_path='test_data/sample.tsv'),
                              valid_dataset=DataConfig(input_path='test_data/sample.tsv'))
    return config


@pytest.fixture(scope='class')
def setup_dataset():
    if os.path.exists('test_data'):
        shutil.rmtree('test_data')
    os.mkdir('test_data')
    temp = [['sample text', 'sample text2', 'No']] * 400
    data = pd.DataFrame(temp, columns=['#1 String', '#2 String', 'Quality'])
    data.to_csv('test_data/sample.tsv', sep='\t')
    yield
    shutil.rmtree('test_data')


@pytest.fixture(scope='class')
def config_dataset():
    config = DataConfig(input_path='test_data/sample.tsv')
    return config
