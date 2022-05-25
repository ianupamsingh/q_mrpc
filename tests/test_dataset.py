import pytest
from dataset import Dataset


@pytest.mark.usefixtures('setup_dataset', 'config_dataset')
class TestDataset:
    def test_initialise(self, config_dataset):
        _ = Dataset(config_dataset)

    def test_generate(self, config_dataset):
        dataset = Dataset(config_dataset)
        train_data = dataset.generate()
        for data in train_data.take(1):
            assert len(data[0]['str1'].numpy()) == config_dataset.batch_size, "Train batch size don't match"
            assert len(data[0]['str2'].numpy()) == config_dataset.batch_size, "Train batch size don't match"
            assert len(data[1].numpy()) == config_dataset.batch_size, "Label Train batch size don't match"

    def test_batch_size(self, config_dataset):
        dataset = Dataset(config_dataset)
        train_data = dataset.generate()
        assert len(train_data) == (400 / config_dataset.batch_size), "Number of batch do not match"
