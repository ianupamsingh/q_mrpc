import os
from typing import Optional

from config import ExperimentConfig, DataConfig
from model import BERTClassifier


def get_experiment_config(configuration: Optional[str] = None) -> ExperimentConfig:
    configuration_ = None
    if type(configuration) == str:
        params_path = os.path.join(configuration, 'params.yaml')
        if not os.path.exists(params_path):
            raise FileNotFoundError(f"{params_path} does not exists")
        try:
            configuration_ = ExperimentConfig.from_yaml(params_path)
        except AttributeError:
            raise ValueError(f"{params_path} is not valid")
    else:
        # return default
        configuration_ = ExperimentConfig(
            preprocessor_model='https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            pretrained_model='https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2',
            output_dir='model',
            labels=['No', 'Yes'],
            max_len=512,
            learning_rate=2e-5,
            epochs=1,
            optimizer='adam',
            dropout=0.2,
            train_dataset=DataConfig(
                str1_column='#1 String',
                str2_column='#2 String',
                label_column='Quality',
                batch_size=8,
                input_path='data/train.tsv',
                seed=42),
            valid_dataset=DataConfig(
                str1_column='#1 String',
                str2_column='#2 String',
                label_column='Quality',
                batch_size=8,
                input_path='data/dev.tsv',
                seed=42))
    return configuration_


if __name__ == '__main__':
    config = get_experiment_config()
    # train
    classifier = BERTClassifier(config)
    classifier.train()

