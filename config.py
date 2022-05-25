import dataclasses

from official.modeling.hyperparams.base_config import Config
from typing import Optional, List


@dataclasses.dataclass
class DataConfig(Config):
    str1_column: str = '#1 String'
    str2_column: str = '#2 String'
    label_column: str = 'Quality'
    batch_size: int = 8
    input_path: str = 'data/train.tsv'
    seed: int = 42


@dataclasses.dataclass
class ExperimentConfig(Config):
    preprocessor_model: str = 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3'
    pretrained_model: str = 'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2'
    output_dir: str = 'model'
    labels: Optional[List[str]] = None
    max_len: int = 128
    learning_rate: float = 2e-5
    epochs: int = 1
    optimizer: str = 'adam'
    dropout: float = 0.2
    train_dataset: DataConfig = DataConfig()
    valid_dataset: DataConfig = DataConfig(input_path='data/dev.tsv')
