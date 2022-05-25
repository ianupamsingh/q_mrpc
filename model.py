import os
import shutil

import yaml
from typing import Union, Optional, List, Tuple

import numpy as np
import tensorflow as tf
import tensorflow_text
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import tensorflow_hub as hub

from config import ExperimentConfig, DataConfig
from dataset import Dataset


class BERTClassifier:
    """BERT based Text classification model"""
    def __init__(self, config: Union[ExperimentConfig, str], training: Optional[bool] = True):
        if type(config) == str:
            self.config = ExperimentConfig.from_yaml(os.path.join(config, 'params.yaml'))
        else:
            self.config = config
        self.config.labels = ['No', 'Yes']

        if training:
            self.model = None
            self.history = None
            if os.path.exists(self.config.output_dir) and os.listdir(self.config.output_dir):
                shutil.rmtree(self.config.output_dir)
                # raise FileExistsError(f"Output directory '{self.config.output_dir}' already exists and is not empty.")
            if not os.path.exists(self.config.output_dir):
                os.makedirs(self.config.output_dir)
        else:
            if not os.path.exists(self.config.output_dir):
                raise FileNotFoundError(f"Model directory '{self.config.output_dir}' does not exists.")
            if not os.path.exists(os.path.join(self.config.output_dir, 'saved_model')):
                raise FileNotFoundError(f"SavedModel not found in directory '{self.config.output_dir}'.")
            self.model = tf.keras.models.load_model(os.path.join(self.config.output_dir, 'saved_model'))

    def _create_model(self) -> tf.keras.Model:
        """Creates and compiles `tf.keras.Model` that takes `text` input and outputs `logits`"""
        # Step 1: tokenize batches of text inputs.
        self.preprocessor = hub.load(self.config.preprocessor_model)
        text_input = [tf.keras.layers.Input(shape=(), dtype=tf.string, name='str1'),
                      tf.keras.layers.Input(shape=(), dtype=tf.string, name='str2')]
        tokenize = hub.KerasLayer(self.preprocessor.tokenize, name='tokenizer')
        tokenized_inputs = [tokenize(t) for t in text_input]

        # Step 2: pack input sequences for the Transformer encoder.
        bert_pack_inputs = hub.KerasLayer(self.preprocessor.bert_pack_inputs,
                                          arguments=dict(seq_length=self.config.max_len), name='packer')
        encoder_inputs = bert_pack_inputs(tokenized_inputs)

        input_word_ids = encoder_inputs['input_word_ids']
        input_mask = encoder_inputs['input_mask']
        input_type_ids = encoder_inputs['input_type_ids']

        # Step 3: pass `encoder_inputs` to `bert_layer`
        self.bert_layer = hub.KerasLayer(self.config.pretrained_model, trainable=True, name='bert')
        pooled_output, sequence_output = self.bert_layer([input_word_ids, input_mask, input_type_ids])

        # Step 4: Add dropout
        dropout = tf.keras.layers.Dropout(self.config.dropout, name='dropout')(pooled_output)

        # Step 5: Add classifier layer
        output = tf.keras.layers.Dense(1, activation='sigmoid', name='output')(dropout)

        model = tf.keras.Model(inputs=text_input, outputs=output)

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        return model

    def _get_data(self):
        train_dataset = Dataset(self.config.train_dataset, self.config.max_len)
        train_data = train_dataset.generate()
        valid_dataset = Dataset(self.config.valid_dataset, self.config.max_len)
        valid_data = valid_dataset.generate()
        # self.config.labels = list(dataset.label_encoder.classes_)
        return train_data, valid_data

    def train(self):
        """Start training model using given `ExperimentConfig`"""
        train_data, valid_data = self._get_data()

        # create model
        self.model = self._create_model()

        # save config
        yaml.dump(self.config.as_dict(), open(os.path.join(self.config.output_dir, 'params.yaml'), 'w'),
                  default_flow_style=False)

        # define callbacks
        checkpoint = ModelCheckpoint(os.path.join(self.config.output_dir, 'saved_model'),
                                     monitor='val_accuracy', save_best_only=True, verbose=1)
        early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, verbose=1)

        # train
        self.history = self.model.fit(train_data, validation_data=valid_data, epochs=self.config.epochs,
                                      callbacks=[checkpoint, early_stopping], verbose=1)

    def predict(self, texts: Union[Tuple[str, str], List[Tuple[str, str]]]) -> List[float]:
        """Predicts class of given pair(s) of strings
            Args:
                texts: Tuple[str, str] or list[Tuple[str, str]], text to predict classes for
            Ret:
                labels: list[str], predicted class texts belong to
        """
        if type(texts) == tuple:
            texts = [texts]
        strings = list(map(list, zip(*texts)))

        predictions = self.model.predict({'str1': tf.constant(strings[0]), 'str2': tf.constant(strings[1])})
        # label_ids = np.argmax(predictions, axis=1)
        # if not self.config.labels:
        #     raise ValueError(f'Labels not defined. Make sure `params.yaml` is present in `{self.config.output_dir}`'
        #                      f' and contains `labels` as not `None`')
        # labels = [self.config.labels[label_id] for label_id in label_ids]

        return list(np.squeeze(predictions, axis=1))

    def evaluate(self) -> dict:
        """
        Evaluate model on test dataset
            Ret:
                dictionary containing model accuracy, precision, recall and f1
        """
        from sklearn.metrics import precision_recall_fscore_support
        from sklearn.metrics import accuracy_score

        test_dataset = Dataset(DataConfig(input_path='data/msr_paraphrase_test.txt', batch_size=1), 128, shuffle=False)
        test_data = test_dataset.generate()

        y_true = [each[1][0] for each in list(test_data.as_numpy_iterator())]
        y_pred = np.round(self.model.predict(test_data), 0)

        f1_scores = precision_recall_fscore_support(y_true, y_pred, average='binary')
        accuracy = accuracy_score(y_true, y_pred)

        return {'accuracy': accuracy, 'precision': f1_scores[0],
                'recall': f1_scores[1], 'f1_score': f1_scores[2]}
