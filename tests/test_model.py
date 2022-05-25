import pytest
import os
import yaml
import tensorflow as tf

from model import BERTClassifier


@pytest.mark.usefixtures('setup_model', 'setup_dataset', 'config_model')
class TestModel:
    def test_initialise_with_path(self, config_model):
        yaml.dump(config_model.as_dict(), open(os.path.join('test_model', 'params.yaml'), 'w'), default_flow_style=False)
        _ = BERTClassifier('test_model')

    def test_initialise_with_params(self, config_model):
        _ = BERTClassifier(config_model)

    def test_get_data(self, config_model):
        classifier = BERTClassifier(config_model)
        train_data, valid_data = classifier._get_data()
        for data in train_data.take(1):
            assert len(data[0]['str1'].numpy()) == config_model.train_dataset.batch_size, "Train batch size don't match"
            assert len(data[0]['str2'].numpy()) == config_model.train_dataset.batch_size, "Train batch size don't match"
            assert len(data[1].numpy()) == config_model.train_dataset.batch_size, "Label Train batch size don't match"
        for data in valid_data.take(1):
            assert len(data[0]['str1'].numpy()) == config_model.valid_dataset.batch_size, "Valid batch size don't match"
            assert len(data[0]['str2'].numpy()) == config_model.valid_dataset.batch_size, "Valid batch size don't match"
            assert len(data[1].numpy()) == config_model.valid_dataset.batch_size, "Label Valid batch size don't match"

    def test_create_model(self, config_model):
        classifier = BERTClassifier(config_model)
        _, _ = classifier._get_data()
        model = classifier._create_model()
        for layer in model.layers:
            assert layer.name in ['str1', 'str2', 'tokenizer', 'packer', 'bert', 'dropout', 'output'], \
                f"'{layer.name}' is not defined"
        input_typespec = tf.TensorSpec(shape=(None,), dtype=tf.string, name='str1')
        assert model.input[0].type_spec == input_typespec, f"Input typespec does not match expected `{input_typespec}`"
        input_typespec = tf.TensorSpec(shape=(None,), dtype=tf.string, name='str2')
        assert model.input[1].type_spec == input_typespec, f"Input typespec does not match expected `{input_typespec}`"
        output_typespec = tf.TensorSpec(shape=(None, len(config_model.labels)), dtype=tf.float32, name=None)
        assert model.layers[-1].output.type_spec == output_typespec, \
            f"Output typespec does not match expected '{output_typespec}'"

    @pytest.mark.skip
    def test_train(self, config_model):
        classifier = BERTClassifier(config_model)
        classifier.train()

    @pytest.mark.skip
    def test_predict(self, config_model):
        classifier = BERTClassifier(config_model, training=False)
        sample_text = ['sample text', 'sample text']
        predictions = classifier.predict(sample_text)
        assert len(predictions) == len(sample_text), "Prediction output does not match"
