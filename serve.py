import os
from typing import Union, List, Tuple
from fastapi import FastAPI
import pydantic

from config import ExperimentConfig
from model import BERTClassifier

app = FastAPI()
os.environ['MODEL_DIR'] = 'C:\\Users\\Anupam Singh\\Documents\\GitHub\\q_mrpc\\model'
params_path = os.path.join(os.environ["MODEL_DIR"], 'params.yaml')
config = ExperimentConfig.from_yaml(params_path)
classifier = BERTClassifier(config, training=False)


class ClassifyRequest(pydantic.BaseModel):
    text: Union[Tuple[str, str], List[Tuple[str, str]]]


class InitRequest(pydantic.BaseModel):
    config: Union[str, ExperimentConfig]


@app.post('/classify')
def classify(request: ClassifyRequest):
    if type(request.text) != tuple and type(request.text) != list:
        return {"error": "`text` should be tuple of two strings or list[tuple of two strings]"}
    if type(request.text) == tuple:
        request.text = [request.text]

    global classifier
    if not classifier:
        return {"error": "Model is not initialized, please use /init_classifier API to initialize model"}

    predictions = classifier.predict(request.text)

    return {'paraphrase_prob': predictions}

