# MRPC Paraphrase Classification

## Installation
Install required packages:
>`pip install -r requirement.txt`

Activate environement
>`source path/to/pyenv/bin/activate`

## Training
Training the model requires you to define `config.ExpermientConfig`, 
which can be done using the config.py or instantiating `config.ExpermientConfig` `from_json` or `from_yaml`

The default config is already configured if you want to test run.

After defining config:
> `python main.py`

## Serve as an API
The model can be served as an API to make inferences after training.
- Run server
  - `python serve.py`
  - This will expose following API endpoints on `http://localhost:8000`
    - POST `/classifiy`
      ```
      Request body:
      {
            "text": "tuple of two string or list(tuple of two strings)" 
      }
      ```
    
## Create new Docker image using Dockerfile
- Make sure the saved model is stored in `model` folder in project root
- Build and run docker container
```commandline
docker build -t mrpc .
docker run -d --name mrpc -p 8000:8000 mrpc
```
- Test API:
```commandline
curl --location --request POST '127.0.0.1:8000/classify' \
--header 'Content-Type: application/json' \
--data-raw '{
    "text": [
        ["PCCW '\''s chief operating officer , Mike Butcher , and Alex Arena , the chief financial officer , will report directly to Mr So .", "Current Chief Operating Officer Mike Butcher and Group Chief Financial Officer Alex Arena will report to So ."], 
        ["The company didn '\''t detail the costs of the replacement and repairs .", "But company officials expect the costs of the replacement work to run into the millions of dollars ."]] 
}'
```
