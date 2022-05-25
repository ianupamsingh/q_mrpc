FROM python:3.9

WORKDIR /code

ENV MODEL_DIR=/code/model

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir -r /code/requirements.txt

COPY . /code/

CMD [ "uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]