from fastapi.testclient import TestClient
from serve import app


client = TestClient(app)


def test_classify_single_instance():
    response = client.post("/classify", json={"text": ["blah", "blah"]})
    assert response.status_code == 200
    assert response.json() == {"is_paraphrase": ["Yes"]}


def test_classify_multiple_instance():
    response = client.post("/classify", json={"text": [["blah", "blah"], ["blah", "blah"]]})
    assert response.status_code == 200
    assert response.json() == {"is_paraphrase": ["Yes", "Yes"]}
