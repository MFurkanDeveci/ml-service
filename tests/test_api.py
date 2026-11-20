"""API katmanının testleri: gerçek HTTP istekleri, sahte sunucu üzerinden."""


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_model_info(client):
    body = client.get("/model/info").json()
    assert body["class_names"] == ["setosa", "versicolor", "virginica"]
    assert "test_accuracy" in body["metrics"]


def test_predict_valid(client, setosa):
    response = client.post("/predict", json=setosa)
    assert response.status_code == 200
    body = response.json()
    assert body["label"] == "setosa"
    assert set(body["probabilities"]) == {"setosa", "versicolor", "virginica"}
    assert "X-Process-Time-Ms" in response.headers


def test_predict_field_order_does_not_matter(client, setosa):
    reordered = dict(reversed(list(setosa.items())))
    assert client.post("/predict", json=reordered).json()["label"] == "setosa"


def test_predict_rejects_negative_value(client, setosa):
    response = client.post("/predict", json={**setosa, "petal_length": -1})
    assert response.status_code == 422


def test_predict_rejects_missing_field(client, setosa):
    setosa.pop("petal_width")
    assert client.post("/predict", json=setosa).status_code == 422


def test_predict_rejects_unknown_field(client, setosa):
    assert client.post("/predict", json={**setosa, "petal_lenght": 1.4}).status_code == 422


def test_batch(client, setosa):
    response = client.post("/predict/batch", json={"items": [setosa, setosa]})
    assert response.status_code == 200
    assert len(response.json()["predictions"]) == 2


def test_batch_too_large(client, setosa):
    response = client.post("/predict/batch", json={"items": [setosa] * 6})
    assert response.status_code == 413


def test_batch_empty(client):
    assert client.post("/predict/batch", json={"items": []}).status_code == 422
