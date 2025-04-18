# test_endpoints.py
import requests
import pytest

BASE = "http://127.0.0.1:{port}/{path}"

@pytest.mark.parametrize("port,path", [
    (8000, "process"),
    (8001, "train"),
    (8002, "deploy-auto"),
    (8002, "predict"),
])
def test_endpoint_returns_json(port, path):
    url = BASE.format(port=port, path=path)
    resp = requests.get(url)
    assert resp.status_code == 200, f"{url} gave {resp.status_code}"
    # ensure it’s valid JSON and (optionally) a dict
    data = resp.json()
    assert isinstance(data, dict)
