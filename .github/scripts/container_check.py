"""Check serving and volume persistence for the exact image tested by CI."""
import json
import sys
from time import monotonic, sleep
from urllib.error import URLError
from urllib.request import Request, urlopen


def request(path, payload=None):
    body = None if payload is None else json.dumps(payload).encode()
    req = Request(f"http://127.0.0.1:8765{path}", body, {"Content-Type": "application/json"})
    with urlopen(req, timeout=5) as response:
        return response.read()


def read(path, payload=None):
    return json.loads(request(path, payload))


def wait_ready():
    deadline = monotonic() + 180
    while monotonic() < deadline:
        try:
            return read("/api/readiness")["active"]
        except (URLError, TimeoutError, ConnectionError):
            sleep(2)
    raise SystemExit("Container did not become ready within 180 seconds.")


def main():
    active = wait_ready()
    assert b"ScamShift" in request("/")
    assert b"Swagger UI" in request("/docs")
    scan = read("/api/scan", {"text": "The library closes at six today."})
    assert scan["model"] == active and 0 <= scan["score"] <= 1
    mode, state_path = sys.argv[1:]
    if mode == "save":
        result = read("/api/reviews", {"label": "genuine", "text": "The demo library shelves were painted blue."})
        assert result["saved"]
        with open(state_path, "w", encoding="utf-8") as handle:
            json.dump(active, handle)
    else:
        with open(state_path, encoding="utf-8") as handle:
            assert active == json.load(handle)
    assert read("/api/status")["review_counts"]["genuine"] == 1


if __name__ == "__main__":
    main()
