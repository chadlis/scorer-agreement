import json

import httpx
import pytest

from autointerp import activation as act


@pytest.fixture
def tmp_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(act, "CACHE", tmp_path)
    monkeypatch.setattr(act, "MIN_INTERVAL_S", 0)
    monkeypatch.setattr(act, "BACKOFF_S", (0, 0, 0))
    return tmp_path


def test_cached_text_is_never_sent_again(tmp_cache):
    path = act.cache_path(100, "hello", "gpt2-small", "6-res-jb")
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"request": {}, "response": {"maxValue": 3.0}}))

    def refuse(request):
        raise AssertionError("network call on a cached text")

    with httpx.Client(transport=httpx.MockTransport(refuse)) as client:
        assert act.activation(100, "hello", client=client) == {"maxValue": 3.0}


def test_response_is_cached_with_its_request(tmp_cache):
    sent = []

    def reply(request):
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={"tokens": ["a"], "values": [0], "maxValue": 0})

    with httpx.Client(transport=httpx.MockTransport(reply)) as client:
        first = act.activation(7, "a text", client=client)
        again = act.activation(7, "a text", client=client)
    assert first == again == {"tokens": ["a"], "values": [0], "maxValue": 0}
    assert len(sent) == 1
    assert sent[0]["feature"] == {"modelId": "gpt2-small", "source": "6-res-jb", "index": "7"}
    saved = json.loads(act.cache_path(7, "a text", "gpt2-small", "6-res-jb").read_text())
    assert saved["request"]["customText"] == "a text"


def test_rate_limit_is_retried_then_cached(tmp_cache):
    ok = httpx.Response(200, json={"maxValue": 1})
    replies = iter([httpx.Response(429), httpx.Response(503), ok])
    with httpx.Client(transport=httpx.MockTransport(lambda r: next(replies))) as client:
        assert act.activation(7, "busy", client=client) == {"maxValue": 1}
    assert act.cache_path(7, "busy", "gpt2-small", "6-res-jb").exists()


def test_persistent_rate_limit_is_raised_and_not_cached(tmp_cache):
    sent = []

    def busy(request):
        sent.append(request)
        return httpx.Response(429)

    with httpx.Client(transport=httpx.MockTransport(busy)) as client:
        with pytest.raises(httpx.HTTPStatusError):
            act.activation(7, "busy", client=client)
    assert len(sent) == len(act.BACKOFF_S) + 1
    assert not act.cache_path(7, "busy", "gpt2-small", "6-res-jb").exists()


def test_client_error_is_not_retried(tmp_cache):
    sent = []

    def bad(request):
        sent.append(request)
        return httpx.Response(400)

    with httpx.Client(transport=httpx.MockTransport(bad)) as client:
        with pytest.raises(httpx.HTTPStatusError):
            act.activation(7, "bad", client=client)
    assert len(sent) == 1
    assert not act.cache_path(7, "bad", "gpt2-small", "6-res-jb").exists()
