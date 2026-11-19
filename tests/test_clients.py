from autointerp import clients


def fake(answers):
    calls = []

    def call(_clients, name, system, user):
        a = answers[len(calls)]
        calls.append(a)
        return {"ok": True, "text": a, "answer": a if a in ("0", "1") else "invalid"}

    return call, calls


def test_retry_once_on_unreadable_answer(monkeypatch):
    call, calls = fake(["[0, 1]", "1"])
    monkeypatch.setattr(clients, "call_scorer", call)
    rec = clients.score(None, "x", "s", "u")
    assert len(calls) == 2 and rec["n_attempts"] == 2 and rec["final_answer"] == "1"


def test_no_retry_on_readable_answer_and_invalid_after_two_failures(monkeypatch):
    call, calls = fake(["0"])
    monkeypatch.setattr(clients, "call_scorer", call)
    assert clients.score(None, "x", "s", "u")["final_answer"] == "0" and len(calls) == 1
    call, calls = fake(["bad", "bad"])
    monkeypatch.setattr(clients, "call_scorer", call)
    rec = clients.score(None, "x", "s", "u")
    assert rec["final_answer"] == "invalid" and len(calls) == 2


def test_api_error_is_error_not_invalid(monkeypatch):
    monkeypatch.setattr(clients, "call_scorer", lambda *a: {"ok": False, "error": "429"})
    rec = clients.score(None, "x", "s", "u")
    assert rec["final_answer"] == "error" and rec["n_attempts"] == 1


def test_backoff_retries_transient_errors_then_succeeds(monkeypatch):
    import openai

    calls = []

    def send(_c, _cfg, _s, _u, rec):
        calls.append(1)
        if len(calls) < 3:
            raise openai.APIConnectionError(request=None)
        rec.update(text="[1]", answer="1", ok=True)

    monkeypatch.setattr(clients, "_send", send)
    monkeypatch.setattr(clients, "BACKOFF_S", (0, 0, 0))
    monkeypatch.setattr(clients.time, "sleep", lambda s: None)
    rec = clients.call_scorer(None, "haiku-5.5", "s", "u")
    assert rec["ok"] and rec["answer"] == "1" and len(rec["api_errors"]) == 2
