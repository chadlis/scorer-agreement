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
