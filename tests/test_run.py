import json
from types import SimpleNamespace

import pytest

from autointerp import run


def texts(n_feat=2):
    rec = {
        "positives": {"kept": [{"text": f"pos {i}"} for i in range(20)]},
        "negatives": {"kept": [{"text": f"neg {i}"} for i in range(20)]},
    }
    return dict.fromkeys(range(1, n_feat + 1), rec)


EXPL = {f: {"A": "a", "B": "b", "P": "p"} for f in (1, 2)}


def test_plan_has_every_call_once_with_valid_batch_ids():
    calls = run.plan_calls(texts(), EXPL, ["haiku-5.5", "gemini-flash"])
    assert len(calls) == 2 * 40 * 3 * 2 * 2
    ids = [c["id"] for c in calls]
    assert len(set(ids)) == len(ids)
    assert all(len(i) <= 64 and "." not in i for i in ids)
    labels = {c["text"]: c["label"] for c in calls}
    assert all(labels[i] == 1 for i in range(20)) and all(labels[i] == 0 for i in range(20, 40))


def test_cost_uses_prices_batch_discount_and_openrouter_cost():
    claude = {"model": "claude-opus-5-5", "usage": {"input_tokens": 1e6, "output_tokens": 1e5}}
    assert run.cost(claude) == pytest.approx(4.0 + 2.0)
    assert run.cost({**claude, "transport": "batch"}) == pytest.approx(3.0)
    assert run.cost({"usage": {"cost": 0.002, "prompt_tokens": 10}}) == pytest.approx(0.002)


def fake_send(answers):
    it = iter(answers)

    def send(scorer, system, user):
        text = next(it)
        return {
            "ok": True,
            "text": text,
            "answer": "invalid" if text == "?" else text,
            "model": "x",
            "usage": {"cost": 0.01},
        }

    return send


def test_unreadable_answer_is_retried_once_then_final(tmp_path):
    log = run.Log(tmp_path / "calls.jsonl")
    calls = run.plan_calls(texts(1), EXPL, ["gemini-flash"], passes=(1,))[:2]
    run.run_openrouter(log, calls, fake_send(["?", "[1]", "?", "?"]), cap=10, workers=1)
    finals = [r for r in log.records if r["final"]]
    assert [r["answer"] for r in finals] == ["1", "invalid"]
    assert len(log.records) == 4 and log.spent() == pytest.approx(0.04)


def test_cap_stops_new_calls_and_a_run_resumes(tmp_path):
    path = tmp_path / "calls.jsonl"
    calls = run.plan_calls(texts(1), EXPL, ["gemini-flash"], passes=(1,))[:5]
    run.run_openrouter(run.Log(path), calls, fake_send(["[0]"] * 5), cap=0.025, workers=1)
    assert len(run.Log(path).done()) == 3  # third call reaches 0.03 >= cap
    log = run.Log(path)
    run.run_openrouter(log, calls, fake_send(["[1]"] * 5), cap=1, workers=1)
    assert len(log.done()) == 5 and len(log.records) == 5  # done calls are not resent


class FakeBatches:
    def __init__(self, replies):
        self.replies, self.sent = replies, []

    def create(self, requests):
        self.sent.append(list(requests))
        return SimpleNamespace(id=f"b{len(self.sent)}")

    def retrieve(self, bid):
        return SimpleNamespace(processing_status="ended")

    def results(self, bid):
        for r in self.sent[-1]:
            text = self.replies[(r["custom_id"], len(self.sent))]
            msg = SimpleNamespace(
                model="claude-haiku-5-5",
                stop_reason="end_turn",
                content=[SimpleNamespace(type="text", text=text)],
                usage=SimpleNamespace(to_dict=lambda: {"input_tokens": 200, "output_tokens": 10}),
                to_dict=lambda: {},
            )
            yield SimpleNamespace(
                custom_id=r["custom_id"], result=SimpleNamespace(type="succeeded", message=msg)
            )


def test_batch_sends_once_and_retries_unreadable_in_a_second_batch(tmp_path):
    calls = run.plan_calls(texts(1), EXPL, ["haiku-5.5"], passes=(1,))[:2]
    a, b = calls[0]["id"], calls[1]["id"]
    fake = FakeBatches({(a, 1): "[1]", (b, 1): "maybe", (b, 2): "[0]"})
    log = run.Log(tmp_path / "calls.jsonl")
    run.run_batch(log, calls, fake, cap=1, est_per_call=lambda c: 0.001, poll_s=0)
    assert [len(s) for s in fake.sent] == [2, 1]
    assert {r["id"]: r["answer"] for r in log.records if r["final"]} == {a: "1", b: "0"}
    assert fake.sent[0][0]["params"]["model"] == "claude-haiku-5-5"
    assert log.spent() == pytest.approx(3 * (200 * 0.10 + 10 * 0.50) / 1e6 * 0.5)


def test_batch_refuses_to_cross_the_cap(tmp_path):
    calls = run.plan_calls(texts(1), EXPL, ["haiku-5.5"], passes=(1,))[:10]
    with pytest.raises(SystemExit, match="cap"):
        run.run_batch(
            run.Log(tmp_path / "c.jsonl"),
            calls,
            FakeBatches({}),
            cap=0.05,
            est_per_call=lambda c: 0.01,
            poll_s=0,
        )


def test_csv_has_one_row_per_finished_call(tmp_path):
    log = run.Log(tmp_path / "calls.jsonl")
    calls = run.plan_calls(texts(1), EXPL, ["gemini-flash"], passes=(1,))[:2]
    run.run_openrouter(log, calls, fake_send(["?", "[1]", "[0]"]), cap=10, workers=1)
    out = tmp_path / "calls.csv"
    run.export_csv(log, out)
    rows = out.read_text().splitlines()
    assert len(rows) == 3 and rows[0].startswith("feature,text,label")
    assert json.loads(log.path.read_text().splitlines()[0])["attempt"] == 1


def test_a_finished_record_keeps_its_text_index_and_the_reply_apart(tmp_path):
    log = run.Log(tmp_path / "calls.jsonl")
    calls = run.plan_calls(texts(1), EXPL, ["gemini-flash"], passes=(1,))
    c = next(c for c in calls if c["text"] == 7)
    run.run_openrouter(log, [c], fake_send(["[1]"]), cap=10, workers=1)
    (r,) = log.records
    assert r["text"] == 7 and r["reply"] == "[1]" and r["answer"] == "1"
    old = {**r, "text": "[1]"}  # the pilot log: the reply overwrote the index
    del old["reply"]
    assert run.call_fields(old)["text"] == 7 and run.call_fields(old)["reply"] == "[1]"


def test_spend_is_kept_per_transport_and_survives_a_reload(tmp_path):
    path = tmp_path / "calls.jsonl"
    log = run.Log(path)
    calls = run.plan_calls(texts(1), EXPL, ["gemini-flash"], passes=(1,))[:3]
    run.run_openrouter(log, calls, fake_send(["[0]"] * 3), cap=0.015, workers=1, own_spend=True)
    assert len(log.done()) == 2  # second call reaches 0.02 >= cap
    claude = {"model": "claude-opus-5-5", "transport": "batch"}
    log.add({**claude, "id": "x", "usage": {"input_tokens": 1e6, "output_tokens": 0}})
    assert log.spent("openrouter") == pytest.approx(0.02)
    assert log.spent("batch") == pytest.approx(2.0)
    assert run.Log(path).spent() == pytest.approx(2.02)


def test_a_compressed_log_is_read_when_the_plain_one_is_absent(tmp_path):
    import gzip

    path = tmp_path / "calls.jsonl"
    log = run.Log(path)
    calls = run.plan_calls(texts(1), EXPL, ["gemini-flash"], passes=(1,))[:2]
    run.run_openrouter(log, calls, fake_send(["[1]", "[0]"]), cap=10, workers=1)
    (tmp_path / "calls.jsonl.gz").write_bytes(gzip.compress(path.read_bytes()))
    path.unlink()
    again = run.Log(path)
    assert again.done() == log.done() and again.spent() == pytest.approx(log.spent())
