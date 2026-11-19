"""Check Neuronpedia's activation API on feature 100 (free, no key).

Recomputes five stored windows (three positives at different levels, two zero
windows) and compares the new activations with the stored ones.
Produced data/activation_test/feature100.json on 2026-11-19. It predates
autointerp.activation, which later scripts use (cached, spaced calls).
Usage: uv run scripts/check_activation_api.py
"""

import json
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from autointerp.neuronpedia import fetch_feature, window_text  # noqa: E402 (path set above)
from autointerp.tls import ssl_context  # noqa: E402

URL = "https://www.neuronpedia.org/api/activation/new"
OUT = ROOT / "data" / "activation_test" / "feature100.json"


def main() -> None:
    acts = sorted(fetch_feature(100)["activations"], key=lambda a: -a["maxValue"])
    picks = [acts[0], acts[len(acts) // 4], acts[len(acts) // 2]]
    picks += [a for a in acts if a["maxValue"] == 0][:2]
    out = []
    with httpx.Client(verify=ssl_context(), timeout=60) as client:
        for a in picks:
            text = window_text(a["tokens"])
            body = {
                "feature": {"modelId": "gpt2-small", "source": "6-res-jb", "index": "100"},
                "customText": text,
            }
            t0 = time.time()
            r = client.post(URL, json=body)
            out.append(
                {
                    "status": r.status_code,
                    "latency_s": round(time.time() - t0, 2),
                    "rate_headers": {k: v for k, v in r.headers.items() if "rate" in k.lower()},
                    "stored_max": a["maxValue"],
                    "stored_argmax": a["maxValueTokenIndex"],
                    "stored_values": a["values"],
                    "text": text,
                    "resp": r.json(),
                }
            )
            time.sleep(1.5)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    for rec in out:
        d = rec["resp"]
        print(
            rec["status"],
            f"stored max {rec['stored_max']:.2f} at {rec['stored_argmax']}",
            f"| recomputed max {d['maxValue']:.2f} at {d['maxValueTokenIndex']}",
        )


if __name__ == "__main__":
    main()
