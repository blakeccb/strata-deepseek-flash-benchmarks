#!/usr/bin/env python3
"""Reproduce the historical synthetic long-prompt request against an existing API."""
import hashlib
import json
import os
import sys
import time
import urllib.request

base = os.environ.get("STRATA_BASE_URL", "http://127.0.0.1:8096").rstrip("/")
key = os.environ["STRATA_API_KEY"]
model = os.environ.get("STRATA_MODEL", "deepseek-v4.1-flash")
units = int(sys.argv[1]) if len(sys.argv) > 1 else 17300
tag = sys.argv[2] if len(sys.argv) > 2 else str(time.time_ns())
unit = "\nRecord alpha: the routine monitoring system reports nominal status and no pending incidents.\n"
a, b = units // 3, 2 * units // 3
prompt = (tag + "Memorize the three secret values in this document and answer the final question.\n"
          + unit * a + "\nFirst secret: amber 4827.\n" + unit * (b-a)
          + "\nSecond secret: cobalt 7931.\n" + unit * (units-b)
          + "\nThird secret: jade 1648.\nWhat are the three secret values? Give all three.")
payload = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 128,
           "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(payload).encode(),
                             headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
t = time.monotonic()
with urllib.request.urlopen(req, timeout=21600) as response:
    result = json.load(response)
choice = result["choices"][0]
text = choice["message"].get("content") or ""
print(json.dumps({"status": "completed", "tag": tag, "units": units,
                  "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                  "elapsed_s": round(time.monotonic()-t, 2), "usage": result.get("usage"),
                  "timings": result.get("timings"), "finish": choice.get("finish_reason"),
                  "markers": {x: x in text for x in ("4827", "7931", "1648")}}, ensure_ascii=False))
