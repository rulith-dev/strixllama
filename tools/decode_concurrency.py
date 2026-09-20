#!/usr/bin/env python3
"""Fire N decode requests at once and report per-stream and aggregate token rates.

If the GPU has idle time between the per-token graph launches of one stream (host-side work:
sampling, the PLE gather, input setup), a second stream fills those gaps and aggregate throughput
rises well above one stream's. If decode is at the memory-bandwidth wall instead, per-stream rates
halve and the aggregate barely moves. The server must have been started with -np >= N.

Usage: python tools/decode_concurrency.py [--port 8080] [--streams 2] [--n 128]
"""
import argparse
import json
import sys
import threading
import time
import urllib.request

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--streams", type=int, default=2)
    ap.add_argument("--n", type=int, default=128)
    args = ap.parse_args()

    results = [None] * args.streams
    start = threading.Barrier(args.streams)

    def run(i):
        body = {"prompt": "Write a long detailed essay about topic number %d." % i, "n_predict": args.n,
                "temperature": 0, "cache_prompt": False, "ignore_eos": True}
        req = urllib.request.Request("http://127.0.0.1:%d/completion" % args.port, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        start.wait()
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=3600) as r:
                t = json.loads(r.read().decode("utf-8", errors="replace")).get("timings", {})
            results[i] = (t.get("predicted_n") or 0, t.get("predicted_per_token_ms") or 0, time.time() - t0)
        except Exception as e:
            results[i] = ("FAIL", str(e), time.time() - t0)

    threads = [threading.Thread(target=run, args=(i,)) for i in range(args.streams)]
    t0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.time() - t0

    total = 0
    for i, r in enumerate(results):
        if r is None or r[0] == "FAIL":
            print("stream %d: FAIL %s" % (i, r[1] if r else "no result"))
            continue
        n, ms, secs = r
        total += n
        print("stream %d: %d tokens, %.2f ms/token = %.2f tok/s (%.1f s)" % (i, n, ms, 1000.0 / ms if ms else 0, secs))
    print("aggregate: %d tokens in %.1f s = %.2f tok/s over %d streams" % (total, wall, total / wall if wall else 0, args.streams))


if __name__ == "__main__":
    main()
