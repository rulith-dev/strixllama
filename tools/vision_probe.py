#!/usr/bin/env python3
"""Send one image to a running llama-server and print what it says, to check multimodal end to end.

    python tools/vision_probe.py [port] [image]

With no image it uses the one tools/vision_stress.py generates - a red circle, a green triangle and
a blue square - so a correct answer is checkable without shipping a test asset.
"""
import base64, json, os, sys, time, urllib.request
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
IMG = sys.argv[2] if len(sys.argv) > 2 else None
def ready():
    try:
        with urllib.request.urlopen("http://127.0.0.1:%d/health" % PORT, timeout=3) as r: return r.status == 200
    except Exception: return False
t0 = time.time()
while not ready() and time.time() - t0 < 1200: time.sleep(5)
print("server ready after %.0fs" % (time.time() - t0), flush=True)
if IMG:
    b64 = base64.b64encode(open(IMG, "rb").read()).decode()
else:
    from vision_stress import png_with_shapes, SHAPES
    b64 = base64.b64encode(png_with_shapes()).decode()
    print("using the generated image: %s" % SHAPES, flush=True)
body = {"messages": [{"role": "user", "content": [
          {"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64}},
          {"type": "text", "text": "List the shapes and their colours, and read any text. Be brief."}]}],
        "max_tokens": 250, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
req = urllib.request.Request("http://127.0.0.1:%d/v1/chat/completions" % PORT,
                             data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
t1 = time.time()
try:
    with urllib.request.urlopen(req, timeout=1200) as r:
        d = json.loads(r.read().decode("utf-8", "replace"))
    print("ANSWER (%.1fs):" % (time.time() - t1), flush=True)
    print(d["choices"][0]["message"]["content"][:900], flush=True)
    print("USAGE:", json.dumps(d.get("usage", {})), flush=True)
except Exception as e:
    print("FAILED:", type(e).__name__, e, flush=True)
    if hasattr(e, "read"):
        print(e.read().decode("utf-8", "replace")[:900], flush=True)
