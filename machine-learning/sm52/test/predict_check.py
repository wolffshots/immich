"""Send Immich-style /predict requests to the local ML server and print the timings.

Run inside the ML container: python /s/predict_check.py
"""

import io
import json
import time
import urllib.request
import uuid

from PIL import Image, ImageDraw

URL = "http://127.0.0.1:3003/predict"


def post(entries: dict, text: str | None = None, image: bytes | None = None) -> tuple[int, str, float]:
    boundary = uuid.uuid4().hex
    parts = [("entries", None, json.dumps(entries).encode())]
    if text is not None:
        parts.append(("text", None, text.encode()))
    if image is not None:
        parts.append(("image", "test.jpg", image))
    body = b""
    for name, filename, data in parts:
        disp = f'form-data; name="{name}"' + (f'; filename="{filename}"' if filename else "")
        body += f"--{boundary}\r\nContent-Disposition: {disp}\r\n\r\n".encode() + data + b"\r\n"
    body += f"--{boundary}--\r\n".encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    start = time.monotonic()
    with urllib.request.urlopen(req, timeout=600) as resp:
        return resp.status, resp.read().decode()[:120], time.monotonic() - start


img = Image.new("RGB", (640, 480), "white")
ImageDraw.Draw(img).text((50, 200), "Hello Immich 123", fill="black")
buf = io.BytesIO()
img.save(buf, "JPEG")
jpeg = buf.getvalue()

checks = [
    ("clip text", {"clip": {"textual": {"modelName": "ViT-B-32__openai"}}}, "a photo of a dog", None),
    ("clip image", {"clip": {"visual": {"modelName": "ViT-B-32__openai"}}}, None, jpeg),
    ("faces", {"facial-recognition": {"detection": {"modelName": "buffalo_l", "options": {"minScore": 0.5}}, "recognition": {"modelName": "buffalo_l"}}}, None, jpeg),
    ("ocr", {"ocr": {"detection": {"modelName": "PP-OCRv5_mobile", "options": {"minScore": 0.5, "maxResolution": 736}}, "recognition": {"modelName": "PP-OCRv5_mobile", "options": {"minScore": 0.8}}}}, None, jpeg),
]
for label, entries, text, image in checks:
    for attempt in (1, 2):
        try:
            status, snippet, secs = post(entries, text, image)
            print(f"{label} #{attempt}: HTTP {status} in {secs:.2f}s {snippet!r}")
        except Exception as e:
            print(f"{label} #{attempt}: FAILED {e}")
