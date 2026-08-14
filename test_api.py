# -*- encoding: utf-8 -*-
"""冒烟测试脚本：验证 FastAPI 服务各接口是否正常（仅在服务运行时执行）。"""
import json
import urllib.request
import urllib.parse
import uuid
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

BASE = "http://127.0.0.1:8001"
PASS, FAIL = "✓", "✗"


def http_json(method, url, data=None, headers=None, timeout=120):
    req = urllib.request.Request(BASE + url, data=data, headers=headers or {}, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read()
        return resp.status, body


# 1. 首页
try:
    status, body = http_json("GET", "/", timeout=10)
    html = body.decode("utf-8", errors="replace")
    ok = status == 200 and "图片检测" in html
    print(f"{PASS if ok else FAIL} 首页 GET /  status={status} 含'图片检测'={ok}")
except Exception as e:
    print(f"{FAIL} 首页: {e}")

# 2. 模型列表
try:
    status, body = http_json("GET", "/api/models", timeout=10)
    data = json.loads(body)
    ok = data.get("code") == 0 and bool(data.get("models"))
    print(f"{PASS if ok else FAIL} /api/models code={data.get('code')} models={data.get('models')}")
except Exception as e:
    print(f"{FAIL} /api/models: {e}")

# 3. 图片检测（用 ultralytics 自带示例图）
try:
    with open(r"E:\ultralytics-main\ultralytics\assets\bus.jpg", "rb") as f:
        img_data = f.read()
    boundary = "----test" + uuid.uuid4().hex
    parts = [
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"bus.jpg\"\r\nContent-Type: image/jpeg\r\n\r\n".encode(),
        img_data,
        b"\r\n",
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"conf\"\r\n\r\n0.25\r\n".encode(),
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"iou\"\r\n\r\n0.45\r\n".encode(),
        f"--{boundary}--\r\n".encode(),
    ]
    body_data = b"".join(parts)
    status, body = http_json("POST", "/api/detect/image", data=body_data,
                             headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
                             timeout=180)
    data = json.loads(body)
    ok = data.get("code") == 0 and bool(data.get("image_base64"))
    print(f"{PASS if ok else FAIL} 图片检测 code={data.get('code')} total={data.get('total')} "
          f"类别统计={data.get('label_counts')}")
    print(f"    检测明细前2条={data.get('rows', [])[:2]}")
except Exception as e:
    print(f"{FAIL} 图片检测: {e}")

# 4. 视频上传
try:
    with open(r"E:\ultralytics-main\data\datas\sample_video.mp4", "rb") as f:
        vid_data = f.read()
    boundary = "----test" + uuid.uuid4().hex
    parts = [
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"sample.mp4\"\r\nContent-Type: video/mp4\r\n\r\n".encode(),
        vid_data,
        b"\r\n",
        f"--{boundary}--\r\n".encode(),
    ]
    body_data = b"".join(parts)
    status, body = http_json("POST", "/api/detect/video/upload", data=body_data,
                             headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
                             timeout=60)
    data = json.loads(body)
    ok = data.get("code") == 0 and bool(data.get("stream_url"))
    print(f"{PASS if ok else FAIL} 视频上传 code={data.get('code')} stream_url={data.get('stream_url')}")
except Exception as e:
    print(f"{FAIL} 视频上传: {e}")

print("冒烟测试结束")
