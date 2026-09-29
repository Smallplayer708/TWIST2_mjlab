#!/usr/bin/env python3
"""Web control page for the TWIST2 OrcaLab bridge.

Why this exists: on PICO, the browser window holds the input focus, so
controller buttons (A/B) are consumed by the browser and never reach the
XRoboToolkit data stream (bridge can't see them). Pose tracking keeps working
in the background, but button-driven bridge control is dead.

Fix: invert the problem — since the browser owns focus, put BIG touch targets
inside the browser. This server serves a control page that embeds the go2rtc
stream and posts commands to Redis (bridge_cmd), which the bridge polls each
loop iteration.

Usage:
    python web_control_server.py [--port 1985] [--go2rtc http://IP:1984]

PICO side:  pico-stream "http://<PC-IP>:1985/"
"""
import argparse
import socket

import redis
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

redis_client = redis.Redis(decode_responses=True)
GO2RTC_URL = "http://192.168.120.56:1984"

BUTTONS = [
    ("start_pause", "▶ / ⏸ 启动·暂停", "#2e7d32"),
    ("refresh", "↻ 刷新场景", "#1565c0"),
    ("record", "⏺ 录制", "#c62828"),
    ("replay", "⏵ 回放", "#6a1b9a"),
]

PAGE = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TWIST2 Bridge 控制</title>
<style>
  html,body {{ margin:0; height:100%; background:#000; overflow:hidden;
               font-family: sans-serif; }}
  #stream {{ position:absolute; inset:0 0 96px 0; width:100%; height:calc(100% - 96px);
             border:0; }}
  #bar {{ position:absolute; left:0; right:0; bottom:0; height:96px;
          display:flex; gap:12px; padding:10px; box-sizing:border-box;
          background:#111; }}
  button {{ flex:1; font-size:26px; font-weight:bold; color:#fff;
            border:0; border-radius:12px; cursor:pointer; }}
  button:active {{ filter: brightness(1.5); }}
  #toast {{ position:absolute; top:12px; left:50%; transform:translateX(-50%);
            background:rgba(0,0,0,.75); color:#7fff7f; padding:8px 18px;
            border-radius:20px; font-size:20px; opacity:0; transition:opacity .2s; }}
</style>
</head>
<body>
<iframe id="stream" src="{go2rtc}/stream.html?src=orcalab&mode=webrtc"></iframe>
<div id="bar">
{buttons}
</div>
<div id="toast"></div>
<script>
function send(name) {{
  fetch('/cmd?name=' + name, {{method:'POST'}})
    .then(r => r.text())
    .then(t => {{
      const el = document.getElementById('toast');
      el.textContent = '已发送: ' + t;
      el.style.opacity = 1;
      setTimeout(() => el.style.opacity = 0, 1200);
    }})
    .catch(() => {{
      const el = document.getElementById('toast');
      el.textContent = '发送失败';
      el.style.opacity = 1;
      setTimeout(() => el.style.opacity = 0, 1200);
    }});
}}
</script>
</body>
</html>"""


def local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype):
        data = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path.startswith("/cmd"):
            self._send(400, "use POST", "text/plain")
            return
        buttons = "".join(
            f'<button style="background:{color}" onclick="send(\'{name}\')">{label}</button>'
            for name, label, color in BUTTONS
        )
        self._send(200, PAGE.format(go2rtc=GO2RTC_URL, buttons=buttons), "text/html; charset=utf-8")

    def do_POST(self):
        q = parse_qs(urlparse(self.path).query)
        name = (q.get("name") or [""])[0].strip().lower()
        valid = {n for n, _, _ in BUTTONS}
        if name not in valid:
            self._send(400, f"unknown cmd: {name}", "text/plain")
            return
        try:
            redis_client.set("bridge_cmd", name)
            print(f"[WebCtl] bridge_cmd = {name}")
            self._send(200, name, "text/plain")
        except Exception as e:
            self._send(500, f"redis error: {e}", "text/plain")

    def log_message(self, *args):
        pass


def main():
    global GO2RTC_URL
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=1985)
    ap.add_argument("--go2rtc", default=None, help="go2rtc base URL (default: auto local IP :1984)")
    args = ap.parse_args()
    GO2RTC_URL = args.go2rtc or f"http://{local_ip()}:1984"

    try:
        redis_client.ping()
    except Exception as e:
        raise SystemExit(f"Redis unreachable: {e}")

    srv = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    print(f"[WebCtl] control page:  http://{local_ip()}:{args.port}/")
    print(f"[WebCtl] embedded stream: {GO2RTC_URL}  |  redis key: bridge_cmd")
    srv.serve_forever()


if __name__ == "__main__":
    main()
