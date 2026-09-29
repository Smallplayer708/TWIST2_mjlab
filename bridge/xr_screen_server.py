#!/usr/bin/env python3
"""HTTPS server for the PICO WebXR head-locked stream screen.

Why HTTPS: WebXR requires a secure context. Self-signed cert is expected —
the PICO browser will show a certificate warning, tap through it once.

Endpoints:
    /xr_screen.html     WebXR page (head-locked stream quad + 3D buttons)
    /three.min.js       three.js UMD build (served from xr_www/)
    /whep?src=NAME      same-origin proxy → go2rtc /api/whep (go2rtc sends no
                        CORS headers, and the page is a different origin)
    /cmd?name=...       same as web_control_server.py → redis bridge_cmd

Usage:
    python xr_screen_server.py [--port 1986] [--go2rtc http://IP:1984]
                               [--cert PATH] [--key PATH]

PICO side:  pico-stream "https://<PC-IP>:1986/xr_screen.html?src=orcalab"
"""
import argparse
import os
import socket
import ssl
import urllib.request
import urllib.error

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import redis

redis_client = redis.Redis(decode_responses=True)
GO2RTC_URL = "http://192.168.120.56:1984"
WWW_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "xr_www")
VALID_CMDS = {"start_pause", "refresh", "record", "replay"}


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
    protocol_version = "HTTP/1.1"

    def _send(self, code, body, ctype, extra=None):
        data = body if isinstance(body, bytes) else body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = urlparse(self.path).path
        if path.startswith("/whep"):
            self._send(400, "WHEP: use POST", "text/plain")
            return
        if path.startswith("/cmd"):
            self._send(400, "use POST", "text/plain")
            return
        name = {"": "xr_screen.html", "/": "xr_screen.html"}.get(path, path.lstrip("/"))
        fp = os.path.normpath(os.path.join(WWW_DIR, name))
        if not fp.startswith(WWW_DIR) or not os.path.isfile(fp):
            self._send(404, "not found", "text/plain")
            return
        ctype = ("text/html; charset=utf-8" if fp.endswith(".html")
                 else "application/javascript" if fp.endswith(".js")
                 else "application/octet-stream")
        with open(fp, "rb") as f:
            self._send(200, f.read(), ctype)

    def do_POST(self):
        url = urlparse(self.path)
        q = parse_qs(url.query)
        if url.path.startswith("/whep"):
            self._proxy_whep(q)
            return
        if url.path.startswith("/cmd"):
            name = (q.get("name") or [""])[0].strip().lower()
            if name not in VALID_CMDS:
                self._send(400, f"unknown cmd: {name}", "text/plain")
                return
            try:
                redis_client.set("bridge_cmd", name)
                print(f"[XR] bridge_cmd = {name}")
                self._send(200, name, "text/plain")
            except Exception as e:
                self._send(500, f"redis error: {e}", "text/plain")
            return
        self._send(404, "not found", "text/plain")

    def _proxy_whep(self, q):
        src = (q.get("src") or ["orcalab"])[0]
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        # go2rtc 1.9.x signaling: POST SDP offer → SDP answer (endpoint is
        # /api/webrtc, NOT the WHEP path which only exists in 2.x)
        req = urllib.request.Request(
            f"{GO2RTC_URL}/api/webrtc?src={urllib.parse.quote(src)}",
            data=body, method="POST",
            headers={"Content-Type": self.headers.get("Content-Type",
                                                      "application/sdp")})
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                self._send(r.status, r.read(),
                           r.headers.get("Content-Type", "application/sdp"),
                           extra={"Location": r.headers.get("Location", "")})
        except urllib.error.HTTPError as e:
            self._send(e.code, e.read(), e.headers.get("Content-Type", "text/plain"))
        except Exception as e:
            self._send(502, f"go2rtc unreachable: {e}", "text/plain")

    def log_message(self, *args):
        pass


def main():
    global GO2RTC_URL
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=1986)
    ap.add_argument("--go2rtc", default=None)
    ap.add_argument("--cert", default=None)
    ap.add_argument("--key", default=None)
    args = ap.parse_args()
    GO2RTC_URL = args.go2rtc or f"http://{local_ip()}:1984"

    cert = args.cert or os.path.expanduser("~/.local/share/xr-certs/cert.pem")
    key = args.key or os.path.expanduser("~/.local/share/xr-certs/key.pem")
    if not (os.path.isfile(cert) and os.path.isfile(key)):
        raise SystemExit(
            f"cert/key not found ({cert}, {key}). Generate with:\n"
            f"  mkdir -p ~/.local/share/xr-certs && openssl req -x509 -newkey "
            f"rsa:2048 -keyout {key} -out {cert} -days 3650 -nodes "
            f"-subj '/CN={local_ip()}' "
            f"-addext 'subjectAltName=IP:{local_ip()},DNS:localhost'")

    try:
        redis_client.ping()
    except Exception as e:
        raise SystemExit(f"Redis unreachable: {e}")

    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(cert, key)
    srv = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    srv.socket = ctx.wrap_socket(srv.socket, server_side=True)
    print(f"[XR] page:    https://{local_ip()}:{args.port}/xr_screen.html?src=orcalab")
    print(f"[XR] go2rtc:  {GO2RTC_URL}  |  redis key: bridge_cmd")
    srv.serve_forever()


if __name__ == "__main__":
    main()
