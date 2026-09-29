#!/usr/bin/env python3
"""OrcaLab window → PICO XRoboToolkit "Remote Vision" native stream sender.

Implements the official operator-bridge protocol (reverse-engineered from
XRoboToolkit-Orin-Video-Sender, XRoboToolkit-Unity-Client and
BRobotAssistantLib sources):

  :13579  control channel (headset OperatorControlClient connects here)
          - frames: [4B big-endian body length][body]
          - body  : NetworkDataProtocol [4B LE cmdLen][cmd][4B LE dataLen][data]
          - headset sends OPEN_CAMERA (CameraRequestData) carrying the IP:port
            of its MediaDecoder server; we then CONNECT to that address and
            push [4B BE length][H.264 access unit] frames
          - headset sends CLOSE_CAMERA to stop; PING must be answered with
            PONG; we also emit PING every 2s (client idles out after 5s)
  :13580  silent raw PCM (s16le 16 kHz mono) so the headset's audio client
          has a source; --no-audio to disable

Video encoding: xwd grabs the OrcaLab window → ffmpeg libx264 baseline
yuv420p Annex-B (AUD-free, SPS/PPS repeated before every IDR). The window
is scaled to 1280x720 and duplicated into a 2560x720 side-by-side frame —
the headset's MediaDecoder is initialized with the per-source resolution
from video_source.yml (ZEDMINI = 2560x720) and ignores the stream SPS.

Usage:
    python3 orcalab_vision_sender.py [--window REGEX] [--fps 15]
        [--port 13579] [--audio-port 13580] [--bitrate 4M] [--no-audio]

PICO: XRoboToolkit → Remote Vision → source ZEDMINI → Listen → this PC's IP.
"""
import argparse
import os
import re
import socket
import struct
import subprocess
import threading
import time

FFMPEG = os.environ.get("FFMPEG", "/home/user/miniconda3/envs/stream/bin/ffmpeg")

AUD_START = b"\x00\x00\x00\x01\x09"
START_CODE = re.compile(b"\x00\x00\x01")


# ---------------------------------------------------------------- H.264 helpers
def _nal_type(part):
    m = START_CODE.search(part)
    return part[m.end()] & 0x1F if m else None


def _split_nals(au):
    return re.split(b"(?=\x00\x00\x01)", au)


def strip_aud(au):
    """Remove AUD NALs (type 9) — official senders do not emit them."""
    return b"".join(p for p in _split_nals(au) if _nal_type(p) != 9)


def has_slice(au):
    return any(1 <= (_nal_type(p) or 0) <= 5 for p in _split_nals(au))


def access_units(stream):
    """Yield H.264 access units split on AUD NALs.

    ffmpeg emits  [SPS][PPS][AUD][IDR][AUD][P][AUD][P]...
    We strip AUDs and merge parameter-set-only chunks ([SPS][PPS]) into
    the following AU, so every packet = one decodable frame — same as the
    official FFmpeg-encoder-callback senders (each AVPacket is one AU).
    """
    buf = b""
    pending = b""  # parameter sets waiting to be prepended to next frame
    while True:
        chunk = stream.read(65536)
        if not chunk:
            break
        buf += chunk
        while True:
            idx = buf.find(AUD_START, 1)
            if idx < 0:
                break
            raw, buf = buf[:idx], buf[idx:]
            au = strip_aud(raw)
            if not au:
                continue
            if has_slice(au):
                yield pending + au
                pending = b""
            else:
                pending += au
    au = strip_aud(buf)
    if au:
        yield pending + au


# ---------------------------------------------------------------- window/encoder
def find_window_id(name_regex):
    for _ in range(60):
        try:
            out = subprocess.run(
                ["xwininfo", "-root", "-tree"], capture_output=True,
                text=True, timeout=10).stdout
            for line in out.splitlines():
                if re.search(name_regex, line, re.I):
                    m = re.search(r"(0x[0-9a-f]+)", line)
                    if m:
                        return m.group(1)
        except Exception:
            pass
        time.sleep(2)
    raise SystemExit(f"[Vision] window not found: {name_regex}")


def spawn_encoder(args, wid):
    xwd_loop = (f'while true; do xwd -id {wid} -silent 2>/dev/null || break; '
                f'sleep {1.0 / args.fps:.3f}; done')
    xwd_proc = subprocess.Popen(["bash", "-c", xwd_loop], stdout=subprocess.PIPE)
    cmd = [
        FFMPEG, "-hide_banner", "-loglevel", "error",
        "-f", "xwd_pipe", "-framerate", str(args.fps), "-i", "pipe:0",
        "-filter_complex",
        "[0:v]scale=1280:720,split[a][b];[a][b]hstack[out]",
        "-map", "[out]",
        "-c:v", "libx264", "-preset", "veryfast", "-tune", "zerolatency",
        "-bf", "0", "-pix_fmt", "yuv420p",
        # baseline profile — matches the official RobotVisionTest sender
        "-profile:v", "baseline",
        # SPS/PPS before every IDR so a mid-stream decoder can start; 2s GOP
        "-g", str(args.fps * 2), "-x264-params",
        f"aud=1:repeat-headers=1:keyint={args.fps * 2}:min-keyint={args.fps}",
        "-b:v", args.bitrate, "-maxrate", args.bitrate, "-bufsize", "2M",
        "-f", "h264", "pipe:1",
    ]
    enc = subprocess.Popen(cmd, stdin=xwd_proc.stdout, stdout=subprocess.PIPE)
    xwd_proc.stdout.close()
    return xwd_proc, enc


# ---------------------------------------------------------------- protocol
def frame(body):
    return struct.pack(">I", len(body)) + body


def serialize_command(cmd, data=b""):
    cb = cmd.encode()
    return struct.pack("<i", len(cb)) + cb + struct.pack("<i", len(data)) + data


def deserialize_command(body):
    if len(body) < 8:
        return None, None
    (cmd_len,) = struct.unpack_from("<i", body, 0)
    if cmd_len < 0 or 4 + cmd_len + 4 > len(body):
        return None, None
    cmd = body[4:4 + cmd_len].decode(errors="replace")
    (data_len,) = struct.unpack_from("<i", body, 4 + cmd_len)
    off = 8 + cmd_len
    if data_len < 0 or off + data_len > len(body):
        return cmd, None
    return cmd, body[off:off + data_len]


def deserialize_camera_request(data):
    """CA FE | ver | 7x int32 LE | str8 camera | str8 ip  → dict."""
    if len(data) < 31 or data[0] != 0xCA or data[1] != 0xFE or data[2] != 1:
        return None
    o = 3
    w, h, fps, br, mvhevc, rmode, port = struct.unpack_from("<7i", data, o)
    o += 28
    def rs(o):
        n = data[o]; return data[o + 1:o + 1 + n].decode(errors="replace"), o + 1 + n
    camera, o = rs(o)
    ip, o = rs(o)
    return {"width": w, "height": h, "fps": fps, "bitrate": br,
            "port": port, "camera": camera, "ip": ip}


# ---------------------------------------------------------------- sender state
state_lock = threading.Lock()
control_conns = []      # headset OperatorControlClient sockets
video_sock = None       # our connection to the headset MediaDecoder server
last_keyframe = None    # most recent AU containing SPS (for instant start)
frames_sent = 0


def set_video(sock):
    global video_sock
    with state_lock:
        if video_sock:
            try:
                video_sock.close()
            except OSError:
                pass
        video_sock = sock


def push_au(au):
    """Send one access unit to the headset video socket (if connected)."""
    global frames_sent, last_keyframe
    pkt = struct.pack(">I", len(au)) + au
    with state_lock:
        if _nal_type(re.split(b"(?=\x00\x00\x01)", au)[0]) == 7 or \
           b"\x00\x00\x00\x01\x67" in au[:64] or b"\x00\x00\x01\x67" in au[:64]:
            last_keyframe = au
        s = video_sock
        if not s:
            return
        try:
            s.sendall(pkt)
            frames_sent += 1
            if frames_sent % (15 * 5) == 0:
                print(f"[Vision] sent {frames_sent} AUs")
        except OSError:
            print("[Vision] video connection lost")
            try:
                s.close()
            except OSError:
                pass
            set_video(None)


def au_pump(enc_stdout):
    for au in access_units(enc_stdout):
        push_au(au)


def video_pusher(ip, port, req):
    """Connect to the headset's MediaDecoder server and hand the socket over."""
    global last_keyframe
    try:
        s = socket.create_connection((ip, port), timeout=5)
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        print(f"[Vision] streaming to headset {ip}:{port} "
              f"({req['width']}x{req['height']}@{req['fps']} wanted)")
        set_video(s)
        with state_lock:
            kf, last_keyframe = last_keyframe, None
        if kf:  # instant picture instead of waiting for the next GOP
            try:
                s.sendall(struct.pack(">I", len(kf)) + kf)
            except OSError:
                pass
    except OSError as e:
        print(f"[Vision] connect to {ip}:{port} failed: {e}")


def handle_control(conn, addr):
    """Per-connection control loop: parse frames, react to OPEN/CLOSE_CAMERA."""
    global last_keyframe
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    print(f"[Vision] headset control channel connected: {addr[0]}")
    buf = b""
    try:
        while True:
            chunk = conn.recv(65536)
            if not chunk:
                break
            buf += chunk
            while len(buf) >= 4:
                (body_len,) = struct.unpack_from(">I", buf, 0)
                if body_len > 2 * 1024 * 1024:
                    raise OSError("bad frame length")
                if len(buf) < 4 + body_len:
                    break
                body, buf = buf[4:4 + body_len], buf[4 + body_len:]
                cmd, data = deserialize_command(body)
                if cmd == "OPEN_CAMERA" and data:
                    req = deserialize_camera_request(data)
                    if req and req["camera"] == "ZED":
                        threading.Thread(target=video_pusher,
                                         args=(req["ip"], req["port"], req),
                                         daemon=True).start()
                    else:
                        print(f"[Vision] OPEN_CAMERA rejected: {req}")
                elif cmd == "CLOSE_CAMERA":
                    print("[Vision] CLOSE_CAMERA")
                    set_video(None)
                # PONG and anything else: ignored
    except OSError as e:
        print(f"[Vision] control channel {addr[0]} closed: {e}")
    finally:
        conn.close()
        with state_lock:
            if conn in control_conns:
                control_conns.remove(conn)
        print(f"[Vision] headset control channel disconnected: {addr[0]}")


def control_keepalive():
    """PING every 2s — the client idles out after 5s of silence."""
    ping = frame(serialize_command("PING"))
    while True:
        time.sleep(2)
        with state_lock:
            conns, control_conns[:] = control_conns[:], []
            dead = []
            for c in conns:
                try:
                    c.sendall(ping)
                    control_conns.append(c)
                except OSError:
                    dead.append(c)
        for c in dead:
            try:
                c.close()
            except OSError:
                pass


def control_server(port):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", port))
    srv.listen(4)
    print(f"[Vision] control server on tcp:{port} "
          f"(Remote Vision → Listen → this PC's IP)")
    while True:
        conn, addr = srv.accept()
        with state_lock:
            control_conns.append(conn)
        threading.Thread(target=handle_control, args=(conn, addr),
                         daemon=True).start()


def audio_server(port):
    """Silent s16le 16 kHz mono PCM so the client's audio socket gets data."""
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", port))
    srv.listen(4)
    silence = b"\x00" * 3200  # 100 ms
    print(f"[Vision] silent audio relay on tcp:{port}")
    while True:
        conn, _ = srv.accept()
        def run(c):
            try:
                while c.sendall(silence):
                    time.sleep(0.1)
            except OSError:
                pass
        threading.Thread(target=run, args=(conn,), daemon=True).start()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--window", default="orca", help="X11 window name regex")
    ap.add_argument("--port", type=int, default=13579)
    ap.add_argument("--audio-port", type=int, default=13580)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--bitrate", default="4M")
    ap.add_argument("--no-audio", action="store_true")
    args = ap.parse_args()

    wid = find_window_id(args.window)
    print(f"[Vision] capturing window {wid} ({args.window}) "
          f"@ 2560x720 SBS, {args.fps}fps, {args.bitrate}")

    threading.Thread(target=control_server, args=(args.port,), daemon=True).start()
    threading.Thread(target=control_keepalive, daemon=True).start()
    if not args.no_audio:
        threading.Thread(target=audio_server, args=(args.audio_port,),
                         daemon=True).start()

    xwd_proc, enc = spawn_encoder(args, wid)
    try:
        au_pump(enc.stdout)
    except KeyboardInterrupt:
        pass
    finally:
        enc.kill()
        xwd_proc.kill()


if __name__ == "__main__":
    main()
