# -*- coding: utf-8 -*-
"""트윈 링크 — MuJoCo 셀 트윈(Windows 프로세스)과 ROS2 노드(WSL2) 사이의 TCP 프로토콜.

한 메시지 = [4바이트 big-endian 길이][JSON(utf-8)] + (JSON 에 "blob" 바이트 수가 있으면) [원시 바이트 blob].
요청 JSON 은 {"cmd": ..., ...}, 응답 JSON 은 {"ok": bool, ...}. 프레임(X/Y/D/I 640x480 float32, 약 4.9 MB)은
JSON 에 넣지 않고 np.savez_compressed 바이트를 blob 으로 붙인다.

  cmd      요청 필드                                응답
  hello    -                                        arm, n_boxes, cam_height_mm, shape, seq
  frame    -                                        name, seq, remaining(소스 팔레트 위 박스 수, 정답), blob=npz(X,Y,D,I)
  execute  pre_pick, pick, lift [m, base_link],     result(placed|misplaced|grasp_miss|ik_unreachable|path_collision|
           quat [x,y,z,w], frame_id                 descent_*|place_*), cycle_s(시뮬), place_err_mm, held, tcp_path[[t,x,y,z]...]
  state    -                                        remaining, placed, tcp[x,y,z], q[관절], busy
  reset    seed, n_boxes                            hello 와 같음

이 모듈은 numpy 만 쓴다(ROS 없이 pytest). 서버 쪽 구현은 tools/twin_server.py, 클라이언트는 여기 TwinClient.
WSL2 -> Windows 호스트 주소는 기본 게이트웨이(/proc/net/route)로 찾는다 — NAT 모드의 WSL2 에서 호스트가 그 주소다.
"""
from __future__ import annotations

import io
import json
import os
import socket
import struct
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

DEFAULT_PORT = 5555
CHANNELS = ("X", "Y", "D", "I")
_HDR = struct.Struct(">I")
MAX_MSG = 64 * 1024 * 1024


# ----------------------------------------------------------------------------- 프레임 <-> 바이트
def frame_to_blob(frame) -> bytes:
    """dict 또는 Frame(X,Y,D,I) -> npz 압축 바이트."""
    g = (lambda k: frame[k]) if isinstance(frame, dict) else (lambda k: getattr(frame, k))
    buf = io.BytesIO()
    np.savez_compressed(buf, **{k: np.ascontiguousarray(g(k), dtype=np.float32) for k in CHANNELS})
    return buf.getvalue()


def blob_to_frame_dict(blob: bytes) -> dict:
    with np.load(io.BytesIO(blob)) as z:
        return {k: np.asarray(z[k], dtype=np.float32) for k in CHANNELS}


# ----------------------------------------------------------------------------- 메시지 송수신
def send_msg(sock: socket.socket, obj: dict, blob: Optional[bytes] = None) -> None:
    obj = dict(obj)
    if blob is not None:
        obj["blob"] = len(blob)
    body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    sock.sendall(_HDR.pack(len(body)) + body + (blob or b""))


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    chunks, got = [], 0
    while got < n:
        c = sock.recv(min(n - got, 1 << 20))
        if not c:
            raise ConnectionError("peer closed the connection")
        chunks.append(c)
        got += len(c)
    return b"".join(chunks)


def recv_msg(sock: socket.socket) -> Tuple[dict, Optional[bytes]]:
    (n,) = _HDR.unpack(_recv_exact(sock, _HDR.size))
    if n > MAX_MSG:
        raise ValueError(f"message too large: {n} bytes")
    obj = json.loads(_recv_exact(sock, n).decode("utf-8"))
    blob = None
    nb = int(obj.pop("blob", 0) or 0)
    if nb:
        if nb > MAX_MSG:
            raise ValueError(f"blob too large: {nb} bytes")
        blob = _recv_exact(sock, nb)
    return obj, blob


# ----------------------------------------------------------------------------- 호스트 찾기
def default_gateway(route_file: str = "/proc/net/route") -> Optional[str]:
    """리눅스 기본 게이트웨이 IPv4. WSL2(NAT) 에서는 이것이 Windows 호스트다. 못 찾으면 None."""
    p = Path(route_file)
    if not p.exists():
        return None
    for line in p.read_text().splitlines()[1:]:
        f = line.split()
        if len(f) >= 3 and f[1] == "00000000":
            gw = int(f[2], 16)                       # little-endian hex
            return ".".join(str((gw >> (8 * i)) & 0xFF) for i in range(4))
    return None


def parse_spec(spec: str) -> Tuple[str, int]:
    """'twin://host:port' | 'twin://' | 'twin' -> (host, port). host 가 비면 기본 게이트웨이, 없으면 127.0.0.1."""
    s = spec.strip()
    if s.startswith("twin://"):
        s = s[len("twin://"):]
    elif s == "twin":
        s = ""
    host, _, port = s.partition(":")
    host = host or default_gateway() or "127.0.0.1"
    return host, int(port) if port else DEFAULT_PORT


# ----------------------------------------------------------------------------- 클라이언트
class TwinClient:
    """요청-응답 한 쌍씩. 연결이 끊기면 다음 call 에서 다시 연결한다."""

    def __init__(self, host: Optional[str] = None, port: int = DEFAULT_PORT, timeout_s: float = 120.0):
        self.host = host or default_gateway() or "127.0.0.1"
        self.port = int(port)
        self.timeout_s = float(timeout_s)
        self.sock: Optional[socket.socket] = None

    @classmethod
    def from_spec(cls, spec: str, timeout_s: float = 120.0) -> "TwinClient":
        h, p = parse_spec(spec)
        return cls(h, p, timeout_s)

    def connect(self) -> None:
        if self.sock is not None:
            return
        self.sock = socket.create_connection((self.host, self.port), timeout=self.timeout_s)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def close(self) -> None:
        if self.sock is not None:
            try:
                self.sock.close()
            finally:
                self.sock = None

    def call(self, cmd: str, **kw) -> Tuple[dict, Optional[bytes]]:
        self.connect()
        try:
            send_msg(self.sock, dict(kw, cmd=cmd))
            return recv_msg(self.sock)
        except (OSError, ConnectionError, ValueError):
            self.close()
            raise

    # 편의 메서드
    def hello(self) -> dict:
        return self.call("hello")[0]

    def frame(self):
        """(meta dict, frame dict X/Y/D/I float32 mm). 소스가 비었으면 meta['remaining'] == 0."""
        meta, blob = self.call("frame")
        if not meta.get("ok"):
            raise RuntimeError(f"twin frame failed: {meta.get('error')}")
        return meta, blob_to_frame_dict(blob)

    def execute(self, pre_pick, pick, lift, quat, frame_id: str = "base_link") -> dict:
        f = lambda v: [float(a) for a in v]  # noqa: E731
        return self.call("execute", pre_pick=f(pre_pick), pick=f(pick), lift=f(lift), quat=f(quat), frame_id=frame_id)[0]

    def state(self) -> dict:
        return self.call("state")[0]

    def reset(self, seed: Optional[int] = None, n_boxes: Optional[int] = None) -> dict:
        kw = {}
        if seed is not None:
            kw["seed"] = int(seed)
        if n_boxes is not None:
            kw["n_boxes"] = int(n_boxes)
        return self.call("reset", **kw)[0]


# ----------------------------------------------------------------------------- 서버 골격
def serve(handler, host: str = "0.0.0.0", port: int = DEFAULT_PORT, log=print, stop=None) -> None:
    """handler(obj) -> (resp_dict, blob|None). 연결마다 수신 스레드 하나, **명령 처리는 이 함수를 부른 스레드에서** 순서대로.

    렌더러(OpenGL 컨텍스트)는 만든 스레드에서만 쓸 수 있어서(다른 스레드에서 렌더하면 'Failed to make context current' 로
    빈 프레임이 나온다 — 실제로 겪음) 클라이언트 스레드는 요청을 큐에 넣고 응답을 기다리기만 한다. 트윈은 한 번에 하나만
    움직이므로 직렬 처리가 맞다. stop: callable -> True 면 종료."""
    import queue
    import threading

    q: "queue.Queue" = queue.Queue()

    def client_thread(conn, addr):
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        log(f"client {addr[0]}:{addr[1]} connected")
        try:
            while True:
                obj, _ = recv_msg(conn)
                box, ev = {}, threading.Event()
                q.put((obj, box, ev))
                ev.wait()
                send_msg(conn, box["resp"], box.get("blob"))
        except (ConnectionError, OSError):
            pass
        finally:
            conn.close()
            log(f"client {addr[0]}:{addr[1]} disconnected")

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    if os.name == "nt":
        # Windows 의 SO_REUSEADDR 은 다른 프로세스가 LISTEN 중인 포트에도 bind 를 허용한다(리눅스의 EADDRINUSE 가 안 난다).
        # 그러면 남아 있던 서버와 새 서버가 공존하고 연결은 전부 먼저 뜬 쪽으로 가서, 요청한 --arm/--seed 가 조용히 무시된다
        # (리뷰에서 실측). 배타 바인드로 두 번째 서버가 즉시 실패하게 한다.
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
    else:
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(4)
    srv.settimeout(0.5)
    log(f"twin server listening on {host}:{port}")
    alive = True

    def acceptor():
        while alive:
            try:
                conn, addr = srv.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            threading.Thread(target=client_thread, args=(conn, addr), daemon=True).start()

    threading.Thread(target=acceptor, daemon=True).start()
    try:
        while not (stop and stop()):
            try:
                obj, box, ev = q.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                resp, blob = handler(obj)
            except Exception as e:  # noqa: BLE001 — 한 요청의 예외가 서버를 죽이면 안 된다
                log(f"handler error for {obj.get('cmd')}: {e!r}")
                resp, blob = {"ok": False, "error": repr(e)}, None
            box["resp"], box["blob"] = resp, blob
            ev.set()
    finally:
        alive = False
        srv.close()
