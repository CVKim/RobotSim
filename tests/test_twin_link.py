# -*- coding: utf-8 -*-
"""트윈 링크 프로토콜 테스트 — 프레임 blob 왕복, 길이 접두 메시지, 서버 골격 + 클라이언트 (루프백, MuJoCo 불필요)."""
import socket
import threading

import numpy as np
import pytest

from robotsim_perception.twin_link import (TwinClient, blob_to_frame_dict, default_gateway, frame_to_blob,
                                           parse_spec, recv_msg, send_msg, serve)


def _frame(h=6, w=8):
    rng = np.random.default_rng(0)
    D = rng.uniform(2900, 3300, (h, w)).astype(np.float32)
    D[0, 0] = 16383.75
    return {"X": (D * 0.1).astype(np.float32), "Y": (-D * 0.2).astype(np.float32), "D": D,
            "I": rng.uniform(0, 500, (h, w)).astype(np.float32)}


def test_frame_blob_roundtrip_exact():
    f = _frame()
    g = blob_to_frame_dict(frame_to_blob(f))
    for k in ("X", "Y", "D", "I"):
        assert g[k].dtype == np.float32 and g[k].shape == f[k].shape
        np.testing.assert_array_equal(g[k], f[k])           # 무손실 (센티넬 값 포함)


def test_send_recv_with_and_without_blob():
    a, b = socket.socketpair()
    send_msg(a, {"cmd": "frame", "n": 3}, b"\x00\x01\x02")
    send_msg(a, {"ok": True})
    obj, blob = recv_msg(b)
    assert obj == {"cmd": "frame", "n": 3} and blob == b"\x00\x01\x02"     # blob 길이 필드는 벗겨진다
    obj2, blob2 = recv_msg(b)
    assert obj2 == {"ok": True} and blob2 is None
    a.close()
    b.close()


def test_recv_peer_closed_raises():
    a, b = socket.socketpair()
    a.close()
    with pytest.raises(ConnectionError):
        recv_msg(b)
    b.close()


def test_default_gateway_parses_route_file(tmp_path):
    p = tmp_path / "route"
    p.write_text("Iface Destination Gateway Flags RefCnt Use Metric Mask MTU Window IRTT\n"
                 "eth0 00000000 01501BAC 0003 0 0 0 00000000 0 0 0\n"
                 "eth0 00501BAC 00000000 0001 0 0 0 00F0FFFF 0 0 0\n")
    assert default_gateway(str(p)) == "172.27.80.1"          # little-endian hex -> 점 표기
    assert default_gateway(str(tmp_path / "missing")) is None


def test_parse_spec():
    assert parse_spec("twin://10.0.0.7:6000") == ("10.0.0.7", 6000)
    assert parse_spec("twin://10.0.0.7") == ("10.0.0.7", 5555)
    host, port = parse_spec("twin")
    assert port == 5555 and isinstance(host, str) and host


def test_client_server_roundtrip_over_loopback():
    """serve() 골격 + TwinClient: hello / frame(blob) / execute / 알 수 없는 명령 / 핸들러 예외."""
    frame = _frame()
    calls = []

    def handler(obj):
        calls.append(obj["cmd"])
        if obj["cmd"] == "hello":
            return {"ok": True, "arm": "track"}, None
        if obj["cmd"] == "frame":
            return {"ok": True, "name": "twin#1", "remaining": 5}, frame_to_blob(frame)
        if obj["cmd"] == "execute":
            assert obj["pick"] == [0.1, 0.2, 0.3] and len(obj["quat"]) == 4
            return {"ok": True, "result": "placed", "cycle_s": 7.5}, None
        if obj["cmd"] == "boom":
            raise RuntimeError("handler failure")
        return {"ok": False, "error": "unknown"}, None

    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    port = srv.getsockname()[1]
    srv.close()
    stop = threading.Event()
    th = threading.Thread(target=serve, kwargs=dict(handler=handler, host="127.0.0.1", port=port,
                                                    log=lambda s: None, stop=stop.is_set), daemon=True)
    th.start()
    c = TwinClient("127.0.0.1", port, timeout_s=10)
    try:
        for _ in range(50):                      # 서버가 listen 할 때까지
            try:
                c.connect()
                break
            except OSError:
                import time
                time.sleep(0.05)
        assert c.hello()["arm"] == "track"
        meta, f = c.frame()
        assert meta["remaining"] == 5
        np.testing.assert_array_equal(f["D"], frame["D"])
        r = c.execute((0.1, 0.2, 0.45), (0.1, 0.2, 0.3), (0.1, 0.2, 0.7), (1, 0, 0, 0))
        assert r["result"] == "placed"
        bad = c.call("nope")[0]
        assert bad["ok"] is False
        err = c.call("boom")[0]
        assert err["ok"] is False and "handler failure" in err["error"]   # 예외가 응답으로 돌아오고 연결은 살아 있다
        assert c.hello()["ok"] is True
    finally:
        c.close()
        stop.set()
        th.join(timeout=3)
    assert calls[:3] == ["hello", "frame", "execute"]
