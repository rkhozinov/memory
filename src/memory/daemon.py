"""Memory inference daemon — holds ONNX models warm on a Unix socket.

Start:  memory-daemon start
Stop:   memory-daemon stop
Status: memory-daemon status

The CLI auto-connects when the socket exists, falling back to direct ONNX.
Round-trip overhead: ~1ms vs ~70ms for cold ONNX load.
"""

from __future__ import annotations

import json
import os
import signal
import socket
import struct
import sys
import threading
import time
from pathlib import Path

SOCKET_PATH = Path.home() / ".claude" / "tools" / "memory" / "data" / "daemon.sock"
PID_PATH = Path.home() / ".claude" / "tools" / "memory" / "data" / "daemon.pid"
# Max message size: 1MB (embeddings for large batch)
MAX_MSG = 1024 * 1024


def _send_msg(sock: socket.socket, data: bytes) -> None:
    """Send length-prefixed message."""
    sock.sendall(struct.pack(">I", len(data)) + data)


def _recv_msg(sock: socket.socket) -> bytes:
    """Receive length-prefixed message."""
    raw_len = b""
    while len(raw_len) < 4:
        chunk = sock.recv(4 - len(raw_len))
        if not chunk:
            raise ConnectionError("Socket closed")
        raw_len += chunk
    msg_len = struct.unpack(">I", raw_len)[0]
    if msg_len > MAX_MSG:
        raise ValueError(f"Message too large: {msg_len}")
    data = b""
    while len(data) < msg_len:
        chunk = sock.recv(min(msg_len - len(data), 65536))
        if not chunk:
            raise ConnectionError("Socket closed")
        data += chunk
    return data


# ---------------------------------------------------------------------------
# Client (used by embeddings.py / reranker.py)
# ---------------------------------------------------------------------------


def daemon_available() -> bool:
    """Check if daemon socket exists."""
    return SOCKET_PATH.exists()


def daemon_embed(texts: list[str]) -> list[list[float]] | None:
    """Request embeddings from daemon. Returns None if daemon unavailable."""
    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(5.0)
        sock.connect(str(SOCKET_PATH))
        req = json.dumps({"op": "embed", "texts": texts}).encode()
        _send_msg(sock, req)
        resp = json.loads(_recv_msg(sock))
        sock.close()
        if "error" in resp:
            return None
        return resp["embeddings"]
    except (ConnectionError, TimeoutError, OSError):
        return None


def daemon_rerank(query: str, candidates: list[str]) -> list[float] | None:
    """Request reranking from daemon. Returns None if daemon unavailable."""
    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(5.0)
        sock.connect(str(SOCKET_PATH))
        req = json.dumps({"op": "rerank", "query": query, "candidates": candidates}).encode()
        _send_msg(sock, req)
        resp = json.loads(_recv_msg(sock))
        sock.close()
        if "error" in resp:
            return None
        return resp["scores"]
    except (ConnectionError, TimeoutError, OSError):
        return None


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------


class InferenceDaemon:
    """Unix socket server holding ONNX models warm."""

    def __init__(self) -> None:
        self._embedding_model = None
        self._reranker_model = None
        self._running = False
        self._sock: socket.socket | None = None

    def _load_models(self) -> None:
        from .embeddings import EmbeddingModel
        from .reranker import RerankerModel

        print("Loading embedding model...", flush=True)
        t0 = time.perf_counter()
        self._embedding_model = EmbeddingModel()
        self._embedding_model._load()
        # Warm up with a dummy embed
        self._embedding_model.embed("warmup")
        embed_ms = (time.perf_counter() - t0) * 1000
        print(f"  Embedding model ready ({embed_ms:.0f}ms)", flush=True)

        print("Loading reranker model...", flush=True)
        t0 = time.perf_counter()
        self._reranker_model = RerankerModel("tinybert")
        self._reranker_model._load()
        self._reranker_model.score_pairs("warmup", ["warmup"])
        rerank_ms = (time.perf_counter() - t0) * 1000
        print(f"  Reranker model ready ({rerank_ms:.0f}ms)", flush=True)

    def _handle_request(self, data: bytes) -> bytes:
        """Process a single request, return response bytes."""
        try:
            req = json.loads(data)
            op = req.get("op")

            if op == "embed":
                texts = req["texts"]
                embeddings = self._embedding_model.embed_batch(texts)
                return json.dumps({"embeddings": embeddings.tolist()}).encode()

            elif op == "rerank":
                query = req["query"]
                candidates = req["candidates"]
                scores = self._reranker_model.score_pairs(query, candidates)
                return json.dumps({"scores": scores.tolist()}).encode()

            elif op == "health":
                return json.dumps(
                    {
                        "status": "running",
                        "pid": os.getpid(),
                        "embedding_model": "bge-small-en-v1.5",
                        "reranker_model": "tinybert",
                    }
                ).encode()

            else:
                return json.dumps({"error": f"Unknown op: {op}"}).encode()

        except Exception as e:
            return json.dumps({"error": str(e)}).encode()

    def _handle_client(self, conn: socket.socket) -> None:
        """Handle a single client connection."""
        try:
            data = _recv_msg(conn)
            resp = self._handle_request(data)
            _send_msg(conn, resp)
        except (ConnectionError, OSError, ValueError):
            pass  # Client disconnected or bad data — nothing to do
        finally:
            conn.close()

    def start(self) -> None:
        """Start the daemon."""
        # Clean up stale socket
        if SOCKET_PATH.exists():
            try:
                # Check if another daemon is running
                sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                sock.connect(str(SOCKET_PATH))
                sock.close()
                print("Daemon already running.", file=sys.stderr)
                sys.exit(1)
            except (ConnectionRefusedError, OSError):
                SOCKET_PATH.unlink()

        self._load_models()

        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._sock.bind(str(SOCKET_PATH))
        self._sock.listen(5)
        self._running = True

        # Write PID file
        PID_PATH.write_text(str(os.getpid()))

        # Handle SIGTERM/SIGINT gracefully
        def shutdown(signum, frame):
            self._running = False
            if self._sock:
                self._sock.close()

        signal.signal(signal.SIGTERM, shutdown)
        signal.signal(signal.SIGINT, shutdown)

        print(f"Daemon listening on {SOCKET_PATH} (PID {os.getpid()})", flush=True)

        while self._running:
            try:
                self._sock.settimeout(1.0)
                conn, _ = self._sock.accept()
                # Handle each client in a thread (for concurrent requests)
                t = threading.Thread(target=self._handle_client, args=(conn,), daemon=True)
                t.start()
            except TimeoutError:
                continue
            except OSError:
                break

        # Cleanup
        if SOCKET_PATH.exists():
            SOCKET_PATH.unlink()
        if PID_PATH.exists():
            PID_PATH.unlink()
        print("Daemon stopped.", flush=True)


def cmd_start(foreground: bool = False) -> None:
    """Start the daemon."""
    if not foreground:
        # Daemonize
        pid = os.fork()
        if pid > 0:
            # Parent — wait for child to load models and bind socket
            time.sleep(3.0)
            if SOCKET_PATH.exists():
                print(f"Daemon started (PID {pid})")
            else:
                print("Daemon failed to start", file=sys.stderr)
                sys.exit(1)
            return
        # Child
        os.setsid()
        # Redirect stdout/stderr to log
        log_path = Path.home() / ".claude" / "tools" / "memory" / "data" / "daemon.log"
        log_fd = open(log_path, "a")  # noqa: SIM115
        os.dup2(log_fd.fileno(), sys.stdout.fileno())
        os.dup2(log_fd.fileno(), sys.stderr.fileno())

    daemon = InferenceDaemon()
    daemon.start()


def cmd_stop() -> None:
    """Stop the daemon."""
    if PID_PATH.exists():
        pid = int(PID_PATH.read_text().strip())
        try:
            os.kill(pid, signal.SIGTERM)
            print(f"Stopped daemon (PID {pid})")
        except ProcessLookupError:
            print("Daemon not running (stale PID file)")
            PID_PATH.unlink()
            if SOCKET_PATH.exists():
                SOCKET_PATH.unlink()
    else:
        print("Daemon not running")
        if SOCKET_PATH.exists():
            SOCKET_PATH.unlink()


def cmd_status() -> None:
    """Check daemon status."""
    if not SOCKET_PATH.exists():
        print(json.dumps({"status": "stopped"}))
        return
    result = daemon_embed(["health check"])
    if result is not None:
        # Also get health info
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.settimeout(2.0)
            sock.connect(str(SOCKET_PATH))
            _send_msg(sock, json.dumps({"op": "health"}).encode())
            resp = json.loads(_recv_msg(sock))
            sock.close()
            print(json.dumps(resp, indent=2))
        except Exception:
            print(json.dumps({"status": "running", "detail": "socket exists but health failed"}))
    else:
        print(json.dumps({"status": "stopped", "detail": "socket exists but not responding"}))
        SOCKET_PATH.unlink(missing_ok=True)


def main() -> None:
    """CLI entry point for memory-daemon."""
    if len(sys.argv) < 2:
        print("Usage: memory-daemon {start|stop|status|start-fg}")
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "start":
        cmd_start(foreground=False)
    elif cmd == "start-fg":
        cmd_start(foreground=True)
    elif cmd == "stop":
        cmd_stop()
    elif cmd == "status":
        cmd_status()
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)
