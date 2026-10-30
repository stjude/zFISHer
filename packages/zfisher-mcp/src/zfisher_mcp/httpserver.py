"""Serving over HTTP: the server on a port of this machine, behind a token.

``HttpServer`` runs an ``MCPServer`` as a Streamable HTTP endpoint
(``http://127.0.0.1:<port>/mcp``) on a thread of its own with its own event
loop, so a program with a GUI can start and stop it. Every request must carry
``Authorization: Bearer <token>``; others get 401. The token is made fresh for
each start unless one is given.
"""
import hmac
import ipaddress
import secrets
import socket
import threading
import time

import uvicorn


def is_loopback(host):
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return host == "localhost"


class BearerToken:
    """ASGI middleware: HTTP requests without the token get 401."""

    def __init__(self, app, token):
        self.app = app
        self.expected = f"Bearer {token}".encode()

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            given = dict(scope.get("headers") or []).get(b"authorization", b"")
            if not hmac.compare_digest(given, self.expected):
                await send({"type": "http.response.start", "status": 401,
                            "headers": [(b"content-type", b"text/plain"), (b"www-authenticate", b"Bearer")]})
                await send({"type": "http.response.body", "body": b"This server needs its bearer token.\n"})
                return
        await self.app(scope, receive, send)


class HttpServer:
    """``server`` (an ``MCPServer``) served on ``host:port`` (port 0: any free
    one) at ``path``. An ``MCPServer`` serves HTTP once; build a new one to
    serve again after ``stop``."""

    def __init__(self, server, host="127.0.0.1", port=0, token=None, path="/mcp"):
        self.token = token or secrets.token_urlsafe(24)
        self.host, self.path = host, path
        app = BearerToken(server.streamable_http_app(streamable_http_path=path, host=host), self.token)
        family = socket.AF_INET6 if ":" in host else socket.AF_INET
        self._socket = socket.socket(family, socket.SOCK_STREAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._socket.bind((host, port))
        self.port = self._socket.getsockname()[1]
        self._uvicorn = uvicorn.Server(uvicorn.Config(app, log_level="warning", access_log=False, lifespan="on"))
        self._thread = None

    @property
    def url(self):
        host = f"[{self.host}]" if ":" in self.host else self.host
        return f"http://{host}:{self.port}{self.path}"

    @property
    def running(self):
        return self._thread is not None and self._thread.is_alive()

    def start(self, timeout=15):
        """Serve on a thread of its own; returns once requests are taken."""
        self._thread = threading.Thread(target=self._uvicorn.run, kwargs={"sockets": [self._socket]},
                                        name="zfisher-mcp-http", daemon=True)
        self._thread.start()
        deadline = time.monotonic() + timeout
        while not self._uvicorn.started:
            if not self._thread.is_alive() or time.monotonic() > deadline:
                raise RuntimeError("The MCP server did not start.")
            time.sleep(0.02)
        return self

    def stop(self, timeout=15):
        """Stop serving, ending the sessions of connected clients."""
        self._uvicorn.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout)
        try:
            self._socket.close()
        except OSError:
            pass

    def serve_forever(self):
        """Serve on this thread until interrupted (the ``zfisher-mcp --http`` command)."""
        self._uvicorn.run(sockets=[self._socket])
