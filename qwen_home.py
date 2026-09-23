"""Serve the project guide at / without changing vLLM's API routes."""

from pathlib import Path


PAGE = (Path(__file__).with_name("qwen_home.html")).read_bytes()
HEADERS = [
    (b"content-type", b"text/html; charset=utf-8"),
    (b"cache-control", b"no-store"),
    (b"content-length", str(len(PAGE)).encode("ascii")),
]


class HomePageMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["path"] != "/":
            await self.app(scope, receive, send)
            return

        method = scope["method"]
        if method not in {"GET", "HEAD"}:
            await self.app(scope, receive, send)
            return

        await send({"type": "http.response.start", "status": 200, "headers": HEADERS})
        await send(
            {
                "type": "http.response.body",
                "body": PAGE if method == "GET" else b"",
            }
        )
