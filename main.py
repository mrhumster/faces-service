import os

import uvicorn

from app import create_app

app = create_app()


def _parse_addr(addr: str) -> tuple[str, int]:
    host, sep, port = addr.rpartition(":")
    if not sep:
        return "0.0.0.0", int(addr)
    return host or "0.0.0.0", int(port)


def main() -> None:
    addr = os.getenv("SERVER_ADDR", ":8080")
    host, port = _parse_addr(addr)
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        workers=int(os.getenv("UVICORN_WORKERS", "4")),
    )


if __name__ == "__main__":
    main()