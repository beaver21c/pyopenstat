"""
OpenStat Studio v3.0 — On-Demand Desktop Web Engine Runner (MIT License)
- Runs automatic startup package & API self-diagnostics (`startup_self_check_guard`)
- Binds on-demand ONLY when explicitly launched by the user (zero persistent background daemon by default)
"""

import asyncio
import argparse
import os
import socket
import sys
import uvicorn

WORKSPACE_ROOT = os.path.dirname(os.path.abspath(__file__))
RECONSTRUCTED_DIR = os.path.join(WORKSPACE_ROOT, "reconstructed")
if RECONSTRUCTED_DIR not in sys.path:
    sys.path.insert(0, RECONSTRUCTED_DIR)

from pyjamovi.diagnostics import startup_self_check_guard
from pyjamovi.server import app


def _port_available(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, port))
            return True
        except OSError:
            return False


async def serve_multi(host: str, ports: list[int]):
    diag = startup_self_check_guard(force=False, check_pypi=False)
    compat = diag.get("compat", {})

    servers = []
    for p in ports:
        if _port_available(host, p):
            cfg = uvicorn.Config(app, host=host, port=p, log_level="info")
            servers.append((p, uvicorn.Server(cfg)))
        else:
            print(f"[Notice] Port {p} is already in use; skipping bind on {p}.")

    if not servers:
        raise RuntimeError(f"None of the requested ports {ports} are free on {host}.")

    try:
        s_ip = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s_ip.connect(("8.8.8.8", 80))
        local_lan_ip = s_ip.getsockname()[0]
        s_ip.close()
    except Exception:
        local_lan_ip = "127.0.0.1"

    print("=================================================================")
    print("  OpenStat Studio v3.0 (MIT Open-Source Statistical Engine)")
    print(f"  Self-Check : {'HEALTHY' if compat.get('healthy') else 'CHECK REQUIRED'} "
          f"(Shims active: {len(compat.get('activeShims', []))})")
    for p, _ in servers:
        print(f"  -> Local URL  : http://localhost:{p}")
        if local_lan_ip and local_lan_ip != "127.0.0.1":
            print(f"  -> Network URL: http://{local_lan_ip}:{p}")
    print("  Press Ctrl+C to stop the on-demand server.")
    print("=================================================================", flush=True)

    await asyncio.gather(*(srv.serve() for _, srv in servers))


def main():
    parser = argparse.ArgumentParser(description="Start OpenStat Studio v3.0 On-Demand Web Server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=None, help="Single port override")
    args = parser.parse_args()

    if args.port:
        ports = [args.port]
    elif "PORT" in os.environ:
        ports = [int(os.environ["PORT"])]
    else:
        ports = [8500, 8000]

    asyncio.run(serve_multi(args.host, ports))


if __name__ == "__main__":
    main()
