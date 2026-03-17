#!/usr/bin/env python3
"""
ts_relay.py — SOCKS5-aware TCP relay for Tailscale userspace-networking.
Listens on a local port and forwards all connections to a target host:port
through the Tailscale SOCKS5 proxy at localhost:1055.

Usage: python3 ts_relay.py <listen_port> <target_host> <target_port>
"""
import socket
import threading
import sys

import socks


def _pipe(src: socket.socket, dst: socket.socket) -> None:
    try:
        while True:
            data = src.recv(65536)
            if not data:
                break
            dst.sendall(data)
    except OSError:
        pass
    finally:
        try:
            dst.shutdown(socket.SHUT_WR)
        except OSError:
            pass


def handle(client: socket.socket, target_host: str, target_port: int,
           proxy_host: str, proxy_port: int) -> None:
    try:
        remote = socks.socksocket()
        remote.set_proxy(socks.SOCKS5, proxy_host, proxy_port)
        remote.connect((target_host, target_port))
        t1 = threading.Thread(target=_pipe, args=(client, remote), daemon=True)
        t2 = threading.Thread(target=_pipe, args=(remote, client), daemon=True)
        t1.start()
        t2.start()
        t1.join()
        t2.join()
    except Exception as e:
        print(f"[ts_relay] connection error: {e}", flush=True)
    finally:
        client.close()


def main() -> None:
    listen_port = int(sys.argv[1])
    target_host = sys.argv[2]
    target_port = int(sys.argv[3])
    proxy_host = sys.argv[4] if len(sys.argv) > 4 else "127.0.0.1"
    proxy_port = int(sys.argv[5]) if len(sys.argv) > 5 else 1055

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", listen_port))
    server.listen(32)
    print(f"[ts_relay] 127.0.0.1:{listen_port} → {target_host}:{target_port} via socks5://{proxy_host}:{proxy_port}", flush=True)

    while True:
        client, _ = server.accept()
        t = threading.Thread(target=handle, args=(client, target_host, target_port, proxy_host, proxy_port), daemon=True)
        t.start()


if __name__ == "__main__":
    main()
