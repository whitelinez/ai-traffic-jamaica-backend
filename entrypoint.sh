#!/bin/sh
set -e

# Start Tailscale daemon
tailscaled --tun=userspace-networking --socks5-server=localhost:1055 &
TAILSCALED_PID=$!

# Authenticate and connect
if [ -n "$TAILSCALE_AUTH_KEY" ]; then
    tailscale up --authkey="$TAILSCALE_AUTH_KEY" --accept-routes --hostname=whitelinez-railway
    echo "Tailscale connected"
else
    echo "WARNING: TAILSCALE_AUTH_KEY not set — skipping Tailscale"
fi

# Start app
exec uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1
