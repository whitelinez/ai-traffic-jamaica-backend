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

# If DIRECT_STREAM_URL is an RTMP source, transcode to HLS locally
if echo "$DIRECT_STREAM_URL" | grep -q "^rtmp://"; then
    mkdir -p /tmp/hls

    # Extract host from rtmp://host:port/... and wait for Tailscale routing
    RTMP_HOST=$(echo "$DIRECT_STREAM_URL" | sed 's|rtmp://||' | cut -d: -f1)
    echo "Waiting for Tailscale route to $RTMP_HOST..."
    for i in $(seq 1 30); do
        ping -c 1 -W 2 "$RTMP_HOST" >/dev/null 2>&1 && break
        sleep 2
    done
    echo "Tailscale route ready"

    # Run FFmpeg in a background retry loop
    (while true; do
        echo "FFmpeg: starting HLS transcode from $DIRECT_STREAM_URL"
        ffmpeg -i "$DIRECT_STREAM_URL" \
            -c:v copy -an \
            -f hls -hls_time 2 -hls_list_size 6 \
            -hls_flags delete_segments+append_list \
            /tmp/hls/stream.m3u8
        echo "FFmpeg exited — retrying in 5s"
        sleep 5
    done) &

    # Wait up to 20s for first m3u8
    for i in $(seq 1 20); do
        [ -f /tmp/hls/stream.m3u8 ] && echo "HLS ready" && break
        sleep 1
    done
fi

# Start app
exec uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1
