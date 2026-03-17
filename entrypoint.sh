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
    ffmpeg -i "$DIRECT_STREAM_URL" \
        -c:v copy -an \
        -f hls -hls_time 2 -hls_list_size 6 \
        -hls_flags delete_segments+append_list \
        /tmp/hls/stream.m3u8 &
    echo "FFmpeg HLS transcoder started for $DIRECT_STREAM_URL"
    # Wait for first segments before app starts serving
    for i in $(seq 1 15); do
        [ -f /tmp/hls/stream.m3u8 ] && break
        sleep 1
    done
fi

# Start app
exec uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1
