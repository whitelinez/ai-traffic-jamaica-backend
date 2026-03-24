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

# If DIRECT_STREAM_URL is an RTMP source, set up Tailscale TCP relay + HLS transcode
if echo "$DIRECT_STREAM_URL" | grep -q "^rtmp://"; then
    mkdir -p /tmp/hls

    # Extract host:port from rtmp://host:port/path
    RTMP_HOSTPORT=$(echo "$DIRECT_STREAM_URL" | sed 's|rtmp://||' | cut -d/ -f1)
    RTMP_HOST=$(echo "$RTMP_HOSTPORT" | cut -d: -f1)
    RTMP_PORT=$(echo "$RTMP_HOSTPORT" | cut -s -d: -f2)
    RTMP_PORT=${RTMP_PORT:-1935}
    RTMP_PATH=$(echo "$DIRECT_STREAM_URL" | sed "s|rtmp://$RTMP_HOSTPORT||")

    # Wait for Tailscale SOCKS5 to be ready
    echo "Waiting for Tailscale SOCKS5 on localhost:1055..."
    for i in $(seq 1 30); do
        nc -z 127.0.0.1 1055 2>/dev/null && break
        sleep 1
    done
    echo "Tailscale SOCKS5 ready"

    # Start TCP relay: localhost:19350 → $RTMP_HOST:$RTMP_PORT via SOCKS5
    python3 /app/ts_relay.py 19350 "$RTMP_HOST" "$RTMP_PORT" 127.0.0.1 1055 &
    sleep 2

    # Rewrite DIRECT_STREAM_URL to use local relay
    LOCAL_RTMP="rtmp://127.0.0.1:19350$RTMP_PATH"
    export DIRECT_STREAM_URL="$LOCAL_RTMP"
    echo "RTMP relay active: $LOCAL_RTMP → $RTMP_HOST:$RTMP_PORT via Tailscale"

    # Run FFmpeg in a background retry loop
    # Note: "|| true" prevents set -e from killing the loop on FFmpeg failure
    (while true; do
        echo "FFmpeg: starting HLS transcode from $LOCAL_RTMP"
        ffmpeg \
            -fflags +nobuffer+discardcorrupt \
            -flags low_delay \
            -rtmp_buffer 500 \
            -i "$LOCAL_RTMP" \
            -c:v copy -an \
            -f hls \
            -hls_time 2 \
            -hls_list_size 6 \
            -hls_flags delete_segments+append_list+omit_endlist \
            -hls_allow_cache 0 \
            -loglevel warning \
            /tmp/hls/stream.m3u8 || true
        echo "FFmpeg exited — retrying in 3s"
        sleep 3
    done) &

    # Wait up to 60s for first m3u8 (2s segments need more time to appear)
    for i in $(seq 1 60); do
        [ -f /tmp/hls/stream.m3u8 ] && echo "HLS ready" && break
        sleep 1
    done
fi

# Start app
exec uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1
