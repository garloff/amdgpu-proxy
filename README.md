# GPU Inference Proxy

A lightweight process-manager + reverse proxy that sits between Open-WebUI and your
llama.cpp / stable-diffusion.cpp / whisper.cpp servers, handling lifecycle management
on a single AMD GPU.

## Features

| Feature | Description |
|---|---|
| **On-demand start** | Servers are started when the first request arrives for them |
| **VRAM eviction** | When starting a new server would exhaust VRAM (or exceed `max_concurrent`), the least-recently-used running server is stopped first |
| **Idle unload** | Servers unused for `idle_timeout` seconds are stopped automatically (saves power / VRAM) |
| **Loop detection** | If a server holds the GPU at ≥ `loop_gpu_util_threshold`% for longer than `loop_max_busy_seconds` without completing a request, it is killed |
| **Streaming support** | SSE / chunked responses (llama.cpp chat streaming) are forwarded correctly |
| **Status endpoint** | `GET /proxy/status` returns live VRAM stats and per-server state |
| **Logging** | `The output from llama-server etc can be collected and written to a logfile or forwarded to systemd's system log |

## Requirements

```
pip install fastapi uvicorn httpx pyyaml logging
```

rocm-smi must be on `$PATH` (it is if ROCm is installed correctly).

## Quick start

```bash
# Copy and adjust config
cp config.yaml /etc/gpu-inference-proxy/config.yaml
$EDITOR /etc/gpu-inference-proxy/config.yaml

# Run
python gpu_inference_proxy.py --config /etc/gpu-inference-proxy/config.yaml

# Check status
curl http://localhost:9099/proxy/status | python -m json.tool
```

## Open-WebUI integration

In Open-WebUI → Settings → Connections:

| Field | Value |
|---|---|
| OpenAI API Base URL | `http://localhost:9099/v1` |
| API Key | (anything, ignored) |

For Whisper STT: point the audio transcription URL to `http://localhost:9099/inference`.

For image generation (if using an A1111-compatible frontend):
Base URL `http://localhost:9099/sdapi/v1/`.

## Systemd unit

Save as `/etc/systemd/system/gpu-inference-proxy.service`:

```ini
[Unit]
Description=GPU Inference Proxy (llama/sd/whisper)
After=network.target

[Service]
Type=simple
User=YOUR_USER
ExecStart=/usr/bin/python3 /opt/gpu-inference-proxy/gpu_inference_proxy.py \
    --config /etc/gpu-inference-proxy/config.yaml
Restart=on-failure
RestartSec=5
# Allow enough time for a model to load on startup
TimeoutStartSec=120

[Install]
WantedBy=multi-user.target
```

```bash
systemctl daemon-reload
systemctl enable --now gpu-inference-proxy
journalctl -u gpu-inference-proxy -f
```

## Configuration reference

See `config.yaml` — all keys are commented.

Key tuning knobs:

- **`limits.max_concurrent: 2`** — if you have a small LLM loaded alongside whisper,
  you may be able to run both. Watch VRAM with `watch -n1 rocm-smi`.
- **`gpu.vram_headroom_mb`** — conservative value prevents OOM; lower it if you're sure
  of your models' VRAM footprints.
- **`limits.idle_timeout`** — set lower (e.g. 60) to unload aggressively; higher to
  reduce cold-start latency if you switch models frequently.
- **`limits.loop_max_busy_seconds`** — 300s is conservative. For long image generation
  runs set higher; for chat-only use set lower.

## Route prefix matching

The proxy uses longest-prefix matching. You can add aliases:

```yaml
route_prefixes:
  - /v1/
  - /openai/v1/   # if Open-WebUI is configured with this prefix
```

## Caveats

- Requests that arrive while a server is **starting** will wait (up to `startup_timeout`).
  They do not time out on the client side unless the client itself has a short timeout.
- If `max_concurrent: 1` and the running server has **active requests**, eviction is
  skipped and the new server fails to start (503). The client should retry.
  A future version will queue the start request until the active server goes idle.
- `rocm-smi --json` output format varies slightly across ROCm versions.
  The parser targets ROCm 5.x / 6.x. If VRAM reads as 0, run
  `rocm-smi --showmeminfo vram --json` and adjust the key names in `GpuMonitor._query()`.
