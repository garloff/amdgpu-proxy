#!/usr/bin/env python3
"""
gpu_inference_proxy.py — GPU-aware AI inference proxy
Manages llama.cpp, stable-diffusion.cpp, and whisper.cpp server processes
based on demand, VRAM pressure, idle timeouts, and infinite-loop detection.

Author: generated for Kurt @ S7n Cloud Services
License: Apache-2.0
"""

import asyncio
import json
import logging
import os
import signal
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Optional

import httpx
import yaml
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import Response, StreamingResponse
import uvicorn

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("gpu-proxy")


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------
@dataclass
class ServerCfg:
    name: str
    enabled: bool
    binary: str
    args: list[str]
    port: int
    host: str
    route_prefixes: list[str]
    startup_timeout: int = 60          # seconds to wait for health check
    health_path: str = "/health"       # GET → 200 means ready
    env: dict[str, str] = field(default_factory=dict)
    path_rewrite: dict[str, str] = field(default_factory=dict)  # exact path rewrites


@dataclass
class GpuCfg:
    device: int = 0
    vram_headroom_mb: int = 1024       # keep this free before starting a new server
    poll_interval: float = 2.0


@dataclass
class LimitsCfg:
    max_concurrent: int = 1            # max servers running simultaneously
    idle_timeout: int = 300            # seconds idle before unloading
    loop_gpu_util_threshold: int = 75  # % GPU util = "server is busy"
    loop_max_busy_seconds: int = 300   # kill if busy this long with no completed req
    loop_check_interval: int = 10      # how often to check for loops (seconds)
    request_timeout: int = 600         # max seconds for a proxied request


@dataclass
class LoggingCfg:
    # Where to send subprocess (llama/sd/whisper) stdout+stderr:
    #   off      — discard (default, no noise in proxy log)
    #   journald — emit via Python logging → captured by systemd/journald
    #              use: journalctl -u gpu-inference-proxy -f
    #              or:  journalctl -u gpu-inference-proxy SYSLOG_IDENTIFIER=server.llama
    #   file     — rotating log files in log_dir, one per server
    server_output: str = "off"
    log_dir: str = "/var/log/gpu-inference-proxy"
    max_bytes: int = 10 * 1024 * 1024   # per file before rotation
    backup_count: int = 3


@dataclass
class ProxyCfg:
    host: str = "0.0.0.0"
    port: int = 9099
    log_level: str = "info"


@dataclass
class Config:
    proxy: ProxyCfg
    gpu: GpuCfg
    limits: LimitsCfg
    logging_cfg: LoggingCfg
    servers: dict[str, ServerCfg]

    @staticmethod
    def load(path: str) -> "Config":
        with open(path) as f:
            raw = yaml.safe_load(f)

        proxy = ProxyCfg(**raw.get("proxy", {}))
        gpu = GpuCfg(**raw.get("gpu", {}))
        limits = LimitsCfg(**raw.get("limits", {}))
        logging_cfg = LoggingCfg(**raw.get("logging", {}))

        servers: dict[str, ServerCfg] = {}
        for name, scfg in raw.get("servers", {}).items():
            servers[name] = ServerCfg(
                name=name,
                enabled=scfg.get("enabled", True),
                binary=scfg["binary"],
                args=scfg.get("args") or [],
                port=scfg["port"],
                host=scfg.get("host", "127.0.0.1"),
                route_prefixes=scfg.get("route_prefixes") or [],
                startup_timeout=scfg.get("startup_timeout", 60),
                health_path=scfg.get("health_path", "/health"),
                env=scfg.get("env") or {},
                path_rewrite=scfg.get("path_rewrite") or {},
            )
        return Config(proxy=proxy, gpu=gpu, limits=limits, logging_cfg=logging_cfg, servers=servers)


# ---------------------------------------------------------------------------
# GPU monitor (AMD ROCm via rocm-smi)
# ---------------------------------------------------------------------------
@dataclass
class GpuStats:
    vram_total_mb: float = 0
    vram_used_mb: float = 0
    gpu_util_pct: float = 0

    @property
    def vram_free_mb(self) -> float:
        return self.vram_total_mb - self.vram_used_mb


class GpuMonitor:
    def __init__(self, device: int = 0):
        self.device = device
        self._stats = GpuStats()
        self._lock = asyncio.Lock()

    @property
    def stats(self) -> GpuStats:
        return self._stats

    async def refresh(self):
        stats = await asyncio.get_event_loop().run_in_executor(None, self._query)
        async with self._lock:
            self._stats = stats

    def _query(self) -> GpuStats:
        stats = GpuStats()
        try:
            # VRAM
            r = subprocess.run(
                ["rocm-smi", f"--device={self.device}", "--showmeminfo", "vram", "--json"],
                capture_output=True, text=True, timeout=5,
            )
            if r.returncode == 0:
                j = json.loads(r.stdout)
                card_key = f"card{self.device}"
                if card_key in j:
                    c = j[card_key]
                    total = float(c.get("VRAM Total Memory (B)", 0))
                    used = float(c.get("VRAM Total Used Memory (B)", 0))
                    stats.vram_total_mb = total / 1024 / 1024
                    stats.vram_used_mb = used / 1024 / 1024
        except Exception as e:
            log.warning("rocm-smi VRAM query failed: %s", e)

        try:
            # GPU utilization
            r = subprocess.run(
                ["rocm-smi", f"--device={self.device}", "--showuse", "--json"],
                capture_output=True, text=True, timeout=5,
            )
            if r.returncode == 0:
                j = json.loads(r.stdout)
                card_key = f"card{self.device}"
                if card_key in j:
                    util_str = j[card_key].get("GPU use (%)", "0")
                    stats.gpu_util_pct = float(str(util_str).replace("%", "").strip())
        except Exception as e:
            log.warning("rocm-smi util query failed: %s", e)

        return stats


# ---------------------------------------------------------------------------
# Per-server state machine
# ---------------------------------------------------------------------------
class State(Enum):
    STOPPED = auto()
    STARTING = auto()
    RUNNING = auto()
    STOPPING = auto()


class ManagedServer:
    """Wraps a single inference server process."""

    def __init__(self, cfg: ServerCfg, limits: LimitsCfg, logging_cfg: LoggingCfg):
        self.cfg = cfg
        self.limits = limits
        self.logging_cfg = logging_cfg
        self._state = State.STOPPED
        self._proc: Optional[subprocess.Popen] = None
        self._ready_event = asyncio.Event()
        self._lock = asyncio.Lock()
        self._last_request_at: float = 0.0
        self._active_requests: int = 0
        # loop detection: (timestamp, util) ring buffer
        self._util_history: deque[tuple[float, float]] = deque(maxlen=100)
        self._last_completed_at: float = time.monotonic()

    @property
    def state(self) -> State:
        return self._state

    @property
    def base_url(self) -> str:
        return f"http://{self.cfg.host}:{self.cfg.port}"

    @property
    def is_idle(self) -> bool:
        if self._active_requests > 0:
            return False
        idle_for = time.monotonic() - self._last_request_at
        return idle_for > self.limits.idle_timeout

    def record_gpu_util(self, util: float):
        self._util_history.append((time.monotonic(), util))

    def mark_request_started(self):
        self._last_request_at = time.monotonic()
        self._active_requests += 1

    def mark_request_done(self):
        self._active_requests = max(0, self._active_requests - 1)
        self._last_completed_at = time.monotonic()

    def check_loop(self) -> bool:
        """Return True if server looks stuck (high GPU util, no completed request recently)."""
        if self._active_requests == 0:
            return False
        if not self._util_history:
            return False
        now = time.monotonic()
        window_start = now - self.limits.loop_max_busy_seconds
        recent = [(t, u) for t, u in self._util_history if t >= window_start]
        if not recent:
            return False
        avg_util = sum(u for _, u in recent) / len(recent)
        busy_long_enough = (now - window_start) >= self.limits.loop_max_busy_seconds
        no_recent_completion = (now - self._last_completed_at) >= self.limits.loop_max_busy_seconds
        if busy_long_enough and no_recent_completion and avg_util >= self.limits.loop_gpu_util_threshold:
            log.warning(
                "Loop detected on %s: avg GPU util %.1f%% for %ds with no completion",
                self.cfg.name, avg_util, self.limits.loop_max_busy_seconds,
            )
            return True
        return False

    async def start(self) -> bool:
        """Start the server process. Returns True if successfully started."""
        async with self._lock:
            if self._state in (State.RUNNING, State.STARTING):
                return True
            self._state = State.STARTING
            self._ready_event.clear()

        env = os.environ.copy()
        env.update(self.cfg.env)
        cmd = [self.cfg.binary] + self.cfg.args

        log.info("Starting %s: %s", self.cfg.name, " ".join(cmd))
        try:
            self._proc = subprocess.Popen(
                cmd, env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            )
            # Drain stdout in background so the pipe never fills
            asyncio.get_event_loop().run_in_executor(None, self._drain_stdout)
        except FileNotFoundError:
            log.error("Binary not found for %s: %s", self.cfg.name, self.cfg.binary)
            self._state = State.STOPPED
            return False

        # Poll health endpoint until ready or timeout
        deadline = time.monotonic() + self.cfg.startup_timeout
        async with httpx.AsyncClient() as client:
            while time.monotonic() < deadline:
                if self._proc.poll() is not None:
                    log.error("%s exited during startup (rc=%d)", self.cfg.name, self._proc.returncode)
                    self._state = State.STOPPED
                    return False
                try:
                    r = await client.get(
                        f"{self.base_url}{self.cfg.health_path}", timeout=2.0
                    )
                    if r.status_code < 500:
                        log.info("%s is ready (status %d)", self.cfg.name, r.status_code)
                        self._state = State.RUNNING
                        self._last_request_at = time.monotonic()
                        self._last_completed_at = time.monotonic()
                        self._ready_event.set()
                        return True
                except Exception:
                    pass
                await asyncio.sleep(0.5)

        log.error("%s failed to become ready within %ds", self.cfg.name, self.cfg.startup_timeout)
        await self.stop()
        return False

    async def stop(self):
        async with self._lock:
            if self._state == State.STOPPED:
                return
            self._state = State.STOPPING
            self._ready_event.clear()

        proc = self._proc
        if proc and proc.poll() is None:
            log.info("Stopping %s (pid=%d)", self.cfg.name, proc.pid)
            try:
                proc.send_signal(signal.SIGTERM)
                for _ in range(50):
                    if proc.poll() is not None:
                        break
                    await asyncio.sleep(0.1)
                else:
                    log.warning("SIGKILL %s (pid=%d)", self.cfg.name, proc.pid)
                    proc.kill()
                    proc.wait()
            except ProcessLookupError:
                pass
        self._proc = None
        self._state = State.STOPPED
        log.info("%s stopped", self.cfg.name)

    def _build_server_logger(self) -> logging.Logger:
        """Create (or retrieve) a logger for this server's output according to config."""
        import logging.handlers
        logger = logging.getLogger(f"server.{self.cfg.name}")
        # Avoid adding duplicate handlers if server is restarted
        if logger.handlers:
            return logger
        mode = self.logging_cfg.server_output.lower()
        if mode == "off":
            logger.addHandler(logging.NullHandler())
            logger.propagate = False
        elif mode == "journald":
            # Emit at INFO so systemd captures it alongside the proxy's own log.
            # journalctl -u gpu-inference-proxy -f  shows everything.
            # Filter to just one server:
            #   journalctl -u gpu-inference-proxy -f SYSLOG_IDENTIFIER=server.llama
            # (Python's logging module sets SYSLOG_IDENTIFIER to the logger name
            #  when output reaches systemd-journald via stderr.)
            handler = logging.StreamHandler(stream=None)  # None → stderr
            handler.setFormatter(logging.Formatter(
                f"[%(name)s] %(message)s"
            ))
            logger.addHandler(handler)
            logger.setLevel(logging.DEBUG)
            logger.propagate = False
        elif mode == "file":
            import logging.handlers as lh
            log_dir = Path(self.logging_cfg.log_dir)
            try:
                log_dir.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                log.warning("Cannot create log dir %s: %s — falling back to stderr", log_dir, e)
                logger.addHandler(logging.StreamHandler())
                logger.propagate = False
                return logger
            log_path = log_dir / f"{self.cfg.name}.log"
            handler = lh.RotatingFileHandler(
                log_path,
                maxBytes=self.logging_cfg.max_bytes,
                backupCount=self.logging_cfg.backup_count,
                encoding="utf-8",
            )
            handler.setFormatter(logging.Formatter("%(asctime)s %(message)s",
                                                   datefmt="%Y-%m-%d %H:%M:%S"))
            logger.addHandler(handler)
            logger.setLevel(logging.DEBUG)
            logger.propagate = False
            log.info("%s output → %s", self.cfg.name, log_path)
        else:
            log.warning("Unknown server_output mode %r — using 'off'", mode)
            logger.addHandler(logging.NullHandler())
            logger.propagate = False
        return logger

    def _drain_stdout(self):
        """Read subprocess stdout/stderr in a background thread and route per config."""
        if not (self._proc and self._proc.stdout):
            return
        server_logger = self._build_server_logger()
        for raw_line in self._proc.stdout:
            line = raw_line.decode(errors="replace").rstrip()
            server_logger.info(line)

    async def wait_ready(self, timeout: float = 120) -> bool:
        try:
            await asyncio.wait_for(self._ready_event.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False


# ---------------------------------------------------------------------------
# Process manager — orchestrates all servers
# ---------------------------------------------------------------------------
class ProcessManager:
    def __init__(self, config: Config):
        self.config = config
        self.gpu = GpuMonitor(config.gpu.device)
        self.servers: dict[str, ManagedServer] = {
            name: ManagedServer(scfg, config.limits, config.logging_cfg)
            for name, scfg in config.servers.items()
            if scfg.enabled
        }
        # Route prefix → server name
        self._route_map: dict[str, str] = {}
        for name, scfg in config.servers.items():
            if scfg.enabled:
                for prefix in scfg.route_prefixes:
                    self._route_map[prefix] = name
        self._bg_tasks: list[asyncio.Task] = []

    def resolve(self, path: str) -> Optional[ManagedServer]:
        """Find which server should handle this path."""
        # Longest-prefix match
        best: Optional[tuple[str, str]] = None
        for prefix, name in self._route_map.items():
            if path.startswith(prefix):
                if best is None or len(prefix) > len(best[0]):
                    best = (prefix, name)
        if best:
            return self.servers.get(best[1])
        return None

    async def ensure_running(self, server: ManagedServer) -> bool:
        """Ensure server is running, evicting others if needed."""
        if server.state == State.RUNNING:
            return True
        if server.state == State.STARTING:
            return await server.wait_ready(server.cfg.startup_timeout)

        # Need to start it — check VRAM and concurrency limits
        await self.gpu.refresh()
        stats = self.gpu.stats
        running = [s for s in self.servers.values() if s.state == State.RUNNING]

        # Evict if over concurrent limit or VRAM headroom is insufficient
        while running and (
            len(running) >= self.config.limits.max_concurrent
            or stats.vram_free_mb < self.config.gpu.vram_headroom_mb
        ):
            # Evict the server with the oldest last_request time (LRU)
            victim = min(running, key=lambda s: s._last_request_at)
            if victim._active_requests > 0:
                log.warning(
                    "Need to evict %s but it has %d active requests — waiting is better",
                    victim.cfg.name, victim._active_requests,
                )
                # Don't evict a server with active requests; just fail-fast
                # (caller will queue or reject)
                break
            log.info("Evicting %s to free resources", victim.cfg.name)
            await victim.stop()
            running = [s for s in self.servers.values() if s.state == State.RUNNING]
            await self.gpu.refresh()
            stats = self.gpu.stats

        return await server.start()

    async def start_background_tasks(self):
        self._bg_tasks.append(asyncio.create_task(self._gpu_poll_loop()))
        self._bg_tasks.append(asyncio.create_task(self._idle_reaper_loop()))
        self._bg_tasks.append(asyncio.create_task(self._loop_detector_loop()))

    async def stop_all(self):
        for task in self._bg_tasks:
            task.cancel()
        for server in self.servers.values():
            await server.stop()

    async def _gpu_poll_loop(self):
        while True:
            try:
                await self.gpu.refresh()
                stats = self.gpu.stats
                for server in self.servers.values():
                    if server.state == State.RUNNING:
                        server.record_gpu_util(stats.gpu_util_pct)
            except Exception as e:
                log.debug("GPU poll error: %s", e)
            await asyncio.sleep(self.config.gpu.poll_interval)

    async def _idle_reaper_loop(self):
        while True:
            await asyncio.sleep(30)
            for server in self.servers.values():
                if server.state == State.RUNNING and server.is_idle:
                    log.info(
                        "%s idle for >%ds — unloading",
                        server.cfg.name, self.config.limits.idle_timeout,
                    )
                    await server.stop()

    async def _loop_detector_loop(self):
        while True:
            await asyncio.sleep(self.config.limits.loop_check_interval)
            for server in self.servers.values():
                if server.state == State.RUNNING and server.check_loop():
                    log.error(
                        "Killing %s due to suspected infinite loop", server.cfg.name
                    )
                    await server.stop()


# ---------------------------------------------------------------------------
# FastAPI proxy application
# ---------------------------------------------------------------------------
def build_app(manager: ProcessManager) -> FastAPI:
    app = FastAPI(title="GPU Inference Proxy", version="1.0.0")

    @app.on_event("startup")
    async def startup():
        await manager.start_background_tasks()
        log.info("GPU Inference Proxy started")
        log.info("Routes: %s", manager._route_map)

    @app.on_event("shutdown")
    async def shutdown():
        log.info("Shutting down — stopping all servers")
        await manager.stop_all()

    @app.get("/proxy/status")
    async def status():
        await manager.gpu.refresh()
        stats = manager.gpu.stats
        return {
            "gpu": {
                "vram_total_mb": round(stats.vram_total_mb, 1),
                "vram_used_mb": round(stats.vram_used_mb, 1),
                "vram_free_mb": round(stats.vram_free_mb, 1),
                "gpu_util_pct": stats.gpu_util_pct,
            },
            "servers": {
                name: {
                    "state": srv.state.name,
                    "active_requests": srv._active_requests,
                    "idle_seconds": round(time.monotonic() - srv._last_request_at, 1)
                    if srv._last_request_at > 0 else None,
                }
                for name, srv in manager.servers.items()
            },
        }

    # -----------------------------------------------------------------------
    # OpenAI-compatible image generation translation
    # Open-WebUI calls POST /images/generations (or /v1/images/generations)
    # when the LLM tool invokes image generation.
    # stable-diffusion.cpp speaks A1111 (/sdapi/v1/txt2img), so we translate.
    # -----------------------------------------------------------------------
    async def _openai_images_handler(request: Request):
        # Find the sd server — look for one with an sdapi or txt2img prefix
        sd_server = None
        for srv in manager.servers.values():
            if any(p.startswith("/sdapi") or "txt2img" in p
                   for p in srv.cfg.route_prefixes):
                sd_server = srv
                break
        if sd_server is None:
            raise HTTPException(status_code=404, detail="No image generation backend configured")

        ok = await manager.ensure_running(sd_server)
        if not ok:
            raise HTTPException(status_code=503, detail=f"Failed to start {sd_server.cfg.name}")

        # Parse OpenAI request
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid JSON body")

        prompt = body.get("prompt", "")
        n = body.get("n", 1)
        size = body.get("size", "512x512")
        try:
            width, height = (int(x) for x in size.lower().split("x"))
        except Exception:
            width, height = 512, 512

        # Translate to A1111 format
        a1111_payload = {
            "prompt": prompt,
            "negative_prompt": body.get("negative_prompt", ""),
            "width": width,
            "height": height,
            "batch_size": n,
            "steps": body.get("steps", 20),
            "cfg_scale": body.get("cfg_scale", 7),
            "sampler_name": body.get("sampler_name", "Euler a"),
        }

        target_url = sd_server.base_url + "/sdapi/v1/txt2img"
        log.info("OpenAI images → A1111 txt2img: prompt=%r size=%dx%d n=%d",
                 prompt[:60], width, height, n)

        sd_server.mark_request_started()
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(
                connect=10.0, read=manager.config.limits.request_timeout,
                write=30.0, pool=5.0,
            )) as client:
                resp = await client.post(target_url, json=a1111_payload)
                resp.raise_for_status()
                a1111_resp = resp.json()
        except httpx.HTTPStatusError as e:
            sd_server.mark_request_done()
            raise HTTPException(status_code=e.response.status_code,
                                detail=f"sd-server error: {e.response.text[:200]}")
        except Exception as e:
            sd_server.mark_request_done()
            log.exception("Image generation error: %s", e)
            raise HTTPException(status_code=502, detail=str(e))
        finally:
            sd_server.mark_request_done()

        # Translate A1111 response → OpenAI response
        # A1111 returns {"images": ["base64...", ...]}
        images = a1111_resp.get("images", [])
        openai_data = [{"b64_json": img} for img in images]

        return {
            "created": int(time.time()),
            "data": openai_data,
        }

    @app.post("/images/generations")
    async def images_generations_v0(request: Request):
        return await _openai_images_handler(request)

    @app.post("/v1/images/generations")
    async def images_generations_v1(request: Request):
        return await _openai_images_handler(request)

    @app.api_route(
        "/{path:path}",
        methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"],
    )
    async def proxy(path: str, request: Request):
        full_path = "/" + path
        qs = request.url.query
        if qs:
            full_path += "?" + qs

        server = manager.resolve("/" + path)
        if server is None:
            raise HTTPException(status_code=404, detail=f"No backend configured for /{path}")

        log.info("→ %s %s → %s", request.method, full_path, server.cfg.name)

        ok = await manager.ensure_running(server)
        if not ok:
            raise HTTPException(status_code=503, detail=f"Failed to start {server.cfg.name}")

        # Apply exact-path rewrites (e.g. /v1/audio/transcriptions → /inference)
        request_path = "/" + path
        rewritten_path = server.cfg.path_rewrite.get(request_path, request_path)
        if rewritten_path != request_path:
            log.info("  path rewrite: %s → %s", request_path, rewritten_path)
        full_path_rewritten = rewritten_path
        if qs:
            full_path_rewritten += "?" + qs

        # Forward request
        target_url = server.base_url + full_path_rewritten
        headers = {
            k: v for k, v in request.headers.items()
            if k.lower() not in ("host", "content-length")
        }
        body = await request.body()

        server.mark_request_started()
        try:
            timeout = httpx.Timeout(
                connect=10.0,
                read=manager.config.limits.request_timeout,
                write=30.0,
                pool=5.0,
            )
            client = httpx.AsyncClient(timeout=timeout)

            # Check if client wants streaming (SSE / chunked)
            accept = request.headers.get("accept", "")
            wants_stream = "text/event-stream" in accept or "stream" in accept

            if wants_stream:
                async def stream_gen():
                    try:
                        async with client.stream(
                            request.method, target_url, headers=headers, content=body
                        ) as resp:
                            async for chunk in resp.aiter_bytes():
                                yield chunk
                    finally:
                        server.mark_request_done()
                        await client.aclose()

                return StreamingResponse(
                    stream_gen(),
                    media_type="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
                )
            else:
                try:
                    resp = await client.request(
                        request.method, target_url, headers=headers, content=body
                    )
                    resp_headers = {
                        k: v for k, v in resp.headers.items()
                        if k.lower() not in ("transfer-encoding",)
                    }
                    return Response(
                        content=resp.content,
                        status_code=resp.status_code,
                        headers=resp_headers,
                    )
                finally:
                    server.mark_request_done()
                    await client.aclose()
        except httpx.TimeoutException:
            server.mark_request_done()
            raise HTTPException(status_code=504, detail="Backend request timed out")
        except Exception as e:
            server.mark_request_done()
            log.exception("Proxy error for %s: %s", server.cfg.name, e)
            raise HTTPException(status_code=502, detail=str(e))

    return app


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(description="GPU Inference Proxy")
    parser.add_argument(
        "--config", "-c",
        default="config.yaml",
        help="Path to config.yaml (default: ./config.yaml)",
    )
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        log.error("Config file not found: %s", cfg_path)
        sys.exit(1)

    config = Config.load(str(cfg_path))
    manager = ProcessManager(config)
    app = build_app(manager)

    uvicorn.run(
        app,
        host=config.proxy.host,
        port=config.proxy.port,
        log_level=config.proxy.log_level,
        access_log=False,
    )


if __name__ == "__main__":
    main()
