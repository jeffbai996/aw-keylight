"""Ties sources, effects and the two lighting chips together.

Polling reads the sources and may block on the network; render() only computes
a frame and writes it. The service polls each server on its own thread and
renders on another, so a slow server stalls neither the blink nor the other keys.
"""
from __future__ import annotations

import dataclasses
import logging
import threading
import time
from typing import Any

from .config import Config, LightSpec
from .device import DeviceError
from .effects import Inputs, build_frame, managed_keys, power_state
from .keymap import key_id
from .sources import GPU_DOWN, LLAMA_DOWN, OLLAMA_DOWN, net_rate
from .trackers import GpuTracker, LlamaTracker, OllamaTracker

log = logging.getLogger(__name__)

# One tracker class and one "source is down" state per kind of light.
TRACKERS = {"llama": LlamaTracker, "ollama": OllamaTracker, "gpu": GpuTracker}
DOWN_STATES = {"llama": LLAMA_DOWN, "ollama": OLLAMA_DOWN, "gpu": GPU_DOWN}


class Daemon:
    """Each light is polled on its own (poll_light), and the readings local to this machine
    together (poll_local), so the service can give every server its own thread: a server that
    hangs then holds up only its own key. State shared with render() is replaced under a lock."""

    def __init__(self, sources: Any, keyboard: Any, power: Any, config: Config):
        self._sources = sources
        self._keyboard = keyboard
        self._power = power
        self._config = config
        self._lock = threading.Lock()
        self._inputs = Inputs(net_bytes_per_s=0.0, temp_c=None, muted=False)
        self._trackers = {key_id(light.key): TRACKERS[light.kind]() for light in config.lights}
        self._light_polled: dict[int, float] = {}
        self._last_lights: float | None = None
        self._last_fast: float | None = None
        self._last_slow: float | None = None
        self._prev_net: tuple[int, float] | None = None
        self._backlight: str | None = None

    def tick(self, now: float) -> None:
        self.poll(now)
        self.render(now)

    def poll(self, now: float) -> None:
        """Every source in turn, on the calling thread."""
        if self._last_lights is None or now - self._last_lights >= self._config.poll_interval:
            for light in self._config.lights:
                self.poll_light(light, now)
            self._last_lights = now
        self.poll_local(now)

    def poll_light(self, light: LightSpec, now: float) -> None:
        key = key_id(light.key)
        state = self._safe(
            lambda: self._sources.inference(light.key), DOWN_STATES[light.kind], f"{light.key} inference"
        )
        last = self._light_polled.get(key)
        self._light_polled[key] = now
        reading = self._trackers[key].update(state, now, now - last if last is not None else 0.0)
        with self._lock:
            self._inputs = dataclasses.replace(self._inputs, lights={**self._inputs.lights, key: reading})

    def poll_local(self, now: float) -> None:
        """Network counters and backlight at the fast interval; temperature, mute and connectivity
        at the slow one."""
        if self._last_fast is None or now - self._last_fast >= self._config.poll_interval:
            self._poll_fast(now)
            self._last_fast = now
        if self._last_slow is None or now - self._last_slow >= self._config.slow_poll_interval:
            self._poll_slow()
            self._last_slow = now

    def render(self, now: float) -> None:
        cfg = self._config
        with self._lock:
            inputs, backlight = self._inputs, self._backlight
        boost = cfg.dim_boost if backlight == "dim" else 1.0
        frame = build_frame(
            inputs, now, cfg.base_color, cfg.del_blink_cap, cfg.esc_blink_cap, boost,
            cfg.compact_tokens, cfg.compact_blink,
        )
        try:
            self._keyboard.update(frame)
        except DeviceError as exc:
            log.warning("keyboard update failed: %s", exc)
        try:
            self._power.update(power_state(inputs.lights), now)
        except DeviceError as exc:
            log.warning("power button update failed: %s", exc)

    def shutdown(self) -> None:
        try:
            self._keyboard.update({key: self._config.base_color for key in managed_keys(self._trackers)})
            self._power.restore()
        except DeviceError as exc:
            log.warning("restore failed: %s", exc)

    def _poll_fast(self, now: float) -> None:
        cur_bytes = self._safe(self._sources.net_bytes, None, "network counters")
        net = 0.0
        if cur_bytes is not None:
            if self._prev_net is not None:
                net = net_rate(self._prev_net[0], self._prev_net[1], cur_bytes, now)
            self._prev_net = (cur_bytes, now)
        backlight = self._safe(self._sources.backlight_state, None, "backlight state")
        local_busy = self._safe(self._sources.local_busy, None, "local GPU activity")
        with self._lock:
            self._backlight = backlight
            self._inputs = dataclasses.replace(self._inputs, net_bytes_per_s=net, local_busy=local_busy)

    def _poll_slow(self) -> None:
        temp = self._safe(self._sources.cpu_temp, None, "cpu temperature")
        gpu_temp = self._safe(self._sources.gpu_temp, None, "gpu temperature")
        muted = self._safe(self._sources.mic_muted, False, "mic state")
        online = self._safe(self._sources.online, True, "connectivity")
        with self._lock:
            self._inputs = dataclasses.replace(
                self._inputs, temp_c=temp, gpu_temp_c=gpu_temp, muted=muted, online=online
            )

    @staticmethod
    def _safe(read, fallback, label: str):
        try:
            return read()
        except Exception as exc:  # a broken source must not stop the other lights
            log.warning("%s unavailable: %s", label, exc)
            return fallback


def run(daemon: Daemon, config: Config, stop: threading.Event) -> None:
    """One polling thread per light and one for this machine's own readings; render at tick_hz on
    this thread until stop is set."""
    def every(interval: float, poll) -> None:
        while not stop.is_set():
            poll(time.monotonic())
            stop.wait(interval)

    workers = [
        threading.Thread(
            target=every, args=(config.poll_interval, lambda now, light=light: daemon.poll_light(light, now)),
            name=f"keylights-{light.key.lower()}", daemon=True,
        )
        for light in config.lights
    ]
    workers.append(threading.Thread(
        target=every, args=(min(config.poll_interval, 0.25), daemon.poll_local), name="keylights-local", daemon=True,
    ))
    for worker in workers:
        worker.start()
    period = 1.0 / config.tick_hz
    try:
        while not stop.is_set():
            started = time.monotonic()
            daemon.render(started)
            stop.wait(max(0.0, period - (time.monotonic() - started)))
    finally:
        stop.set()
        # A thread still waiting on a server is left behind: it is a daemon thread and holds no device.
        for worker in workers:
            worker.join(timeout=2)
        daemon.shutdown()
