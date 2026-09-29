"""Ties sources, effects and the two lighting chips together.

poll() reads the sources and may block on the network; render() only computes
a frame and writes it. The service runs them on separate threads so a slow
inference server cannot stall the blink.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any

from .config import Config
from .device import DeviceError
from .effects import MANAGED_KEYS, Inputs, build_frame
from .sources import LLAMA_DOWN, LlamaState, net_rate, token_rate

log = logging.getLogger(__name__)


class Daemon:
    def __init__(self, sources: Any, keyboard: Any, power: Any, config: Config):
        self._sources = sources
        self._keyboard = keyboard
        self._power = power
        self._config = config
        self._inputs = Inputs(llama=LLAMA_DOWN, token_rate=0.0, net_bytes_per_s=0.0, temp_c=None, muted=False)
        self._last_fast: float | None = None
        self._last_slow: float | None = None
        self._prev_decoded = 0
        self._prev_net: tuple[int, float] | None = None

    def tick(self, now: float) -> None:
        self.poll(now)
        self.render(now)

    def poll(self, now: float) -> None:
        if self._last_fast is None or now - self._last_fast >= self._config.poll_interval:
            self._poll_fast(now)
            self._last_fast = now
        if self._last_slow is None or now - self._last_slow >= self._config.slow_poll_interval:
            self._poll_slow()
            self._last_slow = now

    def render(self, now: float) -> None:
        cfg = self._config
        frame = build_frame(self._inputs, now, cfg.base_color, cfg.del_blink_cap, cfg.esc_blink_cap)
        try:
            self._keyboard.update(frame)
        except DeviceError as exc:
            log.warning("keyboard update failed: %s", exc)
        try:
            self._power.update(self._inputs.llama, now)
        except DeviceError as exc:
            log.warning("power button update failed: %s", exc)

    def shutdown(self) -> None:
        try:
            self._keyboard.update({key: self._config.base_color for key in MANAGED_KEYS})
            self._power.restore()
        except DeviceError as exc:
            log.warning("restore failed: %s", exc)

    def _poll_fast(self, now: float) -> None:
        llama = self._safe(self._sources.llama, LLAMA_DOWN, "inference server")
        dt = now - self._last_fast if self._last_fast is not None else 0.0
        rate = token_rate(self._prev_decoded, llama.decoded, dt) if llama.processing else 0.0
        self._prev_decoded = llama.decoded

        cur_bytes = self._safe(self._sources.net_bytes, None, "network counters")
        net = 0.0
        if cur_bytes is not None:
            if self._prev_net is not None:
                net = net_rate(self._prev_net[0], self._prev_net[1], cur_bytes, now)
            self._prev_net = (cur_bytes, now)

        self._inputs = Inputs(llama, rate, net, self._inputs.temp_c, self._inputs.muted)

    def _poll_slow(self) -> None:
        temp = self._safe(self._sources.cpu_temp, None, "cpu temperature")
        muted = self._safe(self._sources.mic_muted, False, "mic state")
        i = self._inputs
        self._inputs = Inputs(i.llama, i.token_rate, i.net_bytes_per_s, temp, muted)

    @staticmethod
    def _safe(read, fallback, label: str):
        try:
            return read()
        except Exception as exc:  # a broken source must not stop the other lights
            log.warning("%s unavailable: %s", label, exc)
            return fallback


def run(daemon: Daemon, config: Config, stop: threading.Event) -> None:
    """Poll on a worker thread; render at tick_hz on this one until stop is set."""
    def poll_loop() -> None:
        while not stop.is_set():
            daemon.poll(time.monotonic())
            stop.wait(min(config.poll_interval, 0.25))

    worker = threading.Thread(target=poll_loop, name="keylights-poll", daemon=True)
    worker.start()
    period = 1.0 / config.tick_hz
    try:
        while not stop.is_set():
            started = time.monotonic()
            daemon.render(started)
            stop.wait(max(0.0, period - (time.monotonic() - started)))
    finally:
        stop.set()
        worker.join(timeout=2)
        daemon.shutdown()
