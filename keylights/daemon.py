"""Ties sources, effects and the two lighting chips together.

poll() reads the sources and may block on the network; render() only computes
a frame and writes it. The service runs them on separate threads so a slow
inference server cannot stall the blink.
"""
from __future__ import annotations

import dataclasses
import logging
import threading
import time
from typing import Any

from .config import Config
from .device import DeviceError
from .effects import MANAGED_KEYS, Inputs, build_frame
from .sources import LLAMA_DOWN, LlamaState, net_rate, token_rate

log = logging.getLogger(__name__)

# The server evaluates the prompt in batches and answers /slots between them, so
# progress arrives in lumps seconds apart. Blinking holds this long after the last one.
PROMPT_ADVANCE_HOLD_S = 6.0


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
        self._prompt_done: int | None = None
        self._prompt_mark = 0.0  # when the prompt counter last moved, or reading began
        self._prompt_advanced_at: float | None = None
        self._prompt_rate = 0.0
        self._backlight: str | None = None

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
        boost = cfg.dim_boost if self._backlight == "dim" else 1.0
        frame = build_frame(self._inputs, now, cfg.base_color, cfg.del_blink_cap, cfg.esc_blink_cap, boost)
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
        previous = self._inputs
        if llama.up and not llama.answered and previous.llama.up:
            # A busy server stops answering /slots; its last reading is the best evidence.
            llama, rate = previous.llama, previous.token_rate
        else:
            dt = now - self._last_fast if self._last_fast is not None else 0.0
            rate = token_rate(self._prev_decoded, llama.decoded, dt) if llama.processing else 0.0
            self._prev_decoded = llama.decoded
            self._track_prompt(llama, now)

        cur_bytes = self._safe(self._sources.net_bytes, None, "network counters")
        net = 0.0
        if cur_bytes is not None:
            if self._prev_net is not None:
                net = net_rate(self._prev_net[0], self._prev_net[1], cur_bytes, now)
            self._prev_net = (cur_bytes, now)

        self._backlight = self._safe(self._sources.backlight_state, None, "backlight state")
        advancing = self._prompt_advanced_at is not None and now - self._prompt_advanced_at <= PROMPT_ADVANCE_HOLD_S
        self._inputs = dataclasses.replace(
            previous,
            llama=llama,
            token_rate=rate,
            net_bytes_per_s=net,
            prompt_rate=self._prompt_rate if llama.reading else 0.0,
            prompt_advancing=llama.reading and advancing,
        )

    def _track_prompt(self, llama: LlamaState, now: float) -> None:
        """Turns the prompt-evaluated counter into a smoothed tokens-per-second rate."""
        done = llama.prompt_done
        if not llama.reading or done is None:
            self._prompt_done, self._prompt_advanced_at, self._prompt_rate = None, None, 0.0
            return
        if self._prompt_done is None or done < self._prompt_done:  # reading began, or a new request
            self._prompt_done, self._prompt_mark, self._prompt_advanced_at, self._prompt_rate = done, now, None, 0.0
        elif done > self._prompt_done:
            elapsed = now - self._prompt_mark
            if elapsed > 0:
                sample = (done - self._prompt_done) / elapsed
                self._prompt_rate = sample if self._prompt_rate == 0 else (self._prompt_rate + sample) / 2
            self._prompt_done, self._prompt_mark, self._prompt_advanced_at = done, now, now

    def _poll_slow(self) -> None:
        temp = self._safe(self._sources.cpu_temp, None, "cpu temperature")
        muted = self._safe(self._sources.mic_muted, False, "mic state")
        self._inputs = dataclasses.replace(self._inputs, temp_c=temp, muted=muted)

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
