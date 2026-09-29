"""Readers for the values the lights show. Parsing is pure; the readers do I/O."""
from __future__ import annotations

import json
import subprocess
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

GetJson = Callable[..., Any]
CPU_SENSORS = {"coretemp", "k10temp"}


@dataclass(frozen=True)
class LlamaState:
    up: bool
    processing: bool
    decoded: int

    @property
    def reading(self) -> bool:
        # A busy slot that has produced no tokens yet is still reading the prompt.
        return self.processing and self.decoded == 0


LLAMA_DOWN = LlamaState(up=False, processing=False, decoded=0)


def parse_net_bytes(text: str) -> int:
    """Total rx+tx bytes over every interface except loopback."""
    total = 0
    for line in text.splitlines():
        name, sep, rest = line.partition(":")
        if not sep or name.strip() == "lo":
            continue
        fields = rest.split()
        if len(fields) >= 9:
            total += int(fields[0]) + int(fields[8])
    return total


def net_rate(prev_bytes: int, prev_t: float, cur_bytes: int, cur_t: float) -> float:
    if cur_bytes < prev_bytes or cur_t <= prev_t:
        return 0.0
    return (cur_bytes - prev_bytes) / (cur_t - prev_t)


def token_rate(prev_decoded: int, cur_decoded: int, dt: float) -> float:
    # The counter restarts with each request, so a drop means a new request.
    if cur_decoded < prev_decoded or dt <= 0:
        return 0.0
    return (cur_decoded - prev_decoded) / dt


def _slot_decoded(slot: dict) -> int:
    next_token = slot.get("next_token") or [{}]
    entry = next_token[0] if isinstance(next_token, list) else next_token
    return int(entry.get("n_decoded", 0))


def llama_state(get_json: GetJson) -> LlamaState:
    """Reads /slots for loaded models only: querying an unloaded one would autoload it."""
    try:
        models = get_json("/v1/models")["data"]
    except (OSError, ValueError, KeyError, TypeError):
        return LLAMA_DOWN

    processing, decoded = False, 0
    for model in models:
        if model.get("status", {}).get("value") != "loaded":
            continue
        try:
            slots = get_json("/slots", {"model": model["id"], "autoload": "false"})
        except (OSError, ValueError):
            continue
        for slot in slots:
            processing = processing or bool(slot.get("is_processing"))
            decoded += _slot_decoded(slot)
    return LlamaState(up=True, processing=processing, decoded=decoded)


def cpu_temp_c(hwmon_root: Path = Path("/sys/class/hwmon")) -> float | None:
    hottest: float | None = None
    for sensor in sorted(hwmon_root.glob("hwmon*")):
        try:
            if (sensor / "name").read_text().strip() not in CPU_SENSORS:
                continue
            for reading in sensor.glob("temp*_input"):
                value = int(reading.read_text()) / 1000
                hottest = value if hottest is None else max(hottest, value)
        except (OSError, ValueError):
            continue
    return hottest


def parse_mic_muted(wpctl_output: str) -> bool:
    return "[MUTED]" in wpctl_output


def make_http_get(base_url: str, timeout: float = 1.0) -> GetJson:
    def get(path: str, params: dict | None = None) -> Any:
        url = base_url.rstrip("/") + path
        if params:
            url += "?" + urllib.parse.urlencode(params)
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return json.load(response)

    return get


class Sources:
    """The real readers, bundled behind the methods the daemon calls."""

    def __init__(self, llama_get: GetJson | None):
        self._llama_get = llama_get

    def llama(self) -> LlamaState:
        return llama_state(self._llama_get) if self._llama_get else LLAMA_DOWN

    def net_bytes(self) -> int:
        return parse_net_bytes(Path("/proc/net/dev").read_text())

    def cpu_temp(self) -> float | None:
        return cpu_temp_c()

    def mic_muted(self) -> bool:
        result = subprocess.run(
            ["wpctl", "get-volume", "@DEFAULT_AUDIO_SOURCE@"],
            capture_output=True, text=True, timeout=2, check=False,
        )
        return parse_mic_muted(result.stdout)
