"""Readers for the values the lights show. Parsing is pure; the readers do I/O."""
from __future__ import annotations

import functools
import json
import re
import subprocess
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping

GetJson = Callable[..., Any]
CPU_SENSORS = {"coretemp", "k10temp"}


BACKLIGHT_STATES = ("on", "dim", "off")
CONNECTIVITY_PROBE_S = 30.0


@dataclass(frozen=True)
class LlamaState:
    up: bool
    processing: bool
    decoded: int
    # Prompt tokens still to evaluate (net of the cache) and already evaluated;
    # None when the server does not report them.
    prompt_new: int | None = None
    prompt_done: int | None = None
    # False when a loaded model's /slots did not answer, which happens while it is busy.
    answered: bool = True
    # With several slots, whether any one of them is reading its prompt. A generating slot
    # must not hide another slot's prompt, so this is decided per slot; None means
    # "work it out from decoded and the prompt counters" (single-slot readings).
    prompt_phase: bool | None = None
    # The largest prompt_done of any busy slot. Compaction is judged per slot, never summed.
    peak_prompt_done: int | None = None
    # False when the server is up but the model is not resident (parked). A busy slot implies True.
    loaded: bool = True

    @property
    def reading(self) -> bool:
        if self.prompt_phase is not None:
            return self.processing and self.prompt_phase
        # A busy slot that has produced no tokens yet is reading the prompt, until
        # the whole prompt is evaluated and only the first token is pending.
        if not (self.processing and self.decoded == 0):
            return False
        if self.prompt_new is None or self.prompt_done is None:
            return True
        return self.prompt_done < self.prompt_new


LLAMA_DOWN = LlamaState(up=False, processing=False, decoded=0)


@dataclass(frozen=True)
class OllamaState:
    """Ollama reports which models are loaded, not whether one is busy."""

    up: bool
    loaded: bool  # a chat model is loaded; embedding models are ignored
    touched: float | None  # latest expiry among chat models; it moves when a request finishes
    answered: bool = True
    # Live activity, from an ollama-gate's /_gate/activity; None when the gate has no such
    # endpoint, in which case the light falls back to loaded plus the finished-request pulse.
    in_flight: int | None = None
    awaiting: int | None = None  # in flight with nothing streamed back yet: the prompt is being read
    events: int | None = None  # events streamed so far; only its growth means anything
    # What the host's GPU is doing, when an overlay is configured: it shows work the gate cannot
    # see, such as embeddings and whisper. None when there is no overlay or it could not be read.
    gpu: "GpuState | None" = None


OLLAMA_DOWN = OllamaState(up=False, loaded=False, touched=None)


@dataclass(frozen=True)
class GpuState:
    """What a host's GPU is doing, from a fleet status server's telemetry. Power draw and performance
    state show a short burst of work that utilisation, sampled coarsely, misses."""

    up: bool
    power_w: float | None
    limit_w: float | None
    pstate: str | None  # P8 is the idle state
    answered: bool = True


GPU_DOWN = GpuState(up=False, power_w=None, limit_w=None, pstate=None)
# A host whose last probe is older than this is not trusted to be showing the card now.
GPU_MAX_SAMPLE_AGE_S = 30.0
# Embedders such as the fleet's bge-m3 stay loaded all day and are not inference.
EMBEDDING_MODEL = re.compile(r"bge|embed|nomic|minilm|mxbai|e5-|gte-", re.IGNORECASE)
_EXCESS_FRACTION = re.compile(r"(\.\d{6})\d+")


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


def _is_timeout(exc: BaseException) -> bool:
    """A busy server stops answering; an error reply or a refused connection is a real answer."""
    return isinstance(exc, TimeoutError) or isinstance(getattr(exc, "reason", None), TimeoutError)


def llama_state(get_json: GetJson, only_model: str | None = None) -> LlamaState:
    """Reads /slots for loaded models only: querying an unloaded one would autoload it.

    only_model limits the reading to one model of a router."""
    try:
        models = get_json("/v1/models")["data"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        if _is_timeout(exc):
            return LlamaState(up=True, processing=False, decoded=0, answered=False)
        return LLAMA_DOWN

    answered = True
    loaded = False
    rows: list[tuple[int, int | None, int | None]] = []  # busy slots: (decoded, prompt_new, prompt_done)
    for model in models:
        if only_model is not None and model.get("id") != only_model:
            continue
        if model.get("status", {}).get("value") != "loaded":
            continue
        loaded = True
        try:
            slots = get_json("/slots", {"model": model["id"], "autoload": "false"})
        except (OSError, ValueError) as exc:
            answered = answered and not _is_timeout(exc)
            continue
        for slot in slots:
            # An idle slot keeps its last task's counters, so only a busy one is read.
            if not slot.get("is_processing"):
                continue
            total, processed = slot.get("n_prompt_tokens"), slot.get("n_prompt_tokens_processed")
            has_prompt = isinstance(total, int) and isinstance(processed, int)
            new = max(0, total - int(slot.get("n_prompt_tokens_cache") or 0)) if has_prompt else None
            rows.append((_slot_decoded(slot), new, processed if has_prompt else None))
    return _merge_slots(rows, answered, loaded)


def _merge_slots(rows: list[tuple[int, int | None, int | None]], answered: bool, loaded: bool) -> LlamaState:
    """One state for a server whose slots may be doing different things at once."""
    if not rows:
        return LlamaState(up=True, processing=False, decoded=0, answered=answered, loaded=loaded)

    def is_reading(row) -> bool:
        decoded, new, done = row
        return decoded == 0 and (new is None or done is None or done < new)

    reading = [row for row in rows if is_reading(row)]
    with_prompt = [row for row in rows if row[2] is not None]
    prompt_new = prompt_done = peak = None
    if with_prompt:
        peak = max(row[2] for row in with_prompt)
        shown = [row for row in reading if row[2] is not None] or [max(with_prompt, key=lambda row: row[2])]
        prompt_new = sum(row[1] for row in shown)
        prompt_done = sum(row[2] for row in shown)
    return LlamaState(
        up=True,
        processing=True,
        decoded=sum(row[0] for row in rows),
        prompt_new=prompt_new,
        prompt_done=prompt_done,
        answered=answered,
        prompt_phase=bool(reading),
        peak_prompt_done=peak,
    )


def parse_expires_at(text: str) -> float | None:
    """Ollama stamps expiry with nanoseconds, which datetime does not accept."""
    try:
        return datetime.fromisoformat(_EXCESS_FRACTION.sub(r"\1", text)).timestamp()
    except (ValueError, TypeError):
        return None


def ollama_state(get_json: GetJson, gpu: Callable[[], GpuState] | None = None) -> OllamaState:
    try:
        models = get_json("/api/ps")["models"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        if _is_timeout(exc):
            return OllamaState(up=True, loaded=False, touched=None, answered=False)
        return OLLAMA_DOWN
    chat = [m for m in models if not EMBEDDING_MODEL.search(str(m.get("name", "")))]
    stamps = [t for t in (parse_expires_at(str(m.get("expires_at", ""))) for m in chat) if t is not None]
    in_flight, awaiting, events = _gate_activity(get_json) if chat else (None, None, None)
    return OllamaState(
        up=True, loaded=bool(chat), touched=max(stamps) if stamps else None,
        in_flight=in_flight, awaiting=awaiting, events=events, gpu=_overlay(gpu),
    )


def _overlay(gpu: Callable[[], GpuState] | None) -> GpuState | None:
    """The GPU reading, or None if there is none to trust: a source that is down or slow
    must not darken a light whose own server answered."""
    if gpu is None:
        return None
    reading = gpu()
    return reading if reading.up and reading.answered else None


def _gate_activity(get_json: GetJson) -> tuple[int | None, int | None, int | None]:
    """The gate's own request counters. A gate without the endpoint, or one too busy to
    answer, just means no live signal: the loaded state still stands."""
    try:
        data = get_json("/_gate/activity")
        values = (data["in_flight"], data["awaiting_first_event"], data["events_total"])
    except (OSError, ValueError, KeyError, TypeError):
        return None, None, None
    if not all(isinstance(v, int) and not isinstance(v, bool) for v in values):
        return None, None, None
    return values


def gpu_state(get_json: GetJson, host: str) -> GpuState:
    try:
        hosts = get_json("/api/telemetry")["hosts"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        if _is_timeout(exc):
            return GpuState(up=True, power_w=None, limit_w=None, pstate=None, answered=False)
        return GPU_DOWN
    entry = next((h for h in hosts if isinstance(h, dict) and h.get("host") == host), None)
    if entry is None or not entry.get("ok") or entry.get("sample_state") != "healthy":
        return GPU_DOWN
    age = entry.get("sample_age_sec")
    if not isinstance(age, (int, float)) or age > GPU_MAX_SAMPLE_AGE_S:
        return GPU_DOWN
    cards = [card for card in entry.get("gpu") or [] if isinstance(card, dict)]
    if not cards:
        return GpuState(up=True, power_w=None, limit_w=None, pstate=None)
    card = max(cards, key=lambda c: c.get("power_draw_w") or 0.0)
    return GpuState(up=True, power_w=card.get("power_draw_w"), limit_w=card.get("power_limit_w"), pstate=card.get("pstate"))


def parse_connectivity(text: str) -> bool:
    """NetworkManager's verdict as online or not. Anything it cannot tell is read as online, so
    a missing or confused check never darkens ESC on a guess."""
    return text.strip().lower() not in ("none", "limited", "portal")


def read_backlight_state(path: str | Path) -> str | None:
    """The kbd-light state file: on, dim or off. None when missing or unrecognised."""
    try:
        value = Path(path).read_text().strip()
    except OSError:
        return None
    return value if value in BACKLIGHT_STATES else None


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

    def __init__(self, readers: Mapping[str, Callable[[], Any]], backlight_path: str | None = None):
        self._readers = readers
        self._backlight_path = backlight_path
        self._last_probe = float("-inf")
        self._probe: subprocess.Popen | None = None

    def backlight_state(self) -> str | None:
        return read_backlight_state(self._backlight_path) if self._backlight_path else None

    def inference(self, name: str) -> Any:
        return self._readers[name]()

    def online(self) -> bool:
        """Whether the internet is up, from NetworkManager. Its own check runs every few minutes, so
        a fresh one is started at most every 30 s without waiting for it; a later read sees the answer.
        A link that drops is noticed at once, an outage upstream within about 30 s."""
        now = time.monotonic()
        if now - self._last_probe >= CONNECTIVITY_PROBE_S:
            self._last_probe = now
            try:
                if self._probe is not None:
                    self._probe.poll()  # reap the previous one
                self._probe = subprocess.Popen(
                    ["nmcli", "networking", "connectivity", "check"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
            except OSError:
                return True  # no nmcli: cannot tell
        try:
            result = subprocess.run(
                ["nmcli", "-t", "-g", "CONNECTIVITY", "general"],
                capture_output=True, text=True, timeout=2, check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return True
        return parse_connectivity(result.stdout)

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


def build_sources(lights, backlight_path: str | None) -> Sources:
    """One reader per configured light; llama.cpp, Ollama and GPU telemetry answer differently."""
    readers: dict[str, Callable[[], Any]] = {}
    for light in lights:
        if light.kind == "llama":
            readers[light.key] = functools.partial(llama_state, make_http_get(light.url, 1.0), light.selector)
        elif light.kind == "gpu":
            readers[light.key] = functools.partial(gpu_state, make_http_get(light.url, 2.0), light.selector)
        else:
            overlay = None
            if light.selector:  # gpu=<telemetry url>@<host>, checked when the config was read
                url, _, host = light.selector.removeprefix("gpu=").rpartition("@")
                overlay = functools.partial(gpu_state, make_http_get(url, 2.0), host)
            readers[light.key] = functools.partial(ollama_state, make_http_get(light.url, 2.0), overlay)
    return Sources(readers, backlight_path)
