"""Runtime settings, read from the environment (a gitignored .env in practice)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping

from .keymap import UnknownKeyError, key_id

Color = tuple[int, int, int]


LIGHT_KINDS = ("llama", "ollama", "gpu")
# Keys the status lights already drive; a light cannot claim them.
RESERVED_KEYS = frozenset({"ESC", "F1", "F2", "F3", "F4", "F5"})


@dataclass(frozen=True)
class LightSpec:
    """One light: which key shows which source. The selector narrows the source: a router
    model for llama, the host whose GPU to show for gpu, and for ollama an optional GPU overlay
    written gpu=telemetry-url@host."""

    key: str
    kind: str
    url: str
    selector: str | None = None


@dataclass(frozen=True)
class Config:
    llama_url: str | None
    lights: tuple[LightSpec, ...]
    base_color: Color
    del_blink_cap: float
    esc_blink_cap: float
    poll_interval: float
    slow_poll_interval: float
    power_zone: int
    tick_hz: float
    dim_boost: float
    backlight_state_path: str
    compact_tokens: int
    compact_blink: float
    reset_interval: float


DEFAULT_BACKLIGHT_STATE = "~/.local/state/kbd-light/state"


def _hex_color(value: str) -> Color:
    value = value.lstrip("#")
    return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)


def _is_gpu_overlay(selector: str) -> bool:
    """gpu=<telemetry url>@<host>: where to read the GPU activity that completes an ollama light."""
    url, _, host = selector.removeprefix("gpu=").rpartition("@") if selector.startswith("gpu=") else ("", "", "")
    return bool(url and host)


def parse_lights(text: str) -> tuple[LightSpec, ...]:
    """KEY=kind:url[#selector], comma separated, e.g. END=llama:http://host:8080#model."""
    lights: list[LightSpec] = []
    for entry in (part.strip() for part in text.split(",")):
        if not entry:
            continue
        name, has_eq, source = entry.partition("=")
        kind, has_kind, location = source.partition(":")
        if not (has_eq and has_kind and location.strip()):
            raise ValueError(f"Light '{entry}': expected KEY=kind:url")
        key, kind = name.strip().upper(), kind.strip().lower()
        if kind not in LIGHT_KINDS:
            raise ValueError(f"Light '{entry}': kind must be one of {', '.join(LIGHT_KINDS)}")
        try:
            key_id(key)
        except UnknownKeyError as exc:
            raise ValueError(str(exc)) from None
        if key in RESERVED_KEYS:
            raise ValueError(f"Light '{entry}': {key} is used by a status light")
        if any(light.key == key for light in lights):
            raise ValueError(f"Light '{entry}': {key} is configured twice")
        url, _, selector = location.strip().partition("#")
        if kind == "gpu" and not selector:
            raise ValueError(f"Light '{entry}': a gpu light needs the host to show, as url#host")
        if kind == "ollama" and selector and not _is_gpu_overlay(selector):
            raise ValueError(f"Light '{entry}': an ollama light's selector must be gpu=telemetry-url@host")
        lights.append(LightSpec(key, kind, url, selector or None))
    return tuple(lights)


def load_config(env: Mapping[str, str]) -> Config:
    llama_url = env.get("KEYLIGHTS_LLAMA_URL") or None
    spec = env.get("KEYLIGHTS_LIGHTS")
    if spec:
        lights = parse_lights(spec)
    else:  # the single-server setting from before per-key lights
        lights = (LightSpec("DEL", "llama", llama_url),) if llama_url else ()
    return Config(
        llama_url=llama_url,
        lights=lights,
        base_color=_hex_color(env.get("KEYLIGHTS_BASE_COLOR", "ffffff")),
        del_blink_cap=float(env.get("KEYLIGHTS_DEL_BLINK_CAP", "20")),
        esc_blink_cap=float(env.get("KEYLIGHTS_ESC_BLINK_CAP", "10")),
        poll_interval=float(env.get("KEYLIGHTS_POLL_INTERVAL", "0.5")),
        slow_poll_interval=float(env.get("KEYLIGHTS_SLOW_POLL_INTERVAL", "2")),
        power_zone=int(env.get("KEYLIGHTS_POWER_ZONE", "1")),
        tick_hz=float(env.get("KEYLIGHTS_TICK_HZ", "50")),
        dim_boost=float(env.get("KEYLIGHTS_DIM_BOOST", "1.6")),
        backlight_state_path=os.path.expanduser(env.get("KEYLIGHTS_BACKLIGHT_STATE", DEFAULT_BACKLIGHT_STATE)),
        compact_tokens=int(env.get("KEYLIGHTS_COMPACT_TOKENS", "12288")),
        compact_blink=float(env.get("KEYLIGHTS_COMPACT_BLINK", "10")),
        reset_interval=float(env.get("KEYLIGHTS_RESET_INTERVAL", "60")),
    )
