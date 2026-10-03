"""Runtime settings, read from the environment (a gitignored .env in practice)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping

Color = tuple[int, int, int]


@dataclass(frozen=True)
class Config:
    llama_url: str | None
    base_color: Color
    del_blink_cap: float
    esc_blink_cap: float
    poll_interval: float
    slow_poll_interval: float
    power_zone: int
    tick_hz: float
    dim_boost: float
    backlight_state_path: str


DEFAULT_BACKLIGHT_STATE = "~/.local/state/kbd-light/state"


def _hex_color(value: str) -> Color:
    value = value.lstrip("#")
    return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)


def load_config(env: Mapping[str, str]) -> Config:
    return Config(
        llama_url=env.get("KEYLIGHTS_LLAMA_URL") or None,
        base_color=_hex_color(env.get("KEYLIGHTS_BASE_COLOR", "ffffff")),
        del_blink_cap=float(env.get("KEYLIGHTS_DEL_BLINK_CAP", "20")),
        esc_blink_cap=float(env.get("KEYLIGHTS_ESC_BLINK_CAP", "10")),
        poll_interval=float(env.get("KEYLIGHTS_POLL_INTERVAL", "0.5")),
        slow_poll_interval=float(env.get("KEYLIGHTS_SLOW_POLL_INTERVAL", "2")),
        power_zone=int(env.get("KEYLIGHTS_POWER_ZONE", "1")),
        tick_hz=float(env.get("KEYLIGHTS_TICK_HZ", "50")),
        dim_boost=float(env.get("KEYLIGHTS_DIM_BOOST", "1.6")),
        backlight_state_path=os.path.expanduser(env.get("KEYLIGHTS_BACKLIGHT_STATE", DEFAULT_BACKLIGHT_STATE)),
    )
