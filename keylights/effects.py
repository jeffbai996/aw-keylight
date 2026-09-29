"""What each key shows for a given set of readings. Pure functions, no I/O."""
from __future__ import annotations

from dataclasses import dataclass

from .config import Color
from .keymap import key_id
from .sources import LlamaState

OFF: Color = (0, 0, 0)
GREEN: Color = (0, 255, 0)
AMBER: Color = (255, 140, 0)
RED: Color = (255, 0, 0)
NET_IDLE: Color = (0, 40, 80)
NET_ACTIVE: Color = (0, 170, 255)

TOKENS_PER_BLINK = 2.0
BYTES_PER_BLINK = 50_000.0
TEMP_LOW_C, TEMP_HIGH_C = 45.0, 90.0

TEMP_KEYS = tuple(key_id(f"F{n}") for n in range(1, 5))
MANAGED_KEYS = (key_id("ESC"), *TEMP_KEYS, key_id("F5"), key_id("DEL"))


@dataclass(frozen=True)
class Inputs:
    llama: LlamaState
    token_rate: float
    net_bytes_per_s: float
    temp_c: float | None
    muted: bool


def temp_color(temp_c: float) -> Color:
    """Green at 45 C through yellow to red at 90 C."""
    x = min(1.0, max(0.0, (temp_c - TEMP_LOW_C) / (TEMP_HIGH_C - TEMP_LOW_C)))
    return round(min(1.0, 2 * x) * 255), round(min(1.0, 2 * (1 - x)) * 255), 0


def blink_rate(tokens_per_s: float, cap: float = 20.0) -> float:
    if tokens_per_s <= 0:
        return 0.0
    return min(cap, tokens_per_s / TOKENS_PER_BLINK)


def net_blink_rate(bytes_per_s: float, cap: float = 10.0) -> float:
    if bytes_per_s <= 0:
        return 0.0
    return min(cap, bytes_per_s / BYTES_PER_BLINK)


def blink_on(rate: float, t: float) -> bool:
    """On for the first half of each period; a zero rate is steady on."""
    if rate <= 0:
        return True
    return (t * rate) % 1.0 < 0.5


def del_color(state: LlamaState, rate: float, t: float) -> Color:
    if not state.up:
        return OFF
    if state.reading:
        return AMBER
    if state.processing and not blink_on(rate, t):
        return OFF
    return GREEN


def esc_color(rate: float, t: float) -> Color:
    return NET_ACTIVE if rate > 0 and blink_on(rate, t) else NET_IDLE


def mute_color(muted: bool, base: Color) -> Color:
    return RED if muted else base


def build_frame(
    inputs: Inputs, t: float, base: Color, del_cap: float = 20.0, esc_cap: float = 10.0
) -> dict[int, Color]:
    frame = {
        key_id("DEL"): del_color(inputs.llama, blink_rate(inputs.token_rate, del_cap), t),
        key_id("ESC"): esc_color(net_blink_rate(inputs.net_bytes_per_s, esc_cap), t),
        key_id("F5"): mute_color(inputs.muted, base),
    }
    temp = base if inputs.temp_c is None else temp_color(inputs.temp_c)
    frame.update({key: temp for key in TEMP_KEYS})
    return frame
