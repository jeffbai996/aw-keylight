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
NET_IDLE: Color = (0, 10, 20)
NET_ACTIVE: Color = (0, 170, 255)

TOKENS_PER_BLINK = 2.0
PROMPT_TOKENS_PER_BLINK = 100.0
MIN_PROMPT_BLINK = 2.0  # slow prompt progress still has to read as blinking, not steady
BYTES_PER_BLINK = 10_000.0
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
    prompt_rate: float = 0.0  # prompt tokens evaluated per second
    prompt_advancing: bool = False  # prompt progress seen within the hold window


def temp_color(temp_c: float) -> Color:
    """Green at 45 C through yellow to red at 90 C."""
    x = min(1.0, max(0.0, (temp_c - TEMP_LOW_C) / (TEMP_HIGH_C - TEMP_LOW_C)))
    return round(min(1.0, 2 * x) * 255), round(min(1.0, 2 * (1 - x)) * 255), 0


def blink_rate(tokens_per_s: float, cap: float = 20.0) -> float:
    if tokens_per_s <= 0:
        return 0.0
    return min(cap, tokens_per_s / TOKENS_PER_BLINK)


def prompt_blink_rate(prompt_tokens_per_s: float, cap: float = 20.0) -> float:
    if prompt_tokens_per_s <= 0:
        return 0.0
    return min(cap, max(MIN_PROMPT_BLINK, prompt_tokens_per_s / PROMPT_TOKENS_PER_BLINK))


def net_blink_rate(bytes_per_s: float, cap: float = 10.0) -> float:
    if bytes_per_s <= 0:
        return 0.0
    return min(cap, bytes_per_s / BYTES_PER_BLINK)


def blink_on(rate: float, t: float) -> bool:
    """On for the first half of each period; a zero rate is steady on."""
    if rate <= 0:
        return True
    return (t * rate) % 1.0 < 0.5


def del_color(state: LlamaState, rate: float, t: float, prompt_advancing: bool = False) -> Color:
    """Amber while the prompt is read (blinking while it advances), then green:
    steady for the first-token wait and when done, flickering while generating."""
    if not state.up:
        return OFF
    if state.reading:
        return OFF if prompt_advancing and not blink_on(rate, t) else AMBER
    if state.processing and not blink_on(rate, t):
        return OFF
    return GREEN


def esc_color(rate: float, t: float) -> Color:
    return NET_ACTIVE if rate > 0 and blink_on(rate, t) else NET_IDLE


def mute_color(muted: bool, base: Color) -> Color:
    return RED if muted else base


def boost_color(color: Color, factor: float) -> Color:
    r, g, b = color
    return min(255, round(r * factor)), min(255, round(g * factor)), min(255, round(b * factor))


def build_frame(
    inputs: Inputs, t: float, base: Color, del_cap: float = 20.0, esc_cap: float = 10.0, boost: float = 1.0
) -> dict[int, Color]:
    """boost lifts ESC and DEL only: the backlight level is global, so in dim mode
    they are the keys that would otherwise vanish."""
    if inputs.llama.reading:
        del_rate = prompt_blink_rate(inputs.prompt_rate, del_cap)
    else:
        del_rate = blink_rate(inputs.token_rate, del_cap)
    del_key = del_color(inputs.llama, del_rate, t, inputs.prompt_advancing)
    esc_key = esc_color(net_blink_rate(inputs.net_bytes_per_s, esc_cap), t)
    frame = {
        key_id("DEL"): boost_color(del_key, boost),
        key_id("ESC"): boost_color(esc_key, boost),
        key_id("F5"): mute_color(inputs.muted, base),
    }
    temp = base if inputs.temp_c is None else temp_color(inputs.temp_c)
    frame.update({key: temp for key in TEMP_KEYS})
    return frame
