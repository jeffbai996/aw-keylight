"""What each key shows for a given set of readings. Pure functions, no I/O."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Union

from .config import Color
from .keymap import key_id
from .sources import GpuState, LlamaState, OllamaState

OFF: Color = (0, 0, 0)
GREEN: Color = (0, 255, 0)
AMBER: Color = (255, 140, 0)
RED: Color = (255, 0, 0)
WHITE: Color = (255, 255, 255)
# Idle: the host is up with its model loaded and nothing is running.
BABY_BLUE: Color = (0, 30, 255)
# ESC while the internet is up. Its own blue, separate from the idle lights'.
NET_LIT: Color = (0, 170, 255)
NET_DIP: Color = (0, 10, 20)  # ESC's dip with traffic. Dark is kept for "no internet".

TOKENS_PER_BLINK = 2.0
PROMPT_TOKENS_PER_BLINK = 100.0
GPU_IDLE_PSTATE = "P8"
GPU_IDLE_POWER_SHARE = 0.2  # below this share of the power limit the card is idle, whatever its pstate: some park in P1
GPU_BLINKS_AT_FULL_POWER = 40.0
MIN_GPU_BLINK = 2.0  # a card that is working has to read as flickering, not steady
OLLAMA_AWAITING_BLINK = 4.0  # blinks a second while a gate waits for a request's first event
MIN_PROMPT_BLINK = 2.0  # slow prompt progress still has to read as blinking, not steady
BYTES_PER_BLINK = 10_000.0
TEMP_LOW_C, TEMP_HIGH_C = 45.0, 90.0

TEMP_KEYS = tuple(key_id(f"F{n}") for n in range(1, 5))
STATUS_KEYS = (key_id("ESC"), *TEMP_KEYS, key_id("F5"))


@dataclass(frozen=True)
class InferenceInput:
    """One llama.cpp light: its slot state and the rates derived from successive readings."""

    state: LlamaState
    token_rate: float = 0.0
    prompt_rate: float = 0.0  # prompt tokens evaluated per second
    prompt_advancing: bool = False  # prompt progress seen within the hold window


@dataclass(frozen=True)
class OllamaInput:
    state: OllamaState
    pulse: bool  # a request finished within the pulse window
    event_rate: float = 0.0  # events streamed per second, from the gate's counter


@dataclass(frozen=True)
class GpuInput:
    state: GpuState


Light = Union[InferenceInput, OllamaInput, GpuInput]


@dataclass(frozen=True)
class Inputs:
    net_bytes_per_s: float
    temp_c: float | None
    muted: bool
    lights: Mapping[int, Light] = field(default_factory=dict)  # key id -> what that key shows
    online: bool = True  # False when there is no internet: ESC goes dark


def managed_keys(light_keys) -> tuple[int, ...]:
    return (*STATUS_KEYS, *light_keys)


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


def is_compaction(state: LlamaState, threshold: float) -> bool:
    """A busy slot that has evaluated at least `threshold` uncached prompt tokens. With
    several slots the largest one counts; their progress is never added together.

    The server does not label compaction. A very large uncached prompt is the closest
    signal: ordinary agent turns reuse the cached prefix and evaluate a few thousand tokens.
    """
    done = state.peak_prompt_done if state.peak_prompt_done is not None else state.prompt_done
    return state.processing and done is not None and done >= threshold


def del_color(
    state: LlamaState, rate: float, t: float, prompt_advancing: bool = False, compaction: bool = False
) -> Color:
    """Amber means a prompt is being read, and only that. A large prompt (`compaction`)
    flashes at `rate`; an ordinary one blinks while it advances and holds steady when it
    stalls. Once the prompt is read the key is green while the job runs: steady for the
    first-token wait, flickering with generated tokens. Idle with the model loaded is baby
    blue, an unloaded model is dark, and an unreachable server is red."""
    if not state.up:
        return RED
    if state.reading:
        if compaction:
            return AMBER if blink_on(rate, t) else OFF
        return OFF if prompt_advancing and not blink_on(rate, t) else AMBER
    if state.processing:
        return GREEN if blink_on(rate, t) else OFF
    return BABY_BLUE if state.loaded else OFF


def esc_color(rate: float, t: float, online: bool = True) -> Color:
    """Lit blue while the internet is up, dipping to dim blue with traffic. Dark with no internet."""
    if not online:
        return OFF
    return NET_DIP if rate > 0 and not blink_on(rate, t) else NET_LIT


def mute_color(muted: bool, base: Color) -> Color:
    return RED if muted else base


def boost_color(color: Color, factor: float) -> Color:
    r, g, b = color
    return min(255, round(r * factor)), min(255, round(g * factor)), min(255, round(b * factor))


def ollama_color(
    state: OllamaState, pulse: bool, t: float = 0.0, event_rate: float = 0.0, cap: float = 20.0
) -> Color:
    """Ollama light, in the same language as the llama one. A chat request through the gate:
    amber blink while its prompt is read, green flicker with streamed events (about one per
    token). Otherwise, if a GPU overlay says the card is working (embeddings, whisper): green
    flicker. Otherwise baby blue when a chat model is loaded (white for a moment as a request
    finishes) and dark when none is. Red when the server is unreachable. Gamemode, which stops
    Ollama on purpose, is dark: the card has been handed to a game."""
    if state.gpu is not None and state.gpu.gamemode:
        return OFF
    if not state.up:
        return RED
    if state.in_flight:
        if state.awaiting:
            return AMBER if blink_on(OLLAMA_AWAITING_BLINK, t) else OFF
        return GREEN if blink_on(blink_rate(event_rate, cap), t) else OFF
    working = gpu_blink_rate(state.gpu, cap) if state.gpu else 0.0
    if working > 0:
        return GREEN if blink_on(working, t) else OFF
    if pulse:
        return WHITE
    return BABY_BLUE if state.loaded else OFF


def gpu_blink_rate(state: GpuState, cap: float = 20.0) -> float:
    """Zero while the card idles; otherwise faster the closer its power draw is to the limit."""
    if not state.up or state.power_w is None or not state.limit_w:
        return 0.0
    share = state.power_w / state.limit_w
    active = share >= GPU_IDLE_POWER_SHARE and state.pstate != GPU_IDLE_PSTATE
    if not active:
        return 0.0
    return min(cap, max(MIN_GPU_BLINK, share * GPU_BLINKS_AT_FULL_POWER))


def gpu_color(state: GpuState, t: float, cap: float = 20.0) -> Color:
    """Baby blue while the host's GPU idles, green flickering while it works. There is no read
    phase to show, so there is no amber. Dark in gamemode, red when the host cannot be reached."""
    if state.gamemode:
        return OFF
    if not state.up:
        return RED
    rate = gpu_blink_rate(state, cap)
    if rate <= 0:
        return BABY_BLUE
    return GREEN if blink_on(rate, t) else OFF


def _inference_color(
    light: InferenceInput, t: float, cap: float, compact_tokens: float, compact_blink: float
) -> Color:
    state = light.state
    compaction = is_compaction(state, compact_tokens)
    if compaction and state.reading:
        rate = min(cap, compact_blink)
    elif state.reading:
        rate = prompt_blink_rate(light.prompt_rate, cap)
    else:
        rate = blink_rate(light.token_rate, cap)
    return del_color(state, rate, t, light.prompt_advancing, compaction)


def power_state(lights: Mapping[int, Light]) -> LlamaState:
    """One summary for the power button: reading if any llama light reads a prompt,
    up if any light has a server to talk to."""
    inference = [light for light in lights.values() if isinstance(light, InferenceInput)]
    if any(light.state.reading for light in inference):
        return LlamaState(up=True, processing=True, decoded=0)
    up = (
        any(light.state.up for light in inference)
        or any(light.state.up for light in lights.values() if isinstance(light, OllamaInput))
        or any(light.state.up for light in lights.values() if isinstance(light, GpuInput))
    )
    return LlamaState(up=up, processing=False, decoded=0)


def build_frame(
    inputs: Inputs,
    t: float,
    base: Color,
    del_cap: float = 20.0,
    esc_cap: float = 10.0,
    boost: float = 1.0,
    compact_tokens: float = float("inf"),
    compact_blink: float = 10.0,
) -> dict[int, Color]:
    """boost lifts ESC and the inference lights only: the backlight level is global, so in
    dim mode they are the keys that would otherwise vanish. The default compact_tokens
    never triggers, so callers opt in to the compaction state. Only configured lights
    appear in the frame; the other keys keep the keyboard's own profile."""
    esc_key = esc_color(net_blink_rate(inputs.net_bytes_per_s, esc_cap), t, inputs.online)
    frame = {
        key_id("ESC"): boost_color(esc_key, boost),
        key_id("F5"): mute_color(inputs.muted, base),
    }
    for key, light in inputs.lights.items():
        if isinstance(light, InferenceInput):
            color = _inference_color(light, t, del_cap, compact_tokens, compact_blink)
        elif isinstance(light, GpuInput):
            color = gpu_color(light.state, t, del_cap)
        else:
            color = ollama_color(light.state, light.pulse, t, light.event_rate, del_cap)
        frame[key] = boost_color(color, boost)
    temp = base if inputs.temp_c is None else temp_color(inputs.temp_c)
    frame.update({key: temp for key in TEMP_KEYS})
    return frame
