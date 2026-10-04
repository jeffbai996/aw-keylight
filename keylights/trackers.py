"""Turns successive readings of one server into what its key shows. No I/O."""
from __future__ import annotations

from dataclasses import replace

from .effects import GpuInput, InferenceInput, OllamaInput
from .sources import OLLAMA_DOWN, GpuState, LlamaState, OllamaState, token_rate

# The server evaluates the prompt in batches and answers /slots between them, so
# progress arrives in lumps seconds apart. Blinking holds this long after the last one.
PROMPT_ADVANCE_HOLD_S = 6.0
# How long a light stays white after an Ollama request finishes.
PULSE_S = 0.4
# How long a silent Ollama host keeps its last reading. A stopped Ollama (gamemode) makes the gate
# hang on a WSL host with mirrored networking instead of refusing, so silence has to end in "dark".
OLLAMA_SILENCE_GRACE_S = 5.0


class LlamaTracker:
    """Keeps one llama.cpp light's counters: token rate, prompt rate and the last good reading."""

    def __init__(self) -> None:
        self._last: InferenceInput | None = None
        self._prev_decoded = 0
        self._prompt_done: int | None = None
        self._prompt_mark = 0.0  # when the prompt counter last moved, or reading began
        self._advanced_at: float | None = None
        self._prompt_rate = 0.0

    def update(self, state: LlamaState, now: float, dt: float) -> InferenceInput:
        last = self._last
        if state.up and not state.answered and last is not None and last.state.up:
            # A busy server stops answering /slots; its last reading is the best evidence.
            state, rate = last.state, last.token_rate
        else:
            rate = token_rate(self._prev_decoded, state.decoded, dt) if state.processing else 0.0
            self._prev_decoded = state.decoded
            self._track_prompt(state, now)
        advancing = self._advanced_at is not None and now - self._advanced_at <= PROMPT_ADVANCE_HOLD_S
        self._last = InferenceInput(
            state=state,
            token_rate=rate,
            prompt_rate=self._prompt_rate if state.reading else 0.0,
            prompt_advancing=state.reading and advancing,
        )
        return self._last

    def _track_prompt(self, state: LlamaState, now: float) -> None:
        done = state.prompt_done
        if not state.reading or done is None:
            self._prompt_done, self._advanced_at, self._prompt_rate = None, None, 0.0
            return
        if self._prompt_done is None or done < self._prompt_done:  # reading began, or a new request
            self._prompt_done, self._prompt_mark, self._advanced_at, self._prompt_rate = done, now, None, 0.0
        elif done > self._prompt_done:
            elapsed = now - self._prompt_mark
            if elapsed > 0:
                sample = (done - self._prompt_done) / elapsed
                self._prompt_rate = sample if self._prompt_rate == 0 else (self._prompt_rate + sample) / 2
            self._prompt_done, self._prompt_mark, self._advanced_at = done, now, now


class OllamaTracker:
    """Detects finished requests from the model expiry moving forward, turns the gate's
    event counter into a rate, and decides when a silent host counts as gone."""

    def __init__(self) -> None:
        self._last: OllamaInput | None = None
        self._touched: float | None = None
        self._events: int | None = None
        self._pulse_until = 0.0
        self._silent_since: float | None = None

    def update(self, state: OllamaState, now: float, dt: float = 0.0) -> OllamaInput:
        last = self._last
        if state.up and not state.answered:
            # A busy gate can be slow, so a silent host keeps its last reading for a moment. Past
            # that it is treated as unreachable: with Ollama stopped the gate does not refuse, it hangs.
            if self._silent_since is None:
                self._silent_since = now
            if last is not None and last.state.up and now - self._silent_since < OLLAMA_SILENCE_GRACE_S:
                state, rate = replace(last.state, gpu=state.gpu), last.event_rate
            else:
                # The latest GPU reading is kept: it says whether the silence is gamemode or an outage.
                state, rate = replace(OLLAMA_DOWN, gpu=state.gpu), 0.0
                self._touched = self._events = None
        else:
            self._silent_since = None
            # A model that has just appeared has no earlier expiry to compare, so it does not pulse.
            if state.touched is not None and self._touched is not None and state.touched > self._touched:
                self._pulse_until = now + PULSE_S
            self._touched = state.touched
            # A counter that went backwards means the gate restarted: no rate, not a burst.
            grew = state.events is not None and self._events is not None and state.events >= self._events
            rate = (state.events - self._events) / dt if grew and dt > 0 else 0.0
            self._events = state.events
        self._last = OllamaInput(state=state, pulse=now < self._pulse_until, event_rate=rate)
        return self._last


class GpuTracker:
    """A GPU reading is already instantaneous, so only a busy source needs handling."""

    def __init__(self) -> None:
        self._last: GpuInput | None = None

    def update(self, state: GpuState, now: float, dt: float = 0.0) -> GpuInput:
        last = self._last
        if state.up and not state.answered and last is not None and last.state.up:
            return last  # a busy source stops answering; keep what it last said
        self._last = GpuInput(state=state)
        return self._last
