"""Power button status light on the AlienFX chip (187c:0550).

This chip has jammed under heavy write load on other Alienware models, so the
button is reprogrammed only when its state changes, and never more often than
min_interval. It shows a steady colour; it is not used for fast blinking.
"""
from __future__ import annotations

import fcntl
import glob
import os
from typing import Protocol

from .config import Color
from .device import DeviceError
from .effects import AMBER, GREEN, OFF
from .sources import LlamaState

STEP_MS = 2000
HID_ID_MARKER = "0000187C:00000550"
REPORT_LEN = 34


class ElcTransport(Protocol):
    def command(self, *payload: int) -> object: ...


def build_status_program(zone: int, color: Color) -> list[tuple[int, ...]]:
    """Start, loop a series on one zone, add a single static step, play."""
    r, g, b = color
    return [
        (0x21, 0x00, 0x01, 0xFF, 0xFF),
        (0x23, 0x01, 0x00, 0x01, zone),
        (0x24, 0x00, STEP_MS >> 8, STEP_MS & 0xFF, 0x00, 0xFA, r, g, b),
        (0x21, 0x00, 0x03, 0xFF, 0xFF),
    ]


def power_color(state: LlamaState) -> Color:
    if not state.up:
        return OFF
    return AMBER if state.reading else GREEN


class PowerButton:
    def __init__(self, transport: ElcTransport, zone: int, min_interval: float = 2.0):
        self._transport = transport
        self._zone = zone
        self._min_interval = min_interval
        self._color: Color | None = None
        self._last_sent: float | None = None

    def update(self, state: LlamaState, now: float) -> None:
        color = power_color(state)
        if color == self._color:
            return
        if self._last_sent is not None and now - self._last_sent < self._min_interval:
            return  # the caller keeps calling; the change lands once the interval passes
        self._program(color, now)

    def restore(self) -> None:
        self._program(GREEN, None)

    def _program(self, color: Color, now: float | None) -> None:
        self._last_sent = now
        try:
            for command in build_status_program(self._zone, color):
                self._transport.command(*command)
        except OSError as exc:
            raise DeviceError(f"Power button write failed: {exc}") from exc
        self._color = color


def _ioc(nr: int) -> int:
    return (3 << 30) | (REPORT_LEN << 16) | (ord("H") << 8) | nr


class HidrawElcTransport:
    """Sends a command over the control pipe and reads its ack, as the vendor tool does."""

    def __init__(self):
        nodes = [
            "/dev/" + path.split("/")[4]
            for path in glob.glob("/sys/class/hidraw/*/device/uevent")
            if HID_ID_MARKER in open(path).read().upper()
        ]
        if not nodes:
            raise DeviceError("AlienFX controller 187c:0550 not found.")
        self._fd = os.open(nodes[0], os.O_RDWR)

    def command(self, *payload: int) -> bytes:
        out = bytearray(REPORT_LEN)
        out[1 : 2 + len(payload)] = bytes([0x03, *payload])
        fcntl.ioctl(self._fd, _ioc(0x0B), out)
        ack = bytearray(REPORT_LEN)
        fcntl.ioctl(self._fd, _ioc(0x0A), ack)
        return bytes(ack[1:8])

    def close(self) -> None:
        os.close(self._fd)
