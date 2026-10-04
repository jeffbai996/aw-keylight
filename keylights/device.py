"""Per-key colour writes to the Darfon keyboard controller (0d62:0a1c).

Commands go over the USB control pipe as HID feature reports: the interrupt
endpoints on this controller do not respond. Sequence per update: [reset,] colour
blocks, loop (8C 13), commit (8B 01 FF). Reset is sent on a schedule, see Keyboard.
"""
from __future__ import annotations

import fcntl
import os
import time
from pathlib import Path
from typing import Callable, Protocol

from .config import Color

REPORT_LEN = 64
BLOCKS_PER_PACKET = 15
HID_ID_MARKER = "00000D62:00000A1C"
SYSFS_HIDRAW = Path("/sys/class/hidraw")


class DeviceError(Exception):
    pass


class ControllerNotFound(DeviceError):
    pass


class Transport(Protocol):
    def send(self, report: bytes) -> None: ...


def _report(*payload: int) -> bytes:
    return bytes([0xCC, *payload]).ljust(REPORT_LEN, b"\x00")


def build_color_packets(colors: dict[int, Color]) -> list[bytes]:
    """One packet per 15 keys; each block is [key id + 1, r, g, b] from byte 4."""
    items = list(colors.items())
    packets = []
    for start in range(0, len(items), BLOCKS_PER_PACKET):
        payload = [0x8C, 0x02, 0x00]
        for key, (r, g, b) in items[start : start + BLOCKS_PER_PACKET]:
            payload += [key + 1, r, g, b]
        packets.append(_report(*payload))
    return packets


def diff_frame(previous: dict[int, Color], new: dict[int, Color]) -> dict[int, Color]:
    return {key: color for key, color in new.items() if previous.get(key) != color}


class Keyboard:
    """reset_interval is how often to send the reset command: at the first update, after any failed
    write (to resync the controller), and then at most this many seconds apart. Zero resets on
    every update, as before. Colour changes do not need it: unchanged keys keep their colour, so
    reset cannot be clearing them, and every transfer saved is one that cannot stall."""

    def __init__(self, transport: Transport, clock: Callable[[], float] = time.monotonic, reset_interval: float = 60.0):
        self._transport = transport
        self._clock = clock
        self._reset_interval = reset_interval
        self._last_reset: float | None = None  # None: the controller's state is not known
        self._last: dict[int, Color] = {}

    def update(self, frame: dict[int, Color]) -> None:
        changed = diff_frame(self._last, frame)
        if not changed:
            return
        started = self._clock()
        reset = (
            self._last_reset is None
            or self._reset_interval <= 0
            or started - self._last_reset >= self._reset_interval
        )
        steps = [("reset", _report(0x94))] if reset else []
        steps += [("colour", packet) for packet in build_color_packets(changed)]
        steps += [("loop", _report(0x8C, 0x13)), ("update", _report(0x8B, 0x01, 0xFF))]
        try:
            for step, report in steps:
                self._transport.send(report)
        except OSError as exc:
            self._last_reset = None  # the controller may be mid-sequence: start the next one clean
            # A control transfer that gets no answer blocks for seconds. Which step it was, and for
            # how long, is what tells a stalled controller from a slow one.
            raise DeviceError(f"Keyboard write failed at {step} after {self._clock() - started:.1f}s: {exc}") from exc
        if reset:
            self._last_reset = started
        # Recorded only after a full write, so a failed update is retried in full.
        self._last.update(changed)


def find_hidraw(sysfs_root: Path = SYSFS_HIDRAW) -> str:
    for node in sorted(sysfs_root.glob("hidraw*")):
        try:
            if HID_ID_MARKER in (node / "device" / "uevent").read_text().upper():
                return "/dev/" + node.name
        except OSError:
            continue
    raise ControllerNotFound("Keyboard controller 0d62:0a1c not found.")


def _hidiocsfeature(length: int) -> int:
    # Linux _IOC(read|write, 'H', 0x06, length)
    return (3 << 30) | (length << 16) | (ord("H") << 8) | 0x06


class HidrawTransport:
    def __init__(self, path: str | None = None):
        self._fd = os.open(path or find_hidraw(), os.O_RDWR)

    def send(self, report: bytes) -> None:
        fcntl.ioctl(self._fd, _hidiocsfeature(len(report)), bytearray(report))

    def close(self) -> None:
        os.close(self._fd)
