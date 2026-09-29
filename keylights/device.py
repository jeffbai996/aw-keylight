"""Per-key colour writes to the Darfon keyboard controller (0d62:0a1c).

Commands go over the USB control pipe as HID feature reports: the interrupt
endpoints on this controller do not respond. Sequence per update, as verified
on hardware: reset, colour blocks, loop (8C 13), commit (8B 01 FF).
"""
from __future__ import annotations

import fcntl
import os
from pathlib import Path
from typing import Protocol

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
    def __init__(self, transport: Transport):
        self._transport = transport
        self._last: dict[int, Color] = {}

    def update(self, frame: dict[int, Color]) -> None:
        changed = diff_frame(self._last, frame)
        if not changed:
            return
        reports = [_report(0x94), *build_color_packets(changed), _report(0x8C, 0x13), _report(0x8B, 0x01, 0xFF)]
        try:
            for report in reports:
                self._transport.send(report)
        except OSError as exc:
            raise DeviceError(f"Keyboard write failed: {exc}") from exc
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
