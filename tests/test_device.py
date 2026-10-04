import pytest

from keylights.device import (
    ControllerNotFound,
    DeviceError,
    Keyboard,
    build_color_packets,
    diff_frame,
    find_hidraw,
)

RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)


class FakeTransport:
    def __init__(self, fail_after=None):
        self.sent = []
        self.fail_after = fail_after

    def send(self, report):
        if self.fail_after is not None and len(self.sent) >= self.fail_after:
            raise OSError("usb write failed")
        self.sent.append(bytes(report))

    def color_packets(self):
        return [r for r in self.sent if r[1] == 0x8C and r[2] == 0x02]


def test_color_packet_uses_key_id_plus_one():
    (packet,) = build_color_packets({15: RED})
    assert len(packet) == 64
    assert packet[:8] == bytes([0xCC, 0x8C, 0x02, 0x00, 16, 255, 0, 0])
    assert packet[8:] == bytes(56)


def test_more_than_15_keys_are_split_across_packets():
    packets = build_color_packets({k: RED for k in range(16)})
    assert len(packets) == 2
    assert packets[1][4:8] == bytes([16, 255, 0, 0])
    assert packets[1][8:] == bytes(56)


def test_diff_frame_keeps_only_changed_keys():
    changed = diff_frame({0: RED, 15: GREEN}, {0: RED, 15: BLUE, 3: RED})
    assert changed == {15: BLUE, 3: RED}


def test_update_sends_only_keys_whose_color_changed():
    transport = FakeTransport()
    keyboard = Keyboard(transport)
    keyboard.update({15: RED, 0: BLUE})
    transport.sent.clear()

    keyboard.update({15: RED, 0: GREEN})

    (packet,) = transport.color_packets()
    assert packet[4:8] == bytes([1, 0, 255, 0])  # ESC is id 0 -> wire id 1
    assert packet[8:12] == bytes(4)


def test_update_with_no_changes_sends_nothing():
    transport = FakeTransport()
    keyboard = Keyboard(transport)
    keyboard.update({15: RED})
    transport.sent.clear()

    keyboard.update({15: RED})

    assert transport.sent == []


def test_update_wraps_color_packets_in_reset_loop_and_commit():
    transport = FakeTransport()
    Keyboard(transport).update({15: RED})

    assert [r[1] for r in transport.sent] == [0x94, 0x8C, 0x8C, 0x8B]
    assert transport.sent[2][2] == 0x13
    assert transport.sent[3][2:4] == bytes([0x01, 0xFF])


def test_transport_oserror_becomes_device_error():
    keyboard = Keyboard(FakeTransport(fail_after=0))
    with pytest.raises(DeviceError):
        keyboard.update({15: RED})


def test_failed_write_is_retried_on_next_update():
    transport = FakeTransport(fail_after=1)
    keyboard = Keyboard(transport)
    with pytest.raises(DeviceError):
        keyboard.update({15: RED})

    transport.fail_after = None
    transport.sent.clear()
    keyboard.update({15: RED})

    assert len(transport.color_packets()) == 1


def test_missing_controller_raises_clear_error(tmp_path):
    with pytest.raises(ControllerNotFound, match="0d62:0a1c"):
        find_hidraw(tmp_path)


def test_find_hidraw_matches_vid_pid_in_uevent(tmp_path):
    for name, hid_id in (("hidraw2", "0003:0000187C:00000550"), ("hidraw3", "0003:00000D62:00000A1C")):
        device = tmp_path / name / "device"
        device.mkdir(parents=True)
        (device / "uevent").write_text(f"DRIVER=hid-generic\nHID_ID={hid_id}\n")

    assert find_hidraw(tmp_path) == "/dev/hidraw3"


@pytest.mark.parametrize("fail_after, step", [(0, "reset"), (1, "colour"), (2, "loop"), (3, "update")])
def test_a_failed_write_names_the_step_that_failed(fail_after, step):
    keyboard = Keyboard(FakeTransport(fail_after=fail_after))
    with pytest.raises(DeviceError, match=f"at {step}"):
        keyboard.update({15: RED})


def test_a_failed_write_says_how_long_it_blocked():
    times = iter([100.0, 105.2])  # start of the update, then the moment the write gave up
    keyboard = Keyboard(FakeTransport(fail_after=1), clock=lambda: next(times))
    with pytest.raises(DeviceError, match=r"after 5\.2s"):
        keyboard.update({15: RED})


def resets(transport):
    return [r for r in transport.sent if r[1] == 0x94]


def test_reset_is_sent_with_the_first_update_and_not_with_every_one():
    transport = FakeTransport()
    keyboard = Keyboard(transport, clock=lambda: 0.0, reset_interval=60.0)
    keyboard.update({15: RED})
    keyboard.update({15: GREEN})
    keyboard.update({15: BLUE})
    assert len(resets(transport)) == 1 and transport.sent[0][1] == 0x94
    assert len(transport.color_packets()) == 3  # every colour change still goes out


def test_reset_is_sent_again_once_the_interval_has_passed():
    now = [0.0]
    transport = FakeTransport()
    keyboard = Keyboard(transport, clock=lambda: now[0], reset_interval=60.0)
    keyboard.update({15: RED})
    now[0] = 30.0
    keyboard.update({15: GREEN})
    assert len(resets(transport)) == 1
    now[0] = 61.0
    keyboard.update({15: BLUE})
    assert len(resets(transport)) == 2


def test_reset_is_sent_again_after_a_failed_write_to_resync_the_controller():
    class FlakyTransport(FakeTransport):
        def __init__(self):
            super().__init__()
            self.fail_next_color = False

        def send(self, report):
            if self.fail_next_color and report[1] == 0x8C and report[2] == 0x02:
                self.fail_next_color = False
                raise OSError("stalled")
            super().send(report)

    transport = FlakyTransport()
    keyboard = Keyboard(transport, clock=lambda: 0.0, reset_interval=60.0)
    keyboard.update({15: RED})
    transport.fail_next_color = True
    with pytest.raises(DeviceError):
        keyboard.update({15: GREEN})
    keyboard.update({15: GREEN})  # the retry
    assert len(resets(transport)) == 2  # the first update, and the retry after the failure


def test_a_zero_interval_resets_on_every_update_as_before():
    transport = FakeTransport()
    keyboard = Keyboard(transport, clock=lambda: 0.0, reset_interval=0.0)
    keyboard.update({15: RED})
    keyboard.update({15: GREEN})
    assert len(resets(transport)) == 2


def commands(transport):
    return [(r[1], r[2]) for r in transport.sent]


def test_an_update_between_resets_sends_only_the_colour_and_the_commit():
    transport = FakeTransport()
    keyboard = Keyboard(transport, clock=lambda: 0.0, reset_interval=60.0)
    keyboard.update({15: RED})
    transport.sent.clear()
    keyboard.update({15: GREEN})
    assert commands(transport) == [(0x8C, 0x02), (0x8B, 0x01)]  # colour block, then update: no reset, no loop


def test_the_update_that_carries_a_reset_is_the_full_sequence():
    transport = FakeTransport()
    keyboard = Keyboard(transport, clock=lambda: 0.0, reset_interval=60.0)
    keyboard.update({15: RED})
    assert commands(transport) == [(0x94, 0x00), (0x8C, 0x02), (0x8C, 0x13), (0x8B, 0x01)]


def test_a_zero_interval_keeps_the_full_sequence_on_every_update():
    transport = FakeTransport()
    keyboard = Keyboard(transport, clock=lambda: 0.0, reset_interval=0.0)
    keyboard.update({15: RED})
    transport.sent.clear()
    keyboard.update({15: GREEN})
    assert commands(transport) == [(0x94, 0x00), (0x8C, 0x02), (0x8C, 0x13), (0x8B, 0x01)]


def test_a_stall_in_a_light_update_names_the_commit_step():
    transport = FakeTransport()
    keyboard = Keyboard(transport, clock=lambda: 0.0, reset_interval=60.0)
    keyboard.update({15: RED})
    transport.fail_after = len(transport.sent) + 1  # let the colour block through, fail the next report
    with pytest.raises(DeviceError, match="at update"):
        keyboard.update({15: GREEN})
