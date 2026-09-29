from keylights.effects import AMBER, GREEN, OFF
from keylights.elc import PowerButton, build_status_program, power_color
from keylights.sources import LlamaState

UP_IDLE = LlamaState(up=True, processing=False, decoded=0)
READING = LlamaState(up=True, processing=True, decoded=0)
DOWN = LlamaState(up=False, processing=False, decoded=0)


class FakeElc:
    def __init__(self):
        self.sent = []

    def command(self, *payload):
        self.sent.append(payload)


def test_power_button_program_is_start_series_action_play():
    assert build_status_program(zone=1, color=(0, 255, 0)) == [
        (0x21, 0x00, 0x01, 0xFF, 0xFF),
        (0x23, 0x01, 0x00, 0x01, 1),
        (0x24, 0x00, 0x07, 0xD0, 0x00, 0xFA, 0, 255, 0),
        (0x21, 0x00, 0x03, 0xFF, 0xFF),
    ]


def test_power_button_solid_amber_off_map_from_server_state():
    assert power_color(UP_IDLE) == GREEN
    assert power_color(READING) == AMBER
    assert power_color(DOWN) == OFF


def test_power_button_reprogrammed_only_when_state_changes():
    elc = FakeElc()
    button = PowerButton(elc, zone=1, min_interval=2.0)
    button.update(UP_IDLE, now=0.0)
    programmed = len(elc.sent)
    assert programmed == 4

    button.update(UP_IDLE, now=10.0)

    assert len(elc.sent) == programmed


def test_power_button_change_inside_min_interval_is_deferred_then_applied():
    elc = FakeElc()
    button = PowerButton(elc, zone=1, min_interval=2.0)
    button.update(UP_IDLE, now=0.0)
    elc.sent.clear()

    button.update(READING, now=0.5)
    assert elc.sent == []

    button.update(READING, now=2.5)
    assert elc.sent[2][-3:] == AMBER


def test_power_button_restore_sets_solid_green():
    elc = FakeElc()
    button = PowerButton(elc, zone=1, min_interval=2.0)
    button.update(DOWN, now=0.0)
    elc.sent.clear()

    button.restore()

    assert elc.sent[2][-3:] == GREEN
