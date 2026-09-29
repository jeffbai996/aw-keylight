from keylights.config import load_config
from keylights.daemon import Daemon
from keylights.device import DeviceError
from keylights.sources import LlamaState

BASE = (255, 255, 255)
UP_IDLE = LlamaState(up=True, processing=False, decoded=0)


class FakeSources:
    def __init__(self):
        self.llama_value = UP_IDLE
        self.net_value = 0
        self.temp_value = 45.0
        self.mic_value = False

    def _get(self, value):
        if isinstance(value, Exception):
            raise value
        return value

    def llama(self):
        return self._get(self.llama_value)

    def net_bytes(self):
        return self._get(self.net_value)

    def cpu_temp(self):
        return self._get(self.temp_value)

    def mic_muted(self):
        return self._get(self.mic_value)


class FakeKeyboard:
    def __init__(self):
        self.frames = []
        self.fail_next = False

    def update(self, frame):
        if self.fail_next:
            self.fail_next = False
            raise DeviceError("usb write failed")
        self.frames.append(dict(frame))


class FakePower:
    def __init__(self):
        self.states = []
        self.restored = False

    def update(self, state, now):
        self.states.append(state)

    def restore(self):
        self.restored = True


def make_daemon():
    config = load_config({"KEYLIGHTS_BASE_COLOR": "ffffff", "KEYLIGHTS_POLL_INTERVAL": "0"})
    sources, keyboard, power = FakeSources(), FakeKeyboard(), FakePower()
    return Daemon(sources, keyboard, power, config), sources, keyboard, power


def test_loop_survives_one_source_failing():
    daemon, sources, keyboard, _ = make_daemon()
    sources.temp_value = RuntimeError("hwmon read failed")

    daemon.tick(now=0.0)

    (frame,) = keyboard.frames
    assert frame[1] == BASE  # F1 falls back to the base color
    assert frame[15] != BASE  # DEL still reflects the inference server


def test_server_failure_marks_del_off_and_power_button_down():
    daemon, sources, keyboard, power = make_daemon()
    sources.llama_value = OSError("connection refused")

    daemon.tick(now=0.0)

    assert keyboard.frames[0][15] == (0, 0, 0)
    assert power.states[0].up is False


def test_loop_survives_keyboard_write_error_and_retries():
    daemon, _, keyboard, _ = make_daemon()
    keyboard.fail_next = True

    daemon.tick(now=0.0)
    assert keyboard.frames == []

    daemon.tick(now=0.05)
    assert len(keyboard.frames) == 1


def test_shutdown_restores_previous_keyboard_lighting():
    daemon, _, keyboard, power = make_daemon()
    daemon.tick(now=0.0)

    daemon.shutdown()

    assert keyboard.frames[-1] == {k: BASE for k in (0, 1, 2, 3, 4, 5, 15)}
    assert power.restored is True
