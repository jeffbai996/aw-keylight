from keylights.config import load_config
from keylights.daemon import Daemon
from keylights.device import DeviceError
from keylights.effects import AMBER, GREEN, NET_IDLE
from keylights.sources import LlamaState

BASE = (255, 255, 255)
UP_IDLE = LlamaState(up=True, processing=False, decoded=0)


def reading(done):
    return LlamaState(up=True, processing=True, decoded=0, prompt_new=4000, prompt_done=done)


class FakeSources:
    def __init__(self):
        self.llama_value = UP_IDLE
        self.net_value = 0
        self.temp_value = 45.0
        self.mic_value = False
        self.backlight_value = None

    def backlight_state(self):
        return self._get(self.backlight_value)

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


def collect_del(daemon, keyboard, start, stop, step=0.01):
    first = len(keyboard.frames)
    t = start
    while t <= stop:
        daemon.tick(now=t)
        t += step
    return {frame[15] for frame in keyboard.frames[first:]}


def test_del_blinks_amber_while_prompt_progress_is_advancing():
    daemon, sources, keyboard, _ = make_daemon()
    sources.llama_value = reading(0)
    daemon.tick(now=0.0)
    sources.llama_value = reading(2048)
    assert collect_del(daemon, keyboard, 1.0, 1.5) == {AMBER, (0, 0, 0)}


def test_del_goes_solid_amber_when_prompt_progress_stalls_past_the_hold():
    daemon, sources, keyboard, _ = make_daemon()
    sources.llama_value = reading(0)
    daemon.tick(now=0.0)
    sources.llama_value = reading(2048)
    daemon.tick(now=1.0)
    assert collect_del(daemon, keyboard, 8.0, 8.5) == {AMBER}


def test_del_turns_solid_green_once_the_prompt_is_fully_read():
    daemon, sources, keyboard, _ = make_daemon()
    sources.llama_value = reading(0)
    daemon.tick(now=0.0)
    sources.llama_value = reading(4000)
    assert collect_del(daemon, keyboard, 1.0, 1.5) == {GREEN}


def test_unanswered_slots_poll_keeps_the_last_known_state():
    daemon, sources, keyboard, _ = make_daemon()
    sources.llama_value = reading(0)
    daemon.tick(now=0.0)
    sources.llama_value = LlamaState(up=True, processing=False, decoded=0, answered=False)
    daemon.tick(now=0.5)
    assert keyboard.frames[-1][15] == AMBER


def test_unanswered_poll_before_any_reading_leaves_the_server_up():
    daemon, sources, keyboard, _ = make_daemon()
    sources.llama_value = LlamaState(up=True, processing=False, decoded=0, answered=False)
    daemon.tick(now=0.0)
    assert keyboard.frames[-1][15] == GREEN


def test_esc_and_del_are_boosted_only_while_the_backlight_is_dim():
    daemon, sources, keyboard, _ = make_daemon()
    sources.backlight_value = "on"
    daemon.tick(now=0.0)
    on_esc = keyboard.frames[-1][0]
    assert on_esc == NET_IDLE

    sources.backlight_value = "dim"
    daemon.tick(now=0.1)
    dim_esc = keyboard.frames[-1][0]
    assert dim_esc == tuple(min(255, round(c * 1.6)) for c in NET_IDLE)
    assert keyboard.frames[-1][1] == keyboard.frames[0][1]  # F1 is not boosted


def test_unreadable_backlight_state_applies_no_boost():
    daemon, sources, keyboard, _ = make_daemon()
    sources.backlight_value = OSError("unreadable")
    daemon.tick(now=0.0)
    assert keyboard.frames[-1][0] == NET_IDLE


def compaction_reading(done=15000):
    return LlamaState(up=True, processing=True, decoded=0, prompt_new=30000, prompt_done=done)


def test_del_flashes_amber_once_the_prompt_is_compaction_sized():
    daemon, sources, keyboard, _ = make_daemon()
    sources.llama_value = compaction_reading()
    assert collect_del(daemon, keyboard, 0.0, 0.5) == {AMBER, (0, 0, 0)}


def test_del_returns_to_green_when_the_compaction_job_ends():
    daemon, sources, keyboard, _ = make_daemon()
    sources.llama_value = compaction_reading()
    daemon.tick(now=0.0)
    sources.llama_value = UP_IDLE
    assert collect_del(daemon, keyboard, 1.0, 1.3) == {GREEN}
