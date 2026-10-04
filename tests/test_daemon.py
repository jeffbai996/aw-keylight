from keylights.config import load_config
from keylights.daemon import Daemon
from keylights.device import DeviceError
from keylights.effects import AMBER, BABY_BLUE, GREEN, NET_IDLE, WHITE
from keylights.sources import GPU_DOWN, GpuState, LlamaState, OllamaState

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
        self.light_values = {}  # per-light overrides; DEL falls back to llama_value

    def backlight_state(self):
        return self._get(self.backlight_value)

    def _get(self, value):
        if isinstance(value, Exception):
            raise value
        return value

    def inference(self, name):
        # Like the real readers, an Ollama light answers with an Ollama state.
        default = OllamaState(up=True, loaded=False, touched=None) if name in ("HOME", "F12") else self.llama_value
        return self._get(self.light_values.get(name, default))

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


def make_daemon(env=None):
    # The legacy single URL makes DEL the inference light, as before.
    config = load_config(
        {"KEYLIGHTS_BASE_COLOR": "ffffff", "KEYLIGHTS_POLL_INTERVAL": "0", "KEYLIGHTS_LLAMA_URL": "http://fake.example", **(env or {})}
    )
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


def test_unanswered_poll_before_any_reading_shows_the_server_idle():
    daemon, sources, keyboard, _ = make_daemon()
    sources.llama_value = LlamaState(up=True, processing=False, decoded=0, answered=False)
    daemon.tick(now=0.0)
    assert keyboard.frames[-1][15] == BABY_BLUE


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


def test_del_goes_baby_blue_when_the_large_job_ends():
    daemon, sources, keyboard, _ = make_daemon()
    sources.llama_value = compaction_reading()
    daemon.tick(now=0.0)
    sources.llama_value = UP_IDLE
    assert collect_del(daemon, keyboard, 1.0, 1.3) == {BABY_BLUE}


LIGHTS_ENV = {"KEYLIGHTS_LIGHTS": "F12=ollama:http://a.example,HOME=ollama:http://b.example,END=llama:http://c.example,DEL=llama:http://d.example"}
F12, HOME, END, DEL = 12, 13, 14, 15


def test_each_light_reads_its_own_source():
    daemon, sources, keyboard, _ = make_daemon(LIGHTS_ENV)
    sources.light_values = {"END": reading(0), "DEL": UP_IDLE, "HOME": OllamaState(True, True, 100.0), "F12": OllamaState(True, False, None)}
    daemon.tick(now=0.0)
    frame = keyboard.frames[-1]
    assert frame[END] == AMBER and frame[DEL] == BABY_BLUE and frame[HOME] == BABY_BLUE and frame[F12] == BABY_BLUE


def test_prompt_progress_blinks_only_the_light_whose_prompt_is_advancing():
    daemon, sources, keyboard, _ = make_daemon(LIGHTS_ENV)
    sources.light_values = {"END": reading(0), "DEL": reading(0)}
    daemon.tick(now=0.0)
    sources.light_values = {"END": reading(2048), "DEL": reading(0)}
    first = len(keyboard.frames)
    t = 1.0
    while t <= 1.5:
        daemon.tick(now=t)
        t += 0.01
    frames = keyboard.frames[first:]
    assert {f[END] for f in frames} == {AMBER, (0, 0, 0)}
    assert {f[DEL] for f in frames} == {AMBER}


def test_an_unanswered_poll_keeps_only_that_lights_last_state():
    daemon, sources, keyboard, _ = make_daemon(LIGHTS_ENV)
    sources.light_values = {"END": reading(0), "DEL": reading(0)}
    daemon.tick(now=0.0)
    sources.light_values = {
        "END": LlamaState(up=True, processing=False, decoded=0, answered=False),
        "DEL": UP_IDLE,
    }
    daemon.tick(now=0.5)
    frame = keyboard.frames[-1]
    assert frame[END] == AMBER and frame[DEL] == BABY_BLUE


def test_ollama_light_pulses_white_after_a_request_finishes_then_settles():
    daemon, sources, keyboard, _ = make_daemon(LIGHTS_ENV)
    sources.light_values = {"HOME": OllamaState(True, True, 100.0)}
    daemon.tick(now=0.0)
    assert keyboard.frames[-1][HOME] == BABY_BLUE

    sources.light_values = {"HOME": OllamaState(True, True, 101.0)}  # expiry moved: a request finished
    daemon.tick(now=1.0)
    assert keyboard.frames[-1][HOME] == WHITE
    daemon.tick(now=3.0)
    assert keyboard.frames[-1][HOME] == BABY_BLUE


def test_ollama_light_does_not_pulse_when_a_model_first_appears():
    daemon, sources, keyboard, _ = make_daemon(LIGHTS_ENV)
    sources.light_values = {"HOME": OllamaState(True, False, None)}
    daemon.tick(now=0.0)
    sources.light_values = {"HOME": OllamaState(True, True, 500.0)}
    daemon.tick(now=1.0)
    assert keyboard.frames[-1][HOME] == BABY_BLUE


def test_a_light_whose_source_fails_goes_dark_without_stopping_the_others():
    daemon, sources, keyboard, _ = make_daemon(LIGHTS_ENV)
    sources.light_values = {"END": OSError("down"), "HOME": OSError("down"), "DEL": UP_IDLE}
    daemon.tick(now=0.0)
    frame = keyboard.frames[-1]
    assert frame[END] == (0, 0, 0) and frame[HOME] == (0, 0, 0) and frame[DEL] == BABY_BLUE


def test_shutdown_restores_the_status_keys_and_every_configured_light():
    daemon, _, keyboard, power = make_daemon(LIGHTS_ENV)
    daemon.tick(now=0.0)
    daemon.shutdown()
    assert keyboard.frames[-1] == {k: BASE for k in (0, 1, 2, 3, 4, 5, F12, HOME, END, DEL)}
    assert power.restored


def test_power_button_follows_any_light_reading_a_prompt():
    daemon, sources, _, power = make_daemon(LIGHTS_ENV)
    sources.light_values = {"END": reading(0), "DEL": UP_IDLE}
    daemon.tick(now=0.0)
    assert power.states[-1].reading


def gate_activity(in_flight=0, awaiting=0, events=0):
    return OllamaState(True, True, 100.0, True, in_flight, awaiting, events)


def test_ollama_light_blinks_amber_while_a_request_waits_for_its_first_event():
    daemon, sources, keyboard, _ = make_daemon(LIGHTS_ENV)
    sources.light_values = {"HOME": gate_activity(in_flight=1, awaiting=1, events=0)}
    daemon.tick(now=0.0)
    seen = set()
    t = 0.0
    while t < 0.5:
        daemon.tick(now=t)
        seen.add(keyboard.frames[-1][HOME])
        t += 0.01
    assert seen == {AMBER, (0, 0, 0)}


def test_ollama_light_flicker_follows_the_event_counter():
    daemon, sources, keyboard, _ = make_daemon({**LIGHTS_ENV, "KEYLIGHTS_POLL_INTERVAL": "0.5"})
    sources.light_values = {"HOME": gate_activity(in_flight=1, events=100)}
    daemon.tick(now=0.0)
    sources.light_values = {"HOME": gate_activity(in_flight=1, events=140)}  # 40 events in 0.5 s
    colors = set()
    t = 0.5
    while t < 1.0:
        daemon.tick(now=t)
        colors.add(keyboard.frames[-1][HOME])
        t += 0.01
    assert colors == {GREEN, (0, 0, 0)}


def test_an_event_counter_that_restarts_reads_as_no_new_events_not_a_burst():
    daemon, sources, keyboard, _ = make_daemon({**LIGHTS_ENV, "KEYLIGHTS_POLL_INTERVAL": "0.5"})
    sources.light_values = {"HOME": gate_activity(in_flight=1, events=5000)}
    daemon.tick(now=0.0)
    sources.light_values = {"HOME": gate_activity(in_flight=1, events=3)}  # the gate restarted
    colors = set()
    t = 0.5
    while t < 0.9:
        daemon.tick(now=t)
        colors.add(keyboard.frames[-1][HOME])
        t += 0.01
    assert colors == {GREEN}


def test_del_goes_green_once_a_large_prompt_has_been_read_and_tokens_flow():
    daemon, sources, keyboard, _ = make_daemon({"KEYLIGHTS_POLL_INTERVAL": "0.5"})
    sources.llama_value = compaction_reading(15000)
    daemon.tick(now=0.0)
    sources.llama_value = LlamaState(up=True, processing=True, decoded=40, prompt_new=30000, prompt_done=30000)
    colors = set()
    t = 0.5
    while t < 0.9:
        daemon.tick(now=t)
        colors.add(keyboard.frames[-1][15])
        t += 0.01
    assert AMBER not in colors and GREEN in colors


GPU_ENV = {"KEYLIGHTS_LIGHTS": "HOME=gpu:http://s.example/squad#host-a,END=llama:http://c.example"}
CARD_IDLE = GpuState(up=True, power_w=37.0, limit_w=350.0, pstate="P8")
CARD_BUSY = GpuState(up=True, power_w=140.0, limit_w=350.0, pstate="P2")


def collect_key(daemon, keyboard, key, start, stop, step=0.01):
    seen = set()
    t = start
    while t <= stop:
        daemon.tick(now=t)
        seen.add(keyboard.frames[-1][key])
        t += step
    return seen


def test_gpu_light_is_baby_blue_while_the_card_idles_and_flickers_when_it_works():
    daemon, sources, keyboard, _ = make_daemon(GPU_ENV)
    sources.light_values = {"HOME": CARD_IDLE}
    assert collect_key(daemon, keyboard, HOME, 0.0, 0.5) == {BABY_BLUE}
    sources.light_values = {"HOME": CARD_BUSY}
    assert collect_key(daemon, keyboard, HOME, 1.0, 1.5) == {GREEN, (0, 0, 0)}


def test_gpu_light_goes_dark_when_its_source_fails_without_stopping_the_other_lights():
    daemon, sources, keyboard, _ = make_daemon(GPU_ENV)
    sources.light_values = {"HOME": OSError("telemetry down"), "END": UP_IDLE}
    daemon.tick(now=0.0)
    frame = keyboard.frames[-1]
    assert frame[HOME] == (0, 0, 0) and frame[END] == BABY_BLUE


def test_an_unanswered_gpu_poll_keeps_the_last_reading():
    daemon, sources, keyboard, _ = make_daemon(GPU_ENV)
    sources.light_values = {"HOME": CARD_BUSY}
    daemon.tick(now=0.0)
    sources.light_values = {"HOME": GpuState(up=True, power_w=None, limit_w=None, pstate=None, answered=False)}
    assert collect_key(daemon, keyboard, HOME, 0.5, 1.0) == {GREEN, (0, 0, 0)}


def test_parked_hosts_show_baby_blue_and_unreachable_ones_stay_dark():
    daemon, sources, keyboard, _ = make_daemon(LIGHTS_ENV)
    sources.light_values = {
        "END": LlamaState(up=True, processing=False, decoded=0, loaded=False),   # router up, model parked
        "HOME": OllamaState(True, False, None),                                 # ollama up, nothing loaded
        "F12": OSError("ollama stopped"),                                       # gamemode: ollama is not running
        "DEL": OSError("server down"),
    }
    daemon.tick(now=0.0)
    frame = keyboard.frames[-1]
    assert frame[END] == BABY_BLUE and frame[HOME] == BABY_BLUE
    assert frame[F12] == (0, 0, 0) and frame[DEL] == (0, 0, 0)


def test_an_ollama_host_that_stops_answering_keeps_its_light_briefly_then_goes_dark():
    daemon, sources, keyboard, _ = make_daemon(LIGHTS_ENV)
    sources.light_values = {"HOME": OllamaState(True, True, 100.0)}
    daemon.tick(now=0.0)
    sources.light_values = {"HOME": OllamaState(True, False, None, answered=False)}
    daemon.tick(now=2.0)
    assert keyboard.frames[-1][HOME] == BABY_BLUE  # a busy gate can be slow; the last reading stands
    daemon.tick(now=8.0)
    assert keyboard.frames[-1][HOME] == (0, 0, 0)  # silent for 6 s: ollama is stopped (gamemode)
    daemon.tick(now=10.0)
    assert keyboard.frames[-1][HOME] == (0, 0, 0)  # and it stays dark, not parked-blue, while silent


def test_an_answer_resets_the_ollama_grace_period():
    daemon, sources, keyboard, _ = make_daemon(LIGHTS_ENV)
    silent = {"HOME": OllamaState(True, False, None, answered=False)}
    sources.light_values = {"HOME": OllamaState(True, True, 100.0)}
    daemon.tick(now=0.0)
    sources.light_values = silent
    daemon.tick(now=3.0)
    sources.light_values = {"HOME": OllamaState(True, True, 100.0)}
    daemon.tick(now=4.0)
    sources.light_values = silent
    daemon.tick(now=5.0)
    daemon.tick(now=8.0)  # 3 s of silence since the last answer
    assert keyboard.frames[-1][HOME] == BABY_BLUE


def test_a_host_that_comes_back_after_gamemode_shows_baby_blue_parked_or_loaded():
    daemon, sources, keyboard, _ = make_daemon(LIGHTS_ENV)
    sources.light_values = {"HOME": OSError("ollama stopped")}
    daemon.tick(now=0.0)
    assert keyboard.frames[-1][HOME] == (0, 0, 0)
    sources.light_values = {"HOME": OllamaState(True, False, None)}  # ollama back, model not warmed yet
    daemon.tick(now=1.0)
    assert keyboard.frames[-1][HOME] == BABY_BLUE
    sources.light_values = {"HOME": OllamaState(True, True, 500.0)}
    daemon.tick(now=2.0)
    assert keyboard.frames[-1][HOME] == BABY_BLUE  # loaded and idle reads the same as parked


def test_an_ollama_light_with_a_gpu_overlay_speaks_the_same_language_as_the_others():
    daemon, sources, keyboard, _ = make_daemon(LIGHTS_ENV)
    parked_idle = OllamaState(True, False, None, gpu=CARD_IDLE)
    parked_working = OllamaState(True, False, None, gpu=CARD_BUSY)
    sources.light_values = {"HOME": parked_idle}
    assert collect_key(daemon, keyboard, HOME, 0.0, 0.5) == {BABY_BLUE}
    sources.light_values = {"HOME": parked_working}
    assert collect_key(daemon, keyboard, HOME, 1.0, 1.5) == {GREEN, (0, 0, 0)}
    sources.light_values = {"HOME": OllamaState(True, True, 100.0, gpu=CARD_IDLE)}
    assert collect_key(daemon, keyboard, HOME, 2.0, 2.5) == {BABY_BLUE}
