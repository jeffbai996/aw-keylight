import urllib.error

import pytest

from keylights.sources import (
    cache_path,
    GPU_DOWN,
    GpuState,
    LlamaState,
    OllamaState,
    cpu_temp_c,
    gpu_state,
    llama_state,
    net_rate,
    ollama_state,
    parse_connectivity,
    parse_expires_at,
    parse_mic_muted,
    parse_net_bytes,
    read_backlight_state,
    token_rate,
)

NET_DEV = """Inter-|   Receive                                                |  Transmit
 face |bytes    packets errs drop fifo frame compressed multicast|bytes    packets errs drop fifo colls carrier compressed
    lo: 1000      10    0    0    0     0          0         0     1000      10    0    0    0     0       0          0
wlp0s20f3: 5000      50    0    0    0     0          0         0     7000      70    0    0    0     0       0          0
"""


class FakeLlama:
    """Stands in for the HTTP client: records every request it is asked for."""

    def __init__(self, models, slots):
        self.models = models
        self.slots = slots
        self.calls = []

    def __call__(self, path, params=None):
        self.calls.append((path, params))
        if path == "/models":
            return {"data": [{"id": m, "status": {"value": s}} for m, s in self.models.items()]}
        if path == "/slots":
            if self.models.get(params["model"]) != "loaded":
                # What a router answers for an unloaded model when autoload is off.
                raise urllib.error.HTTPError(path, 400, "model is not loaded", {}, None)
            return self.slots[params["model"]]
        raise AssertionError(path)


def slot(processing, decoded, n_prompt=None, processed=None, cache=0, sid=None):
    value = {"is_processing": processing, "next_token": [{"n_decoded": decoded}]}
    if sid is not None:
        value["id"] = sid
    if n_prompt is not None:
        value.update(n_prompt_tokens=n_prompt, n_prompt_tokens_processed=processed, n_prompt_tokens_cache=cache)
    return value


def test_net_rate_from_proc_net_dev_sample_ignores_loopback():
    assert parse_net_bytes(NET_DEV) == 5000 + 7000


def test_net_rate_is_byte_delta_over_time():
    assert net_rate(12_000, 0.0, 15_000, 0.5) == 6000.0


def test_net_rate_is_zero_when_counter_goes_backwards():
    assert net_rate(15_000, 0.0, 100, 0.5) == 0.0


def test_token_rate_from_slots_counter_delta():
    assert token_rate(100, 140, 0.5) == 80.0


def test_token_counter_reset_gives_zero_not_negative_rate():
    assert token_rate(2205, 3, 0.5) == 0.0


def test_slots_unreachable_reports_server_down():
    def get(path, params=None):
        raise OSError("connection refused")

    assert llama_state(get) == LlamaState(up=False, processing=False, decoded=0)


def test_slots_prompt_phase_reports_reading():
    get = FakeLlama({"m": "loaded"}, {"m": [slot(True, 0)]})
    state = llama_state(get)
    assert state.up and state.processing and state.decoded == 0 and state.reading


def test_slots_report_prompt_progress_net_of_cached_tokens():
    get = FakeLlama({"m": "loaded"}, {"m": [slot(True, 0, n_prompt=5000, processed=1000, cache=3000)]})
    state = llama_state(get)
    assert (state.prompt_new, state.prompt_done) == (2000, 1000)
    assert state.reading


def test_prompt_fully_processed_is_no_longer_reading():
    get = FakeLlama({"m": "loaded"}, {"m": [slot(True, 0, n_prompt=5000, processed=2000, cache=3000)]})
    state = llama_state(get)
    assert state.processing and state.decoded == 0
    assert not state.reading


def test_slots_without_prompt_fields_fall_back_to_the_decoded_rule():
    get = FakeLlama({"m": "loaded"}, {"m": [slot(True, 0)]})
    state = llama_state(get)
    assert state.prompt_new is None and state.prompt_done is None
    assert state.reading


def test_idle_slot_prompt_fields_are_ignored():
    get = FakeLlama({"m": "loaded"}, {"m": [slot(False, 0, n_prompt=5000, processed=100, cache=0)]})
    state = llama_state(get)
    assert state.prompt_new is None and not state.reading


def test_slots_timeout_on_loaded_model_is_up_but_unanswered():
    def get(path, params=None):
        if path == "/models":
            return {"data": [{"id": "m", "status": {"value": "loaded"}}]}
        raise TimeoutError("busy")

    state = llama_state(get)
    assert state.up and not state.answered and not state.processing


def test_slots_http_error_is_an_answer_not_a_busy_server():
    # A loaded model that does not serve /slots replies with an error; that says
    # nothing about the other models being busy.
    import urllib.error

    def get(path, params=None):
        if path == "/models":
            return {"data": [{"id": "m", "status": {"value": "loaded"}}]}
        raise urllib.error.HTTPError("http://x/slots", 501, "not implemented", {}, None)

    state = llama_state(get)
    assert state.up and state.answered


def test_wrapped_socket_timeout_counts_as_busy():
    import urllib.error

    def get(path, params=None):
        if path == "/models":
            return {"data": [{"id": "m", "status": {"value": "loaded"}}]}
        raise urllib.error.URLError(TimeoutError("timed out"))

    assert not llama_state(get).answered


def test_model_list_timeout_is_busy_not_down():
    def get(path, params=None):
        raise TimeoutError("timed out")

    state = llama_state(get)
    assert state.up and not state.answered


def test_answered_is_true_when_every_loaded_model_responds():
    get = FakeLlama({"m": "loaded"}, {"m": [slot(False, 0)]})
    assert llama_state(get).answered


def test_backlight_state_is_read_from_the_kbd_light_state_file(tmp_path):
    path = tmp_path / "state"
    path.write_text("dim\n")
    assert read_backlight_state(path) == "dim"


@pytest.mark.parametrize("content", ["", "bright\n", None])
def test_backlight_state_is_none_when_missing_or_unknown(tmp_path, content):
    path = tmp_path / "state"
    if content is not None:
        path.write_text(content)
    assert read_backlight_state(path) is None


def test_slots_decoded_tokens_are_summed_across_loaded_models():
    get = FakeLlama({"a": "loaded", "b": "loaded"}, {"a": [slot(True, 40)], "b": [slot(True, 2)]})
    state = llama_state(get)
    assert state.decoded == 42 and not state.reading


def test_unloaded_model_is_never_queried():
    get = FakeLlama({"big": "unloaded", "small": "loaded"}, {"small": [slot(False, 0)]})
    state = llama_state(get)
    assert state.up and not state.processing
    queried = [p["model"] for path, p in get.calls if path == "/slots"]
    assert queried == ["small"]


def test_slots_queries_never_trigger_autoload():
    get = FakeLlama({"small": "loaded"}, {"small": [slot(False, 0)]})
    llama_state(get)
    (_, params), = [c for c in get.calls if c[0] == "/slots"]
    assert params["autoload"] == "false"


def hwmon(root, index, name, temps_milli):
    directory = root / f"hwmon{index}"
    directory.mkdir()
    (directory / "name").write_text(name + "\n")
    for n, value in enumerate(temps_milli, start=1):
        (directory / f"temp{n}_input").write_text(f"{value}\n")


def test_cpu_temp_takes_hottest_sensor(tmp_path):
    hwmon(tmp_path, 0, "nvme", [99000])
    hwmon(tmp_path, 1, "coretemp", [61000, 72000, 65000])
    assert cpu_temp_c(tmp_path) == 72.0


def test_cpu_temp_is_none_without_a_cpu_sensor(tmp_path):
    hwmon(tmp_path, 0, "nvme", [40000])
    assert cpu_temp_c(tmp_path) is None


def test_mic_muted_parsed_from_wpctl_output():
    assert parse_mic_muted("Volume: 1.00 [MUTED]\n") is True
    assert parse_mic_muted("Volume: 1.00\n") is False


def test_llama_state_can_be_limited_to_one_model_of_a_router():
    get = FakeLlama({"a": "loaded", "b": "loaded"}, {"a": [slot(True, 5)], "b": [slot(True, 90)]})
    state = llama_state(get, only_model="a")
    assert state.decoded == 5
    assert [p["model"] for path, p in get.calls if path == "/slots"] == ["a"]


def test_llama_state_for_a_model_that_is_not_loaded_is_up_and_idle():
    get = FakeLlama({"a": "loaded", "b": "unloaded"}, {"a": [slot(True, 5)]})
    state = llama_state(get, only_model="b")
    assert state.up and not state.processing



class ListingTimesOut(FakeLlama):
    """A router whose merged listing (/v1/models) hangs, as when one of its upstreams is unreachable.
    Its own models are still listed at once at /models."""

    def __call__(self, path, params=None):
        if path == "/v1/models":
            self.calls.append((path, params))
            raise TimeoutError("timed out")
        return super().__call__(path, params)


def test_a_named_model_is_read_from_its_slots_without_the_router_listing():
    get = ListingTimesOut({"a": "loaded"}, {"a": [slot(True, 5)]})
    state = llama_state(get, only_model="a")
    assert state.up and state.answered and state.loaded and state.decoded == 5
    assert [path for path, _ in get.calls] == ["/slots"]



def test_every_model_is_read_without_waiting_on_the_routers_upstreams():
    get = ListingTimesOut({"a": "loaded", "b": "unloaded", "c": "loaded"}, {"a": [slot(True, 5)], "c": [slot(True, 2)]})
    state = llama_state(get)
    assert state.answered and state.decoded == 7
    assert "/v1/models" not in [path for path, _ in get.calls]

def test_a_named_model_that_is_not_loaded_is_up_and_parked():
    state = llama_state(ListingTimesOut({"a": "unloaded"}, {}), only_model="a")
    assert state.up and state.answered and not state.loaded and not state.processing


def test_a_named_model_whose_slots_time_out_is_busy_not_down():
    def get(path, params=None):
        raise TimeoutError("timed out")

    state = llama_state(get, only_model="a")
    assert state.up and not state.answered


def test_a_named_model_on_a_refusing_server_is_down():
    def get(path, params=None):
        raise ConnectionRefusedError("refused")

    assert not llama_state(get, only_model="a").up

PS_CHAT = {"name": "qwen3.8:27b-mtp-q4_K_M", "expires_at": "2026-10-04T23:17:05.218197081-07:00"}
PS_EMBED = {"name": "bge-m3:batch4k", "expires_at": "2026-10-04T23:15:57.184440012-07:00"}


def ps(*models):
    return lambda path, params=None: {"models": list(models)}


def test_ollama_expiry_timestamp_parses_with_nanoseconds_and_offset():
    from datetime import datetime, timedelta, timezone

    expected = datetime(2026, 10, 4, 23, 17, 5, 218197, tzinfo=timezone(timedelta(hours=-7))).timestamp()
    assert parse_expires_at(PS_CHAT["expires_at"]) == pytest.approx(expected)
    assert parse_expires_at("not a time") is None


def test_ollama_with_a_chat_model_loaded_reports_loaded_and_its_expiry():
    state = ollama_state(ps(PS_CHAT))
    assert state.up and state.loaded
    assert state.touched == parse_expires_at(PS_CHAT["expires_at"])


def test_a_resident_embedding_model_counts_as_loaded_but_not_as_chat_activity():
    state = ollama_state(ps(PS_EMBED))
    assert state.up and state.loaded
    assert state.touched is None and state.in_flight is None  # chat expiry and the gate's counters stay about chat


def test_ollama_expiry_comes_from_the_chat_model_only():
    state = ollama_state(ps(PS_EMBED, PS_CHAT))
    assert state.loaded and state.touched == parse_expires_at(PS_CHAT["expires_at"])


def test_ollama_unreachable_reports_down():
    def get(path, params=None):
        raise ConnectionRefusedError("refused")

    assert not ollama_state(get).up


def test_ollama_timeout_is_busy_not_down():
    def get(path, params=None):
        raise TimeoutError("timed out")

    state = ollama_state(get)
    assert state.up and not state.answered


def model_with(*slots):
    return FakeLlama({"m": "loaded"}, {"m": list(slots)})


def reading_slot(done=1000, new=3000, sid=0):
    return slot(True, 0, n_prompt=new + 500, processed=done, cache=500, sid=sid)


def generating_slot(decoded=332, done=9800, sid=1):
    return slot(True, decoded, n_prompt=done + 500, processed=done, cache=500, sid=sid)


def test_a_server_with_two_slots_reads_a_prompt_while_the_other_slot_generates():
    state = llama_state(model_with(reading_slot(), generating_slot()))
    assert state.processing and state.reading
    assert state.decoded == 332


def test_prompt_progress_follows_the_reading_slot_not_the_generating_one():
    state = llama_state(model_with(reading_slot(done=1000, new=3000), generating_slot(done=9800)))
    assert (state.prompt_new, state.prompt_done) == (3000, 1000)


def test_peak_prompt_is_the_largest_slot_so_two_slots_never_add_up_to_a_compaction():
    state = llama_state(model_with(reading_slot(done=4000, new=6000), generating_slot(done=9800)))
    assert state.peak_prompt_done == 9800  # not 13800


def test_two_slots_reading_together_add_their_progress():
    state = llama_state(model_with(reading_slot(done=1000, new=3000, sid=0), reading_slot(done=500, new=2000, sid=1)))
    assert state.reading and (state.prompt_new, state.prompt_done) == (5000, 1500)


def test_a_slot_waiting_for_its_first_token_is_not_reading_even_beside_a_generating_slot():
    waiting = slot(True, 0, n_prompt=3500, processed=3000, cache=500, sid=0)
    state = llama_state(model_with(waiting, generating_slot()))
    assert state.processing and not state.reading


def test_an_idle_slots_stale_token_count_does_not_hide_the_other_slots_prompt():
    stale_idle = slot(False, 800, n_prompt=9000, processed=8000, cache=500, sid=1)
    state = llama_state(model_with(reading_slot(sid=0), stale_idle))
    assert state.reading and state.decoded == 0


def test_two_slots_without_prompt_fields_read_when_any_busy_slot_has_no_tokens():
    state = llama_state(model_with(slot(True, 0, sid=0), slot(True, 40, sid=1)))
    assert state.reading and state.decoded == 40 and state.peak_prompt_done is None


def gate(ps_models, activity=None, activity_error=None):
    """A fake client for an ollama-gate: /api/ps plus the optional /_gate/activity."""
    def get(path, params=None):
        if path == "/api/ps":
            return {"models": list(ps_models)}
        if path == "/_gate/activity":
            if activity_error:
                raise activity_error
            return activity
        raise AssertionError(path)
    return get


ACTIVITY = {"in_flight": 1, "awaiting_first_event": 1, "events_total": 40, "requests_total": 7}


def test_ollama_state_carries_the_gates_live_activity():
    state = ollama_state(gate([PS_CHAT], ACTIVITY))
    assert (state.in_flight, state.awaiting, state.events) == (1, 1, 40)
    assert state.up and state.loaded


def test_ollama_state_without_the_activity_endpoint_still_reports_loaded():
    import urllib.error

    error = urllib.error.HTTPError("http://x/_gate/activity", 404, "not found", {}, None)
    state = ollama_state(gate([PS_CHAT], activity_error=error))
    assert state.up and state.loaded
    assert (state.in_flight, state.awaiting, state.events) == (None, None, None)


def test_a_slow_activity_endpoint_does_not_take_down_the_loaded_state():
    state = ollama_state(gate([PS_CHAT], activity_error=TimeoutError("busy")))
    assert state.up and state.loaded and state.in_flight is None


def test_malformed_activity_is_ignored():
    state = ollama_state(gate([PS_CHAT], {"in_flight": "many"}))
    assert state.loaded and state.in_flight is None


def telemetry(*hosts):
    return lambda path, params=None: {"hosts": list(hosts)}


def host_entry(name="host-a", ok=True, state="healthy", age=0.5, gpus=None, gamemode=False):
    if gpus is None:
        gpus = [{"power_draw_w": 37.0, "power_limit_w": 350.0, "pstate": "P8", "utilization_pct": 1.0}]
    return {"host": name, "ok": ok, "sample_state": state, "sample_age_sec": age, "gpu": gpus, "gamemode": gamemode}


def test_gpu_state_reads_power_limit_and_pstate_of_the_named_host():
    state = gpu_state(telemetry(host_entry("host-b"), host_entry("host-a")), "host-a")
    assert state == GpuState(up=True, power_w=37.0, limit_w=350.0, pstate="P8")


def test_gpu_state_asks_telemetry_for_its_host_alone():
    # An unnamed read makes the server keep probing every machine it knows.
    calls = []
    gpu_state(lambda path, params=None: calls.append((path, params)) or {"hosts": []}, "host-a")
    assert calls == [("/api/telemetry", {"hosts": "host-a"})]


def test_gpu_state_takes_the_busiest_card_of_a_host():
    cards = [
        {"power_draw_w": 40.0, "power_limit_w": 350.0, "pstate": "P8"},
        {"power_draw_w": 200.0, "power_limit_w": 450.0, "pstate": "P0"},
    ]
    state = gpu_state(telemetry(host_entry(gpus=cards)), "host-a")
    assert state.power_w == 200.0 and state.limit_w == 450.0 and state.pstate == "P0"


def test_gpu_state_for_a_host_telemetry_does_not_know_is_down():
    assert gpu_state(telemetry(host_entry("host-b")), "host-a") == GPU_DOWN


def test_gpu_state_is_down_when_the_host_probe_failed_or_is_unhealthy_or_stale():
    assert not gpu_state(telemetry(host_entry(ok=False)), "host-a").up
    assert not gpu_state(telemetry(host_entry(state="stalled")), "host-a").up
    assert not gpu_state(telemetry(host_entry(age=45.0)), "host-a").up


def test_a_host_with_no_gpu_data_is_up_with_no_power_reading():
    state = gpu_state(telemetry(host_entry(gpus=[])), "host-a")
    assert state.up and state.power_w is None and state.pstate is None


def test_gpu_state_timeout_is_busy_not_down_and_refusal_is_down():
    def slow(path, params=None):
        raise TimeoutError("busy")

    def refused(path, params=None):
        raise ConnectionRefusedError("refused")

    assert gpu_state(slow, "host-a").up and not gpu_state(slow, "host-a").answered
    assert not gpu_state(refused, "host-a").up


def test_a_router_with_the_model_loaded_reports_it_loaded():
    get = FakeLlama({"a": "loaded", "b": "unloaded"}, {"a": [slot(False, 0)]})
    assert llama_state(get, only_model="a").loaded and llama_state(get).loaded


def test_a_router_without_the_model_loaded_reports_it_parked_but_up():
    get = FakeLlama({"a": "loaded", "b": "unloaded"}, {"a": [slot(False, 0)]})
    state = llama_state(get, only_model="b")
    assert state.up and not state.loaded and not state.processing


def test_a_router_with_nothing_loaded_reports_every_model_parked():
    get = FakeLlama({"a": "unloaded", "b": "unloaded"}, {})
    assert llama_state(get).up and not llama_state(get).loaded


def test_a_busy_slot_means_the_model_is_loaded():
    assert llama_state(model_with(generating_slot())).loaded


BURST = GpuState(up=True, power_w=140.0, limit_w=350.0, pstate="P2")
QUIET = GpuState(up=True, power_w=37.0, limit_w=350.0, pstate="P8")


def test_ollama_state_carries_the_gpu_overlay_when_one_is_configured():
    assert ollama_state(ps(PS_CHAT), gpu=lambda: BURST).gpu == BURST


def test_ollama_state_has_no_gpu_overlay_by_default():
    assert ollama_state(ps(PS_CHAT)).gpu is None


def test_a_gpu_overlay_that_is_down_or_slow_is_left_out_rather_than_darkening_the_light():
    assert ollama_state(ps(PS_CHAT), gpu=lambda: GPU_DOWN).gpu is None
    slow = GpuState(up=True, power_w=None, limit_w=None, pstate=None, answered=False)
    assert ollama_state(ps(PS_CHAT), gpu=lambda: slow).gpu is None


def test_the_gpu_overlay_is_read_when_only_an_embedder_is_loaded():
    state = ollama_state(ps(PS_EMBED), gpu=lambda: BURST)
    assert state.loaded and state.gpu == BURST


@pytest.mark.parametrize("text, online", [
    ("full\n", True),
    ("unknown\n", True),   # NetworkManager cannot tell: do not blank the key on a guess
    ("none\n", False),
    ("limited\n", False),
    ("portal\n", False),   # a captive portal is not the internet
    ("", True),
    ("garbage\n", True),
])
def test_networkmanager_connectivity_decides_whether_the_internet_is_up(text, online):
    assert parse_connectivity(text) is online


def test_gpu_state_carries_the_hosts_gamemode_flag():
    assert gpu_state(telemetry(host_entry(gamemode=True)), "host-a").gamemode is True
    assert gpu_state(telemetry(host_entry(gamemode=False)), "host-a").gamemode is False


def test_a_host_in_gamemode_is_up_even_with_no_gpu_reading():
    state = gpu_state(telemetry(host_entry(gpus=[], gamemode=True)), "host-a")
    assert state.up and state.gamemode


def test_a_host_in_gamemode_stays_gamemode_between_its_slow_samples():
    # A gamemode host is sampled about once a minute, so its sample is normally older than the
    # freshness limit of a working one; that must not turn gamemode into an outage.
    state = gpu_state(telemetry(host_entry(gpus=[], gamemode=True, age=55.0)), "host-a")
    assert state.up and state.gamemode


def test_a_gamemode_sample_older_than_a_few_polls_is_down():
    assert not gpu_state(telemetry(host_entry(gpus=[], gamemode=True, age=400.0)), "host-a").up


GAMING = GpuState(up=True, power_w=30.0, limit_w=450.0, pstate="P8", gamemode=True)


def test_the_gpu_overlay_survives_ollama_being_unreachable_so_gamemode_can_be_told_from_an_outage():
    def stopped(path, params=None):
        raise ConnectionRefusedError("ollama stopped")

    state = ollama_state(stopped, gpu=lambda: GAMING)
    assert not state.up and state.gpu == GAMING


def test_the_gpu_overlay_is_kept_when_ollama_is_merely_silent():
    def slow(path, params=None):
        raise TimeoutError("busy")

    state = ollama_state(slow, gpu=lambda: GAMING)
    assert state.up and not state.answered and state.gpu == GAMING


def test_nothing_resident_is_not_loaded():
    state = ollama_state(ps(), gpu=lambda: QUIET)
    assert state.up and not state.loaded and state.gpu == QUIET


def test_cache_path_reuses_one_path_for_its_ttl():
    calls = []
    now = [0.0]

    def get(path, params=None):
        calls.append(path)
        return {"path": path, "n": len(calls)}

    cached = cache_path(get, "/api/ps", 10.0, clock=lambda: now[0])
    assert cached("/api/ps")["n"] == 1
    now[0] = 9.9
    assert cached("/api/ps")["n"] == 1
    assert cached("/_gate/activity")["path"] == "/_gate/activity"
    assert cached("/_gate/activity")["path"] == "/_gate/activity"
    now[0] = 10.0
    assert cached("/api/ps")["n"] == 4
    assert calls == ["/api/ps", "/_gate/activity", "/_gate/activity", "/api/ps"]


def test_cache_path_holds_a_failure_too():
    calls = []
    now = [0.0]

    def get(path, params=None):
        calls.append(path)
        raise urllib.error.URLError("refused")

    cached = cache_path(get, "/api/ps", 10.0, clock=lambda: now[0])
    for _ in range(3):
        with pytest.raises(urllib.error.URLError):
            cached("/api/ps")
    assert calls == ["/api/ps"]
    assert not ollama_state(cached).up
    now[0] = 11.0
    with pytest.raises(urllib.error.URLError):
        cached("/api/ps")
    assert calls == ["/api/ps", "/api/ps"]
