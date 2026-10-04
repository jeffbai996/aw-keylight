import pytest

from keylights.sources import (
    LlamaState,
    OllamaState,
    cpu_temp_c,
    llama_state,
    net_rate,
    ollama_state,
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
        if path == "/v1/models":
            return {"data": [{"id": m, "status": {"value": s}} for m, s in self.models.items()]}
        if path == "/slots":
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
        if path == "/v1/models":
            return {"data": [{"id": "m", "status": {"value": "loaded"}}]}
        raise TimeoutError("busy")

    state = llama_state(get)
    assert state.up and not state.answered and not state.processing


def test_slots_http_error_is_an_answer_not_a_busy_server():
    # A loaded model that does not serve /slots replies with an error; that says
    # nothing about the other models being busy.
    import urllib.error

    def get(path, params=None):
        if path == "/v1/models":
            return {"data": [{"id": "m", "status": {"value": "loaded"}}]}
        raise urllib.error.HTTPError("http://x/slots", 501, "not implemented", {}, None)

    state = llama_state(get)
    assert state.up and state.answered


def test_wrapped_socket_timeout_counts_as_busy():
    import urllib.error

    def get(path, params=None):
        if path == "/v1/models":
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


def test_ollama_embedding_models_do_not_count_as_inference():
    state = ollama_state(ps(PS_EMBED))
    assert state.up and not state.loaded and state.touched is None


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
