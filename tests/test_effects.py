from keylights.effects import (
    AMBER,
    boost_color,
    BABY_BLUE,
    GpuInput,
    gpu_blink_rate,
    gpu_color,
    WHITE,
    InferenceInput,
    OllamaInput,
    ollama_color,
    power_state,
    GREEN,
    OFF,
    RED,
    Inputs,
    blink_on,
    blink_rate,
    build_frame,
    del_color,
    is_compaction,
    mute_color,
    net_blink_rate,
    prompt_blink_rate,
    temp_color,
)
from keylights.keymap import key_id
from keylights.sources import GPU_DOWN, GpuState, LlamaState, OllamaState

BASE = (255, 255, 255)
UP_IDLE = LlamaState(up=True, processing=False, decoded=0)
READING = LlamaState(up=True, processing=True, decoded=0)
READING_PROMPT = LlamaState(up=True, processing=True, decoded=0, prompt_new=2000, prompt_done=500)
AWAITING_TOKEN = LlamaState(up=True, processing=True, decoded=0, prompt_new=2000, prompt_done=2000)
COMPACT_READING = LlamaState(up=True, processing=True, decoded=0, prompt_new=30000, prompt_done=15000)
COMPACT_AWAITING = LlamaState(up=True, processing=True, decoded=0, prompt_new=24000, prompt_done=24000)
COMPACT_DECODING = LlamaState(up=True, processing=True, decoded=300, prompt_new=24000, prompt_done=24000)
DECODING = LlamaState(up=True, processing=True, decoded=120)
DOWN = LlamaState(up=False, processing=False, decoded=0)


def test_temp_gradient_green_at_45_red_at_90_and_midpoint():
    assert temp_color(45) == (0, 255, 0)
    assert temp_color(90) == (255, 0, 0)
    assert temp_color(67.5) == (255, 255, 0)


def test_temp_gradient_clamps_outside_range():
    assert temp_color(20) == (0, 255, 0)
    assert temp_color(120) == (255, 0, 0)


def test_blink_rate_rises_with_token_rate_and_caps_at_max():
    assert blink_rate(0) == 0
    assert blink_rate(10) < blink_rate(20)
    assert blink_rate(10_000, cap=20.0) == 20.0


def test_blink_is_on_during_first_half_of_each_period():
    assert blink_on(2.0, 0.1) is True
    assert blink_on(2.0, 0.3) is False
    assert blink_on(2.0, 0.6) is True


def test_zero_rate_means_steady_on():
    assert blink_on(0.0, 0.3) is True


def test_del_is_baby_blue_when_server_up_and_idle():
    assert del_color(UP_IDLE, 0.0, 0.3) == BABY_BLUE


def test_del_amber_while_prompt_is_read():
    assert del_color(READING, 0.0, 0.3) == AMBER


def test_del_off_when_server_down():
    assert del_color(DOWN, 0.0, 0.3) == OFF


def test_del_flickers_between_green_and_off_while_decoding():
    assert del_color(DECODING, 4.0, 0.05) == GREEN
    assert del_color(DECODING, 4.0, 0.15) == OFF


def test_del_blinks_amber_while_prompt_is_advancing():
    assert del_color(READING_PROMPT, 4.0, 0.05, prompt_advancing=True) == AMBER
    assert del_color(READING_PROMPT, 4.0, 0.15, prompt_advancing=True) == OFF


def test_del_solid_amber_while_reading_without_progress():
    assert del_color(READING_PROMPT, 4.0, 0.15, prompt_advancing=False) == AMBER


def test_del_solid_green_once_prompt_is_read_and_first_token_is_pending():
    assert not AWAITING_TOKEN.reading
    assert del_color(AWAITING_TOKEN, 0.0, 0.3) == GREEN
    assert del_color(AWAITING_TOKEN, 0.0, 0.7) == GREEN


def test_del_is_baby_blue_when_generation_is_done_and_server_idle():
    assert del_color(UP_IDLE, 0.0, 0.3) == BABY_BLUE
    assert del_color(UP_IDLE, 0.0, 0.7) == BABY_BLUE


def test_prompt_blink_rate_follows_prompt_speed_within_floor_and_cap():
    assert prompt_blink_rate(0) == 0
    assert prompt_blink_rate(400) == 4.0
    assert prompt_blink_rate(10) == 2.0  # slow progress still visibly blinks
    assert prompt_blink_rate(1_000_000, cap=20.0) == 20.0


def test_esc_blink_rate_follows_network_rate():
    assert net_blink_rate(0) == 0
    assert net_blink_rate(20_000) < net_blink_rate(40_000)
    assert net_blink_rate(10_000_000, cap=10.0) == 10.0


def test_dim_boost_scales_only_esc_and_del_and_caps_at_full_channel():
    plain = build_frame(inputs(temp_c=45.0), 0.0, BASE)
    boosted = build_frame(inputs(temp_c=45.0), 0.0, BASE, boost=1.6)
    assert boosted[0] == tuple(min(255, round(c * 1.6)) for c in plain[0])
    assert boosted[15] == boost_color(BABY_BLUE, 1.6)  # the idle colour is lifted like the rest
    assert {k: boosted[k] for k in (1, 2, 3, 4, 5)} == {k: plain[k] for k in (1, 2, 3, 4, 5)}


def test_no_boost_leaves_frame_unchanged():
    assert build_frame(inputs(), 0.0, BASE, boost=1.0) == build_frame(inputs(), 0.0, BASE)


def test_frame_blinks_del_amber_from_prompt_rate_when_advancing():
    live = inputs(llama=READING_PROMPT, prompt_rate=400.0, prompt_advancing=True)
    colors = {build_frame(live, t / 100, BASE)[15] for t in range(0, 40)}
    assert colors == {AMBER, OFF}


def test_f5_red_only_when_muted():
    assert mute_color(True, BASE) == RED
    assert mute_color(False, BASE) == BASE


def inputs(**overrides):
    """The legacy single-light arguments describe the DEL light."""
    light = {k: overrides.pop(k) for k in ("llama", "token_rate", "prompt_rate", "prompt_advancing") if k in overrides}
    state = light.pop("llama", UP_IDLE)
    values = dict(net_bytes_per_s=0.0, temp_c=45.0, muted=False, lights={key_id("DEL"): InferenceInput(state, **light)})
    values.update(overrides)
    return Inputs(**values)


def test_f1_to_f4_share_the_temperature_color():
    frame = build_frame(inputs(temp_c=90.0), 0.0, BASE)
    assert {frame[k] for k in (1, 2, 3, 4)} == {(255, 0, 0)}


def test_temperature_keys_fall_back_to_base_when_sensor_missing():
    frame = build_frame(inputs(temp_c=None), 0.0, BASE)
    assert {frame[k] for k in (1, 2, 3, 4)} == {BASE}


def test_frame_covers_exactly_the_managed_keys():
    frame = build_frame(inputs(), 0.0, BASE)
    assert set(frame) == {0, 1, 2, 3, 4, 5, 15}


def test_compaction_needs_a_busy_slot_with_enough_prompt_tokens_processed():
    assert is_compaction(COMPACT_READING, 12288)
    assert not is_compaction(COMPACT_READING, 20000)
    assert not is_compaction(READING_PROMPT, 12288)
    assert not is_compaction(UP_IDLE, 12288)


def test_compaction_is_false_without_prompt_fields_or_with_an_idle_slot():
    assert not is_compaction(READING, 12288)
    idle_after_big_job = LlamaState(up=True, processing=False, decoded=0, prompt_new=30000, prompt_done=30000)
    assert not is_compaction(idle_after_big_job, 12288)


def test_del_flashes_amber_fast_while_a_compaction_sized_prompt_is_read():
    assert del_color(COMPACT_READING, 10.0, 0.02, compaction=True) == AMBER
    assert del_color(COMPACT_READING, 10.0, 0.07, compaction=True) == OFF


def test_del_is_green_for_the_first_token_wait_even_after_a_large_prompt():
    assert del_color(COMPACT_AWAITING, 0.0, 0.3, compaction=True) == GREEN
    assert del_color(COMPACT_AWAITING, 0.0, 0.7, compaction=True) == GREEN


def test_del_flickers_green_not_amber_while_a_large_job_generates():
    assert del_color(COMPACT_DECODING, 4.0, 0.05, compaction=True) == GREEN
    assert del_color(COMPACT_DECODING, 4.0, 0.15, compaction=True) == OFF


def test_amber_never_appears_outside_reading_for_any_job_size():
    for state in (COMPACT_AWAITING, COMPACT_DECODING, DECODING, AWAITING_TOKEN, UP_IDLE):
        for t in (0.0, 0.05, 0.15, 0.3, 0.7):
            for compaction in (False, True):
                assert del_color(state, 4.0, t, compaction=compaction) != AMBER


def test_frame_compaction_flash_ignores_prompt_speed():
    for prompt_rate in (0.0, 50.0, 900.0):
        live = inputs(llama=COMPACT_READING, prompt_rate=prompt_rate, prompt_advancing=True)
        colors = {build_frame(live, t / 100, BASE, compact_tokens=12288, compact_blink=10.0)[15] for t in range(40)}
        assert colors == {AMBER, OFF}


def test_frame_compaction_flash_is_capped_by_the_del_blink_cap():
    live = inputs(llama=COMPACT_READING)
    uncapped = build_frame(live, 0.2, BASE, del_cap=55.0, compact_tokens=12288, compact_blink=10.0)
    capped = build_frame(live, 0.2, BASE, del_cap=4.0, compact_tokens=12288, compact_blink=10.0)
    assert uncapped[15] == AMBER and capped[15] == OFF


def test_frame_below_the_compaction_threshold_keeps_the_prompt_blink():
    live = inputs(llama=READING_PROMPT, prompt_rate=400.0, prompt_advancing=True)
    colors = {build_frame(live, t / 100, BASE, compact_tokens=12288, compact_blink=10.0)[15] for t in range(40)}
    assert colors == {AMBER, OFF}
    steady = build_frame(inputs(llama=READING_PROMPT), 0.7, BASE, compact_tokens=12288, compact_blink=10.0)
    assert steady[15] == AMBER


OLLAMA_LOADED = OllamaState(up=True, loaded=True, touched=1.0)
OLLAMA_EMPTY = OllamaState(up=True, loaded=False, touched=None)
OLLAMA_DOWN = OllamaState(up=False, loaded=False, touched=None)


def test_ollama_light_is_baby_blue_while_a_chat_model_is_loaded_and_idle():
    assert ollama_color(OLLAMA_LOADED, pulse=False) == BABY_BLUE


def test_ollama_light_is_baby_blue_when_the_server_is_up_but_no_chat_model_is_loaded():
    assert ollama_color(OLLAMA_EMPTY, pulse=False) == BABY_BLUE


def test_ollama_light_is_dark_only_when_the_server_is_unreachable():
    assert ollama_color(OLLAMA_DOWN, pulse=False) == OFF


def test_ollama_light_pulses_white_when_a_request_finishes():
    assert ollama_color(OLLAMA_LOADED, pulse=True) == WHITE


def lights_frame(t, **frame_args):
    lights = {
        key_id("F12"): OllamaInput(OLLAMA_LOADED, pulse=False),
        key_id("HOME"): OllamaInput(OLLAMA_LOADED, pulse=True),
        key_id("END"): InferenceInput(READING_PROMPT, prompt_rate=400.0, prompt_advancing=True),
        key_id("DEL"): InferenceInput(DECODING, token_rate=8.0),
    }
    return build_frame(inputs(lights=lights), t, BASE, **frame_args)


def test_each_light_key_shows_its_own_source():
    frame = lights_frame(0.05)
    assert frame[key_id("F12")] == BABY_BLUE
    assert frame[key_id("HOME")] == WHITE
    assert frame[key_id("END")] == AMBER  # prompt blink, on-phase
    assert frame[key_id("DEL")] == GREEN  # token flicker, on-phase


def test_light_keys_blink_independently_of_each_other():
    frame = lights_frame(0.15)
    assert frame[key_id("END")] == OFF and frame[key_id("DEL")] == OFF
    assert frame[key_id("F12")] == BABY_BLUE and frame[key_id("HOME")] == WHITE


def test_dim_boost_also_lifts_the_light_keys():
    plain = build_frame(inputs(lights={key_id("END"): InferenceInput(READING)}), 0.0, BASE)
    boosted = build_frame(inputs(lights={key_id("END"): InferenceInput(READING)}), 0.0, BASE, boost=1.6)
    assert plain[key_id("END")] == AMBER and boosted[key_id("END")] == (255, 224, 0)


def test_frame_has_only_the_status_keys_when_no_lights_are_configured():
    frame = build_frame(inputs(lights={}), 0.0, BASE)
    assert set(frame) == {0, 1, 2, 3, 4, 5}


def test_frame_adds_exactly_the_configured_light_keys():
    assert set(lights_frame(0.0)) == {0, 1, 2, 3, 4, 5, 12, 13, 14, 15}


def test_power_state_reads_prompt_when_any_inference_light_is_reading():
    lights = {key_id("END"): InferenceInput(READING_PROMPT), key_id("DEL"): InferenceInput(DECODING)}
    state = power_state(lights)
    assert state.up and state.reading


def test_power_state_is_up_when_any_light_is_up_and_none_is_reading():
    lights = {key_id("HOME"): OllamaInput(OLLAMA_DOWN, pulse=False), key_id("END"): InferenceInput(UP_IDLE)}
    state = power_state(lights)
    assert state.up and not state.reading


def test_power_state_counts_a_loaded_ollama_model_as_up():
    state = power_state({key_id("HOME"): OllamaInput(OLLAMA_LOADED, pulse=False)})
    assert state.up and not state.reading


def test_power_state_is_down_when_every_light_is_down_or_none_exist():
    assert not power_state({}).up
    assert not power_state({key_id("HOME"): OllamaInput(OLLAMA_DOWN, pulse=False), key_id("END"): InferenceInput(DOWN)}).up


def test_compaction_looks_at_the_largest_slot_when_the_state_carries_one():
    big_slot = LlamaState(up=True, processing=True, decoded=5, prompt_new=300, prompt_done=100, peak_prompt_done=15000)
    small_slots = LlamaState(up=True, processing=True, decoded=5, prompt_new=300, prompt_done=100, peak_prompt_done=9800)
    assert is_compaction(big_slot, 12288)
    assert not is_compaction(small_slots, 12288)


def live(in_flight=0, awaiting=0, events=None):
    return OllamaState(up=True, loaded=True, touched=1.0, in_flight=in_flight, awaiting=awaiting, events=events)


def test_ollama_light_blinks_amber_while_the_gate_waits_for_the_first_event():
    state = live(in_flight=1, awaiting=1)
    assert ollama_color(state, pulse=False, t=0.05) == AMBER
    assert ollama_color(state, pulse=False, t=0.15) == OFF


def test_ollama_light_flickers_green_with_streamed_events():
    state = live(in_flight=1, awaiting=0)
    assert ollama_color(state, pulse=False, t=0.05, event_rate=8.0) == GREEN
    assert ollama_color(state, pulse=False, t=0.15, event_rate=8.0) == OFF


def test_ollama_light_stays_green_while_streaming_with_no_new_events_yet():
    assert ollama_color(live(in_flight=1), pulse=False, t=0.15, event_rate=0.0) == GREEN


def test_ollama_light_flicker_is_capped():
    state = live(in_flight=1)
    assert ollama_color(state, pulse=False, t=0.2, event_rate=1000.0, cap=4.0) == OFF  # 4/s, off-phase
    assert ollama_color(state, pulse=False, t=0.2, event_rate=1000.0, cap=50.0) == GREEN


def test_ollama_light_pulses_white_only_when_idle():
    assert ollama_color(live(in_flight=0), pulse=True) == WHITE
    assert ollama_color(live(in_flight=1, awaiting=1), pulse=True, t=0.05) == AMBER


def test_ollama_light_without_activity_data_behaves_as_before():
    assert ollama_color(OLLAMA_LOADED, pulse=False, t=0.15, event_rate=5.0) == BABY_BLUE
    assert ollama_color(OLLAMA_LOADED, pulse=True) == WHITE


def test_frame_blinks_a_light_from_its_event_rate_and_cap():
    light = OllamaInput(live(in_flight=1), pulse=False, event_rate=8.0)
    on, off = (build_frame(inputs(lights={key_id("HOME"): light}), t, BASE)[key_id("HOME")] for t in (0.05, 0.15))
    assert (on, off) == (GREEN, OFF)


def test_frame_shows_green_flicker_for_a_large_job_that_is_generating():
    live = inputs(llama=COMPACT_DECODING, token_rate=8.0)
    colors = {build_frame(live, t / 100, BASE, compact_tokens=12288, compact_blink=10.0)[15] for t in range(40)}
    assert colors == {GREEN, OFF}


GPU_IDLE = GpuState(up=True, power_w=37.0, limit_w=350.0, pstate="P8")
GPU_BURST = GpuState(up=True, power_w=140.0, limit_w=350.0, pstate="P2")
GPU_TAIL = GpuState(up=True, power_w=54.0, limit_w=350.0, pstate="P5")


def test_gpu_light_is_steady_baby_blue_while_the_card_idles():
    assert {gpu_color(GPU_IDLE, t / 10) for t in range(10)} == {BABY_BLUE}


def test_gpu_light_flickers_green_while_the_card_works():
    assert {gpu_color(GPU_BURST, t / 100, cap=25.0) for t in range(60)} == {GREEN, OFF}


def test_gpu_light_is_dark_when_the_host_is_down():
    assert gpu_color(GPU_DOWN, 0.1) == OFF


def test_gpu_flicker_speeds_up_with_power_draw():
    assert gpu_blink_rate(GPU_BURST, 25.0) > gpu_blink_rate(GPU_TAIL, 25.0) > 0


def test_gpu_flicker_is_capped():
    full = GpuState(up=True, power_w=350.0, limit_w=350.0, pstate="P0")
    assert gpu_blink_rate(full, 10.0) == 10.0


def test_gpu_activity_falls_back_to_power_share_when_the_pstate_is_missing():
    assert gpu_blink_rate(GpuState(True, 140.0, 350.0, None), 25.0) > 0
    assert gpu_blink_rate(GpuState(True, 37.0, 350.0, None), 25.0) == 0


def test_a_gpu_with_no_power_reading_reads_as_idle():
    unknown = GpuState(up=True, power_w=None, limit_w=None, pstate=None)
    assert gpu_blink_rate(unknown, 25.0) == 0 and gpu_color(unknown, 0.3) == BABY_BLUE


def test_frame_shows_a_gpu_light_from_its_power_state():
    busy = {key_id("HOME"): GpuInput(GPU_BURST)}
    idle = {key_id("HOME"): GpuInput(GPU_IDLE)}
    assert {build_frame(inputs(lights=busy), t / 100, BASE, del_cap=25.0)[key_id("HOME")] for t in range(60)} == {GREEN, OFF}
    assert build_frame(inputs(lights=idle), 0.3, BASE)[key_id("HOME")] == BABY_BLUE


def test_power_state_counts_a_gpu_light_as_up_only_while_its_host_is():
    assert power_state({key_id("HOME"): GpuInput(GPU_IDLE)}).up
    assert not power_state({key_id("HOME"): GpuInput(GPU_DOWN)}).up


PARKED = LlamaState(up=True, processing=False, decoded=0, loaded=False)


def test_a_llama_server_with_its_model_parked_shows_baby_blue_not_dark():
    assert {del_color(PARKED, 0.0, t / 10) for t in range(10)} == {BABY_BLUE}


def test_a_llama_server_that_is_down_is_dark_and_one_that_is_idle_is_not():
    assert del_color(DOWN, 0.0, 0.3) == OFF
    assert del_color(UP_IDLE, 0.0, 0.3) == BABY_BLUE


def test_frame_shows_parked_hosts_as_baby_blue():
    lights = {key_id("DEL"): InferenceInput(PARKED), key_id("END"): OllamaInput(OLLAMA_EMPTY, pulse=False)}
    frame = build_frame(inputs(lights=lights), 0.0, BASE)
    assert frame[key_id("DEL")] == BABY_BLUE and frame[key_id("END")] == BABY_BLUE


def test_power_state_counts_a_parked_ollama_host_as_up():
    assert power_state({key_id("END"): OllamaInput(OLLAMA_EMPTY, pulse=False)}).up


def with_gpu(state, gpu):
    from dataclasses import replace

    return replace(state, gpu=gpu)


def test_a_parked_host_whose_gpu_is_idle_is_baby_blue():
    parked = with_gpu(OLLAMA_EMPTY, GPU_IDLE)
    assert {ollama_color(parked, pulse=False, t=t / 10) for t in range(10)} == {BABY_BLUE}


def test_a_parked_host_whose_gpu_is_working_flickers_green():
    parked = with_gpu(OLLAMA_EMPTY, GPU_BURST)
    assert {ollama_color(parked, pulse=False, t=t / 100, cap=25.0) for t in range(60)} == {GREEN, OFF}


def test_a_loaded_idle_host_flickers_green_when_its_gpu_works_and_is_baby_blue_when_not():
    assert {ollama_color(with_gpu(OLLAMA_LOADED, GPU_BURST), pulse=False, t=t / 100, cap=25.0) for t in range(60)} == {GREEN, OFF}
    assert {ollama_color(with_gpu(OLLAMA_LOADED, GPU_IDLE), pulse=False, t=t / 10) for t in range(10)} == {BABY_BLUE}


def test_a_chat_request_through_the_gate_outranks_the_gpu_flicker():
    waiting = with_gpu(live(in_flight=1, awaiting=1), GPU_BURST)
    assert ollama_color(waiting, pulse=False, t=0.05) == AMBER


def test_an_unreachable_host_is_dark_whatever_the_gpu_says():
    assert ollama_color(with_gpu(OLLAMA_DOWN, GPU_BURST), pulse=False) == OFF


def test_every_kind_of_light_shares_one_vocabulary_idle_blue_working_green_unreachable_dark():
    ollama_busy = OllamaState(up=True, loaded=True, touched=1.0, in_flight=1, awaiting=0, events=5)
    cases = {
        "llama": (UP_IDLE, DECODING, DOWN, lambda st, t: del_color(st, 8.0 if st.processing else 0.0, t)),
        "ollama": (OLLAMA_LOADED, ollama_busy, OLLAMA_DOWN, lambda st, t: ollama_color(st, False, t, event_rate=16.0)),
        "gpu": (GPU_IDLE, GPU_BURST, GPU_DOWN, lambda st, t: gpu_color(st, t, cap=25.0)),
    }
    for kind, (idle, working, down, color) in cases.items():
        assert {color(idle, t / 100) for t in range(60)} == {BABY_BLUE}, kind
        assert {color(working, t / 100) for t in range(60)} == {GREEN, OFF}, kind
        assert {color(down, t / 100) for t in range(60)} == {OFF}, kind
