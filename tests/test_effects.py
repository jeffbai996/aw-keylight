from keylights.effects import (
    AMBER,
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
from keylights.sources import LlamaState, OllamaState

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


def test_del_stays_lit_when_server_up_and_idle():
    assert del_color(UP_IDLE, 0.0, 0.3) == GREEN


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


def test_del_solid_green_when_generation_is_done_and_server_idle():
    assert del_color(UP_IDLE, 0.0, 0.3) == GREEN
    assert del_color(UP_IDLE, 0.0, 0.7) == GREEN


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
    assert boosted[15] == (0, 255, 0)  # already at full channel
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


def test_del_stays_solid_amber_for_the_first_token_wait_of_a_compaction():
    assert del_color(COMPACT_AWAITING, 0.0, 0.3, compaction=True) == AMBER
    assert del_color(COMPACT_AWAITING, 0.0, 0.7, compaction=True) == AMBER


def test_del_flickers_amber_with_tokens_while_a_compaction_generates():
    assert del_color(COMPACT_DECODING, 4.0, 0.05, compaction=True) == AMBER
    assert del_color(COMPACT_DECODING, 4.0, 0.15, compaction=True) == OFF


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


def test_ollama_light_is_green_while_a_chat_model_is_loaded():
    assert ollama_color(OLLAMA_LOADED, pulse=False) == GREEN


def test_ollama_light_is_dark_when_nothing_is_loaded_or_the_server_is_down():
    assert ollama_color(OLLAMA_EMPTY, pulse=False) == OFF
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
    assert frame[key_id("F12")] == GREEN
    assert frame[key_id("HOME")] == WHITE
    assert frame[key_id("END")] == AMBER  # prompt blink, on-phase
    assert frame[key_id("DEL")] == GREEN  # token flicker, on-phase


def test_light_keys_blink_independently_of_each_other():
    frame = lights_frame(0.15)
    assert frame[key_id("END")] == OFF and frame[key_id("DEL")] == OFF
    assert frame[key_id("F12")] == GREEN and frame[key_id("HOME")] == WHITE


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
