from keylights.effects import (
    AMBER,
    GREEN,
    OFF,
    RED,
    Inputs,
    blink_on,
    blink_rate,
    build_frame,
    del_color,
    mute_color,
    net_blink_rate,
    prompt_blink_rate,
    temp_color,
)
from keylights.sources import LlamaState

BASE = (255, 255, 255)
UP_IDLE = LlamaState(up=True, processing=False, decoded=0)
READING = LlamaState(up=True, processing=True, decoded=0)
READING_PROMPT = LlamaState(up=True, processing=True, decoded=0, prompt_new=2000, prompt_done=500)
AWAITING_TOKEN = LlamaState(up=True, processing=True, decoded=0, prompt_new=2000, prompt_done=2000)
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
    values = dict(llama=UP_IDLE, token_rate=0.0, net_bytes_per_s=0.0, temp_c=45.0, muted=False)
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
