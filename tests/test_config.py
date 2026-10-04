import re
from pathlib import Path

import pytest

from keylights.config import LightSpec, load_config


def test_endpoints_come_from_env_and_no_default_url_is_hardcoded():
    assert load_config({}).llama_url is None
    assert load_config({"KEYLIGHTS_LLAMA_URL": "http://llama.example:8080"}).llama_url == "http://llama.example:8080"

    package = Path(__file__).parent.parent / "keylights"
    for source in package.glob("*.py"):
        text = source.read_text()
        assert not re.search(r"\.ts\.net\b", text), source.name
        assert not re.search(r"\d+\.\d+\.\d+\.\d+", text), source.name


def test_blink_caps_default_to_20_per_second_for_del_and_10_for_esc():
    config = load_config({})
    assert config.del_blink_cap == 20.0
    assert config.esc_blink_cap == 10.0


def test_base_color_is_read_from_env_as_hex():
    assert load_config({"KEYLIGHTS_BASE_COLOR": "ff8000"}).base_color == (255, 128, 0)


def test_dim_boost_defaults_to_1_6_and_is_overridable():
    assert load_config({}).dim_boost == 1.6
    assert load_config({"KEYLIGHTS_DIM_BOOST": "2.0"}).dim_boost == 2.0


def test_backlight_state_path_defaults_under_the_home_state_dir_and_is_overridable():
    assert load_config({}).backlight_state_path.endswith("/.local/state/kbd-light/state")
    assert load_config({"KEYLIGHTS_BACKLIGHT_STATE": "/tmp/x"}).backlight_state_path == "/tmp/x"


def test_compaction_threshold_and_flash_rate_have_defaults_and_overrides():
    config = load_config({})
    assert config.compact_tokens == 12288 and config.compact_blink == 10.0
    tuned = load_config({"KEYLIGHTS_COMPACT_TOKENS": "20000", "KEYLIGHTS_COMPACT_BLINK": "6"})
    assert tuned.compact_tokens == 20000 and tuned.compact_blink == 6.0


def test_lights_spec_maps_keys_to_sources_with_an_optional_model():
    config = load_config({"KEYLIGHTS_LIGHTS": "END=llama:http://h.example:8080#fn-1, HOME=ollama:https://o.example:8476 ,F12=ollama:http://x.example:11434"})
    assert config.lights == (
        LightSpec("END", "llama", "http://h.example:8080", "fn-1"),
        LightSpec("HOME", "ollama", "https://o.example:8476", None),
        LightSpec("F12", "ollama", "http://x.example:11434", None),
    )


def test_legacy_llama_url_becomes_the_del_light():
    assert load_config({"KEYLIGHTS_LLAMA_URL": "http://h.example:8080"}).lights == (
        LightSpec("DEL", "llama", "http://h.example:8080", None),
    )


def test_lights_spec_takes_precedence_over_the_legacy_url():
    config = load_config({"KEYLIGHTS_LLAMA_URL": "http://old.example", "KEYLIGHTS_LIGHTS": "END=llama:http://new.example"})
    assert [light.key for light in config.lights] == ["END"]


def test_no_lights_are_configured_by_default():
    assert load_config({}).lights == ()


@pytest.mark.parametrize(
    "spec",
    [
        "DEL=http://x.example",  # no kind
        "NOPE=llama:http://x.example",  # unknown key
        "DEL=gpu:http://x.example",  # unknown kind
        "DEL=llama:http://a.example,DEL=llama:http://b.example",  # duplicate key
        "ESC=llama:http://x.example",  # keys the status lights already own
        "F3=ollama:http://x.example",
    ],
)
def test_bad_lights_specs_are_rejected(spec):
    with pytest.raises(ValueError):
        load_config({"KEYLIGHTS_LIGHTS": spec})


def test_a_gpu_light_names_the_host_whose_gpu_it_shows():
    (light,) = load_config({"KEYLIGHTS_LIGHTS": "HOME=gpu:https://s.example/squad#host-a"}).lights
    assert (light.key, light.kind, light.url, light.selector) == ("HOME", "gpu", "https://s.example/squad", "host-a")


def test_a_gpu_light_without_a_host_is_rejected():
    with pytest.raises(ValueError):
        load_config({"KEYLIGHTS_LIGHTS": "HOME=gpu:https://s.example/squad"})


def test_an_ollama_light_can_overlay_a_hosts_gpu_activity():
    spec = "HOME=ollama:https://o.example:8476#gpu=https://s.example/squad@host-a"
    (light,) = load_config({"KEYLIGHTS_LIGHTS": spec}).lights
    assert (light.kind, light.url, light.selector) == ("ollama", "https://o.example:8476", "gpu=https://s.example/squad@host-a")


@pytest.mark.parametrize("selector", ["gpu=https://s.example/squad", "model-x", "gpu=@host-a"])
def test_a_malformed_ollama_gpu_overlay_is_rejected(selector):
    with pytest.raises(ValueError):
        load_config({"KEYLIGHTS_LIGHTS": f"HOME=ollama:https://o.example#{selector}"})


def test_reset_interval_defaults_to_a_minute_and_zero_means_every_update():
    assert load_config({}).reset_interval == 60.0
    assert load_config({"KEYLIGHTS_RESET_INTERVAL": "0"}).reset_interval == 0.0
