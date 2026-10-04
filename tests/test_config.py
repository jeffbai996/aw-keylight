import re
from pathlib import Path

from keylights.config import load_config


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
