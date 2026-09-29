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
