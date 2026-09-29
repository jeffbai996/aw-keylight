import pytest

from keylights.keymap import UnknownKeyError, key_id


def test_keymap_resolves_names_case_insensitively():
    assert key_id("del") == key_id("DEL") == key_id("  Del ")


def test_keymap_rejects_unknown_key_name():
    with pytest.raises(UnknownKeyError, match="NOPE"):
        key_id("nope")


def test_keymap_del_is_15_esc_is_0_f1_to_f5_are_1_to_5():
    assert key_id("DEL") == 15
    assert key_id("ESC") == 0
    assert [key_id(f"F{n}") for n in range(1, 6)] == [1, 2, 3, 4, 5]
