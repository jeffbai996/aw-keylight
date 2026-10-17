import os
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "systemd" / "keylights-follow.sh"


def run_follow(tmp_path, initial, change_during_first_call=None):
    """Runs the follow script with a fake systemctl that logs its calls and, on the first call,
    can rewrite the state file, as a key press during a slow stop does."""
    state = tmp_path / ".local/state/kbd-light/state"
    state.parent.mkdir(parents=True)
    state.write_text(initial + "\n")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "calls"
    change = f'[ -e "{log}" ] || printf "{change_during_first_call}\\n" > "{state}"' if change_during_first_call else ":"
    fake = bin_dir / "systemctl"
    fake.write_text(f'#!/bin/sh\n{change}\necho "$*" >> "{log}"\n')
    fake.chmod(0o755)
    env = {**os.environ, "HOME": str(tmp_path), "PATH": f"{bin_dir}:{os.environ['PATH']}"}
    subprocess.run(["sh", str(SCRIPT)], env=env, check=True, timeout=10)
    return log.read_text().splitlines()


def test_backlight_off_stops_keylights_and_on_or_dim_starts_it(tmp_path):
    assert run_follow(tmp_path / "a", "off") == ["--user stop keylights.service"]
    assert run_follow(tmp_path / "b", "dim") == ["--user start keylights.service"]


def test_a_press_while_keylights_is_stopping_is_not_missed(tmp_path):
    # Fn+F12 pressed quickly: the state turns on again while the stop for "off" is still running.
    assert run_follow(tmp_path, "off", change_during_first_call="on") == [
        "--user stop keylights.service",
        "--user start keylights.service",
    ]
