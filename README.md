# keylights

Status lights for an Alienware m15 R2 (AlienFX v5 keyboard controller `0d62:0a1c`
and lighting chip `187c:0550`).

| Light | Shows |
|---|---|
| DEL | Inference server. Amber while a prompt is read: blinking at a rate that follows prompt speed while it advances, steady when it has stalled. Steady green while the first token is pending and when idle or done. Flickers green with generated tokens. Off when the server is down. |
| ESC | Network traffic on this machine. Faint blue when idle, blinks with traffic (up to `KEYLIGHTS_ESC_BLINK_CAP` blinks/s). |
| F1-F4 | CPU temperature. Green at 45 C through yellow to red at 90 C. |
| F5 | Red while the default microphone is muted. |
| Power button | Inference server state as a steady colour: green up, amber reading a prompt, off down. |

On exit the managed keys return to `KEYLIGHTS_BASE_COLOR` and the power button to green.

The backlight level is global, so while `kbd-light` is in dim mode ESC and DEL are scaled up by
`KEYLIGHTS_DIM_BOOST` (default 1.6). The mode is read from `KEYLIGHTS_BACKLIGHT_STATE`
(default `~/.local/state/kbd-light/state`).

Blink caps and the render rate are `KEYLIGHTS_DEL_BLINK_CAP`, `KEYLIGHTS_ESC_BLINK_CAP` and
`KEYLIGHTS_TICK_HZ`. The keyboard accepts about 110 updates a second, so a cap above 55 blinks/s
cannot be drawn.

## Setup

    python3 -m venv venv
    venv/bin/pip install -r requirements.txt
    cp .env.example .env        # set KEYLIGHTS_LLAMA_URL
    venv/bin/python -m keylights --seconds 30
    ln -s "$PWD"/systemd/keylights.service ~/.config/systemd/user/
    systemctl --user enable --now keylights

The user needs write access to both hidraw nodes (a udev `uaccess` rule for `187c:0550` and `0d62:0a1c`).

## Notes

- The inference reader queries `/slots` only for models the server reports as `loaded`,
  with `autoload=false`. Querying an unloaded model on a llama.cpp router loads it.
- The power button chip has jammed under heavy writes on other Alienware models. The
  button is reprogrammed only when its state changes, at most once every 2 seconds.
- Never write `/sys/class/leds/dell::kbd_backlight` on Alienware hardware; the SMBIOS path
  has wedged the lighting controller until a full power removal.
- The lights are a status display, not an accurate meter: rates are approximations
  that follow `n_decoded` counters and `/proc/net/dev` bytes.

## Tests

    venv/bin/pytest
