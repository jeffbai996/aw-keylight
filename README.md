# keylights

Status lights for an Alienware m15 R2 (AlienFX v5 keyboard controller `0d62:0a1c`
and lighting chip `187c:0550`).

| Light | Shows |
|---|---|
| DEL | Inference server. Amber while a prompt is read: blinking at a rate that follows prompt speed while it advances, steady when it has stalled. Steady green while the first token is pending and when idle or done. Flickers green with generated tokens. Off when the server is down. A job that has evaluated `KEYLIGHTS_COMPACT_TOKENS` or more uncached prompt tokens is treated as a compaction: amber for its whole run, flashing at `KEYLIGHTS_COMPACT_BLINK` while the prompt is read, then flickering with tokens. |
| ESC | Network traffic on this machine. Faint blue when idle, blinks with traffic (up to `KEYLIGHTS_ESC_BLINK_CAP` blinks/s). |
| F1-F4 | CPU temperature. Green at 45 C through yellow to red at 90 C. |
| F5 | Red while the default microphone is muted. |
| Power button | Summary of the inference lights as a steady colour: green when any server is up, amber while any llama.cpp server reads a prompt, off when none is up. |

Inference lights are configured per key with `KEYLIGHTS_LIGHTS`, a comma-separated list of
`KEY=kind:url[#model]`:

    KEYLIGHTS_LIGHTS=DEL=llama:http://host:8080#model,HOME=ollama:http://host:11434

- `llama` reads a llama.cpp server's slots and shows every state described above. `#model`
  limits it to one model of a router. A server running several slots is read slot by slot:
  a prompt being read on any slot shows as reading, and compaction is judged on the largest
  slot, never on the sum.
- `ollama` shows green while a chat model is loaded and pulses white when a request finishes.
  Ollama reports no busy state. Embedding models are ignored, and the key is dark when no
  chat model is loaded or the server is unreachable.
- `KEYLIGHTS_LLAMA_URL` alone still means "DEL shows this llama.cpp server".
- ESC and F1-F5 belong to the status lights and cannot be used. Keys that are not configured
  keep the keyboard's own colours.

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
