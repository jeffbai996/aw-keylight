# keylights

Status lights for an Alienware m15 R2 (AlienFX v5 keyboard controller `0d62:0a1c`
and lighting chip `187c:0550`).

| Light | Shows |
|---|---|
| Inference keys | Each key listed in `KEYLIGHTS_LIGHTS` (for example DEL, HOME, END). See below. |
| ESC | Network traffic on this machine. Faint blue when idle, blinks with traffic (up to `KEYLIGHTS_ESC_BLINK_CAP` blinks/s). |
| F1-F4 | CPU temperature. Green at 45 C through yellow to red at 90 C. |
| F5 | Red while the default microphone is muted. |
| Power button | Summary of the inference lights as a steady colour: green when any host is up, amber while any llama.cpp server reads a prompt, off when none is up. |

Inference lights are configured per key with `KEYLIGHTS_LIGHTS`, a comma-separated list of
`KEY=kind:url[#selector]`:

    KEYLIGHTS_LIGHTS=DEL=llama:http://host:8080#model,HOME=ollama:http://host:11434#gpu=https://status.example/squad@host-a

Every inference light uses the same colours:

| Colour | Meaning |
|---|---|
| Dark | The host is unreachable, or its GPU is vacant (gamemode stops Ollama). |
| Baby blue | Idle: the host is up and nothing is running, whether its model is loaded or parked. |
| Amber | A prompt is being read, and only that. It blinks at a rate that follows prompt speed, holds steady when progress stalls, and flashes at `KEYLIGHTS_COMPACT_BLINK` when the prompt is large (`KEYLIGHTS_COMPACT_TOKENS` or more uncached tokens). |
| Steady green | The prompt is read and the first token is pending. |
| Green flicker | Working: flickering with generated tokens, or with the GPU's power draw. |
| White pulse | Ollama lights only: a request just finished. |

- `llama` reads a llama.cpp server's slots. `#model` limits it to one model of a router. A
  server running several slots is read slot by slot: a prompt being read on any slot shows as
  reading, and compaction is judged on the largest slot, never on the sum.
- `ollama` reads which models are loaded. Ollama itself reports no busy state, so live activity
  comes from the gate in front of it (`GET /_gate/activity` on an ollama-gate): while a request
  has streamed nothing yet the key blinks amber, and while events stream it flickers green at
  their rate. `#gpu=<telemetry url>@<host>` adds the host's GPU activity, which shows work the
  gate cannot see, such as embeddings and whisper, as green flicker. Embedding models are not
  counted as a loaded chat model. A host silent for 5 seconds goes dark, because with Ollama
  stopped the gate hangs instead of refusing on a WSL host with mirrored networking.
- `gpu` shows only a host's GPU, from a fleet status server's `/api/telemetry`. The URL is that
  server's base and `#host` names the host, for example
  `HOME=gpu:https://status.example/squad#host-a`. Power draw and performance state are used
  because utilisation, sampled coarsely, misses short bursts.
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
