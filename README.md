# aw-keylight

Per-key status lights for Alienware laptops on Linux: a daemon drives individual keys of the
AlienFX keyboard as indicators for network, CPU/GPU temperature, microphone mute, and local
inference servers (llama.cpp, Ollama).

Tested only on an Alienware m15 R2 (keyboard `0d62:0a1c`, AlienFX v5; power button chip `187c:0550`).

## Requirements

- Linux with hidraw, Python 3.10+, write access to the hidraw nodes of both chips.
- NetworkManager (`nmcli`) for ESC, PipeWire (`wpctl`) for the mute key.
- F1: NVIDIA GPU with `nvidia-smi`. GPU temperature: hwmon sensor labelled `GPU` (`alienware_wmi`).

## Install

    git clone https://github.com/jeffbai996/aw-keylight.git ~/repos/keylights
    cd ~/repos/keylights
    python3 -m venv venv
    venv/bin/pip install -r requirements.txt
    cp .env.example .env
    sudo cp udev/70-aw-keylight.rules /etc/udev/rules.d/
    sudo udevadm control --reload && sudo udevadm trigger

## Usage

    venv/bin/python -m keylights --seconds 30      # trial run
    ln -s "$PWD"/systemd/keylights.service ~/.config/systemd/user/
    systemctl --user enable --now keylights
    venv/bin/pytest

The unit expects the checkout at `~/repos/keylights`; edit `WorkingDirectory` and `ExecStart`
otherwise. Settings go in `.env` or a drop-in (`systemctl --user edit keylights`).

Optional: stop the daemon while the keyboard backlight is off (`systemd/keylights-follow.*`
watch the `KEYLIGHTS_BACKLIGHT_STATE` file):

    systemctl --user link ~/repos/keylights/systemd/keylights-follow.path ~/repos/keylights/systemd/keylights-follow.service
    systemctl --user enable --now keylights-follow.path keylights-follow.service

## Keys

| Key | Shows |
|---|---|
| ESC | Blue while the internet is up, dipping with traffic. Dark on no/limited connectivity or captive portal. |
| F1 | Local GPU activity: green blink while working, blue idle, red unreadable. |
| F2–F5 | CPU temperature bar (1 key below 65 °C, 2 from 65, 3 from 75, 4 from 85), green at 45 °C to red at 90 °C. |
| F6–F9 | GPU temperature bar. |
| F10 | Red while the default microphone is muted. |
| Inference keys | One per server, from `KEYLIGHTS_LIGHTS`. |
| Power button | Green if any server is up, amber while any llama.cpp server reads a prompt, off if none. |

Inference key colours:

| Colour | Meaning |
|---|---|
| Red | Server unreachable. |
| Dark | Nothing loaded, or host in gamemode. |
| Blue | Model loaded, idle. |
| Amber | Prompt being read. llama.cpp: blink rate follows prompt speed, fast flash for a large uncached prompt. Ollama: fixed blink until the first token. |
| Steady green | llama.cpp: prompt read, first token pending. |
| Green flicker | Generating (flicker follows tokens), or GPU power draw for work no request counter sees. |

Unconfigured keys keep the keyboard's colours. On exit managed keys return to `KEYLIGHTS_BASE_COLOR`
and the power button to green.

## Inference keys

`KEYLIGHTS_LIGHTS` is a comma-separated list of `KEY=kind:url[#selector]`:

    KEYLIGHTS_LIGHTS=DEL=llama:http://localhost:8080#my-model,END=ollama:http://gpu-box:11434

ESC and F1–F10 are reserved. Other names: F11, F12, HOME, END, DEL, PGUP, PGDOWN, arrows.

**`llama`**: llama.cpp server or router. Reads `/slots` with `autoload=false` (querying an unloaded
model on a router loads it). `#model` limits the key to one router model, read from its slots
alone. Without it, the loaded models come from `/models`. Any slot reading a prompt counts as
reading; the large-prompt flash uses the largest slot, not the sum. A silent server keeps its last
activity for 5 seconds, then shows idle until it answers.

**`ollama`**: reads `/api/ps`; any resident model (embedding included) counts as loaded. Ollama
reports no activity, so live amber/green needs a proxy serving `GET /_gate/activity` on the same
base URL:

    {"in_flight": 1, "awaiting_first_event": 0, "events_total": 41207}

(open requests; requests with no streamed chunk yet; running chunk count). Without it the key
shows only red, dark and blue. A host silent for 5 seconds shows red.

**GPU overlay and `gpu` keys.** `#gpu=<status url>@<host>` on an Ollama key, or a `gpu` key:

    END=ollama:http://gpu-box:11434#gpu=https://status.example@gpu-box
    HOME=gpu:https://status.example#gpu-box

Both read `GET <status url>/api/telemetry?hosts=<host>[,...]`; lights on one status server share a
request. Expected response:

    {"hosts": [{"host": "gpu-box", "ok": true, "sample_state": "healthy",
                "sample_age_sec": 1.2, "gamemode": false,
                "gpu": [{"power_draw_w": 140.0, "power_limit_w": 350.0, "pstate": "P2"}]}]}

Working means out of `P8` and drawing at least a fifth of the power limit. `gamemode: true` shows
dark instead of red. Samples older than 30 s (3 min in gamemode) are ignored.

## Settings

Defaults are in `keylights/config.py`; `.env.example` has a starter file.

| Variable | Meaning |
|---|---|
| `KEYLIGHTS_LIGHTS` | Inference keys, as above. |
| `KEYLIGHTS_LLAMA_URL` | Shorthand for `DEL=llama:<url>` when `KEYLIGHTS_LIGHTS` is unset. |
| `KEYLIGHTS_BASE_COLOR` | Colour managed keys return to on exit. |
| `KEYLIGHTS_TICK_HZ` | Render rate. |
| `KEYLIGHTS_DEL_BLINK_CAP` | Maximum blinks per second on inference keys. |
| `KEYLIGHTS_ESC_BLINK_CAP` | Maximum dips per second on ESC. |
| `KEYLIGHTS_POLL_INTERVAL` | Seconds between server and network reads. |
| `KEYLIGHTS_OLLAMA_PS_INTERVAL` | Seconds an Ollama resident-model list is reused (activity counters are read every poll). |
| `KEYLIGHTS_SLOW_POLL_INTERVAL` | Seconds between temperature and mute reads. |
| `KEYLIGHTS_COMPACT_TOKENS` | Uncached prompt size that counts as large. |
| `KEYLIGHTS_COMPACT_BLINK` | Flash rate for a large prompt. |
| `KEYLIGHTS_RESET_INTERVAL` | Seconds between full keyboard resyncs; `0` = full sequence every update. |
| `KEYLIGHTS_STALL_BACKOFF` | Seconds of reduced update rate after a write stalls; `0` = off. |
| `KEYLIGHTS_STALL_RATE` | Maximum updates per second during that time. |
| `KEYLIGHTS_POWER_ZONE` | AlienFX zone of the power button. |
| `KEYLIGHTS_BACKLIGHT_STATE` | Optional file holding `on`, `dim` or `off`. |
| `KEYLIGHTS_DIM_BOOST` | Brightness multiplier for ESC and inference keys while the file says `dim`. |

A blink needs two frames: a cap above half the render rate cannot be drawn.

## Protocol

AlienFX v5 feature reports (ID `0xCC`) via `HIDIOCSFEATURE` on the hidraw node: `CC 94` reset,
`CC 8C 02 00` + up to 15 × `key+1,r,g,b` colours, `CC 8C 13` loop, `CC 8B 01 FF` commit. On the
m15 R2 the colour report alone applies. The daemon sends changed keys only, and the full sequence on
the first update, after a failed write, and every `KEYLIGHTS_RESET_INTERVAL`. Writes are volatile
(nothing stored in flash). Key IDs follow the alienfx-tools layout for `0d62:0a1c`.

The power button (`187c:0550`, API v4) takes a one-step static colour, written only on change, at
most every 2 s.

Each server polls on its own thread, local readings on another; rendering never waits on the network.

## Troubleshooting

- **Keyboard freezes ~5 s; log shows `Keyboard write failed at <step> after 5.xs`.** The controller
  stops answering control transfers (the kernel's 5 s timeout can't be shortened), and it also
  handles keystrokes. Stalls grow with write traffic and with other traffic on the same USB
  controller (the m15 R2's Bluetooth radio shares it). The daemon backs off after a stall; lower
  the blink caps or render rate, or move wireless receivers to another controller.
- **A key never lights.** Check the key name and hidraw permissions (`ls -l /dev/hidraw*`, udev rule).
- **An inference key stays red.** URL unreachable; try `curl <url>/models` (llama.cpp) or
  `curl <url>/api/ps` (Ollama).
- **Never write `/sys/class/leds/dell::kbd_backlight` on Alienware hardware.** The SMBIOS path can
  wedge the lighting controller until full power removal.

## License

MIT. See [LICENSE](LICENSE).
