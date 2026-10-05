# aw-keylight

Per-key status lights for Alienware laptops on Linux. A small daemon drives individual keys of
the AlienFX keyboard as live indicators: network traffic, CPU temperature, microphone mute, and the
state of local inference servers (llama.cpp and Ollama), including whether a prompt is being read,
tokens are being generated, or the GPU is busy.

Built and tested on an Alienware m15 R2 (keyboard controller `0d62:0a1c`, AlienFX API v5, and the
lighting chip `187c:0550` for the power button). Other Alienware models with the same keyboard
controller may work; nothing else is tested.

## What the keys show

| Key | Shows |
|---|---|
| ESC | Network. Lit blue while the internet is up, dipping to dim blue with traffic. Dark when NetworkManager reports no internet, limited connectivity or a captive portal. |
| F1–F4 | CPU temperature, green at 45 °C through yellow to red at 90 °C. |
| F5 | Red while the default microphone is muted. |
| Inference keys | One key per server, set in `KEYLIGHTS_LIGHTS` (for example DEL, END, HOME). |
| Power button | A steady summary: green when any inference server is up, amber while any llama.cpp server reads a prompt, off when none is up. |

Every inference key uses the same colours:

| Colour | Meaning |
|---|---|
| Red | The server is unreachable. |
| Dark | Nothing is loaded, or the host is in gamemode and its GPU has been handed to a game. |
| Blue | Idle: a model is loaded and nothing is running. |
| Amber | A prompt is being read. On llama.cpp it blinks at a rate that follows prompt speed and flashes fast when the uncached part of the prompt is large. On Ollama it blinks at a fixed rate until the first token arrives. |
| Steady green | llama.cpp only: the prompt is read and the first token is pending. |
| Green flicker | Working: flickering with generated tokens, or with GPU power draw for work no request counter sees (embeddings, speech-to-text). |

Keys that are not configured keep the keyboard's own colours. On exit the managed keys return to
`KEYLIGHTS_BASE_COLOR` and the power button to green.

## Requirements

- Linux with hidraw, and Python 3.10 or later.
- Write access to the keyboard's hidraw node, and to the power-button chip's for that light.
- NetworkManager (`nmcli`) for the ESC online check, and PipeWire (`wpctl`) for the mute key.

## Install

    git clone https://github.com/jeffbai996/aw-keylight.git ~/repos/keylights
    cd ~/repos/keylights
    python3 -m venv venv
    venv/bin/pip install -r requirements.txt
    cp .env.example .env            # then edit .env
    sudo cp udev/70-aw-keylight.rules /etc/udev/rules.d/
    sudo udevadm control --reload && sudo udevadm trigger

Try it for 30 seconds:

    venv/bin/python -m keylights --seconds 30

Run it as a user service. The unit expects the checkout at `~/repos/keylights`; edit
`WorkingDirectory` and `ExecStart` if yours is elsewhere.

    ln -s "$PWD"/systemd/keylights.service ~/.config/systemd/user/
    systemctl --user enable --now keylights

Settings can live in `.env` or in a systemd drop-in (`systemctl --user edit keylights`, with
`Environment=` lines).

## Configuring inference keys

`KEYLIGHTS_LIGHTS` is a comma-separated list of `KEY=kind:url[#selector]`:

    KEYLIGHTS_LIGHTS=DEL=llama:http://localhost:8080#my-model,END=ollama:http://gpu-box:11434

ESC and F1–F5 belong to the built-in lights and cannot be used. Other key names: F6–F12, HOME, END,
DEL, PGUP, PGDOWN and the arrow keys.

### `llama`: llama.cpp server or router

Reads `/slots`, always with `autoload=false`, because querying an unloaded model on a router loads
it. `#model` limits the key to one model of a router, and that model is read from its slots alone:
the router answers 400 for an unloaded model, so the key needs no model listing, which waits on every
upstream the router knows. Without `#model` the key reads `/v1/models` and queries each loaded model. A server with several slots is read slot by slot: any slot reading a prompt shows
as reading, and the large-prompt flash is judged on the largest slot, never on the sum. A busy
server can stop answering, so a silent server keeps showing its last activity for 5 seconds; after
that the key shows it idle until it answers again.

### `ollama`: Ollama

Reads `/api/ps` for which models are resident. Any resident model counts as loaded, an embedding
model included.

Ollama reports no request activity of its own. For live amber and green, put a proxy in front of
it that serves `GET /_gate/activity` on the same base URL:

    {"in_flight": 1, "awaiting_first_event": 0, "events_total": 41207}

`in_flight` is the number of open requests, `awaiting_first_event` how many have streamed nothing
yet, and `events_total` a running count of streamed chunks (about one per token). Without the
endpoint the key still shows red, dark and blue.

A host that stops answering for 5 seconds is shown as unreachable. This covers proxies that hang
rather than refuse when Ollama is stopped.

### GPU overlay and `gpu` keys

An Ollama key can take a GPU overlay, `#gpu=<status url>@<host>`, and a `gpu` key shows a GPU alone:

    END=ollama:http://gpu-box:11434#gpu=https://status.example@gpu-box
    HOME=gpu:https://status.example#gpu-box

Both read `GET <status url>/api/telemetry`, which must return:

    {"hosts": [{"host": "gpu-box", "ok": true, "sample_state": "healthy",
                "sample_age_sec": 1.2, "gamemode": false,
                "gpu": [{"power_draw_w": 140.0, "power_limit_w": 350.0, "pstate": "P2"}]}]}

The card counts as working when it is out of `P8` and drawing at least a fifth of its power limit,
because some cards park in a high performance state at low power. The flicker speeds up with power
draw. `gamemode: true` turns the key dark instead of red when Ollama has been stopped for a game.
Samples older than 30 seconds are ignored, or older than 3 minutes for a host in gamemode, which is
usually sampled less often.

## Settings

| Variable | Default | Meaning |
|---|---|---|
| `KEYLIGHTS_LIGHTS` | none | Inference keys, as above. |
| `KEYLIGHTS_LLAMA_URL` | none | Shorthand for `DEL=llama:<url>` when `KEYLIGHTS_LIGHTS` is unset. |
| `KEYLIGHTS_BASE_COLOR` | `ffffff` | Colour managed keys return to on exit. |
| `KEYLIGHTS_TICK_HZ` | `50` | Render rate. |
| `KEYLIGHTS_DEL_BLINK_CAP` | `20` | Maximum blinks per second on inference keys. |
| `KEYLIGHTS_ESC_BLINK_CAP` | `10` | Maximum dips per second on ESC. |
| `KEYLIGHTS_POLL_INTERVAL` | `0.5` | Seconds between reads of servers and network counters. |
| `KEYLIGHTS_SLOW_POLL_INTERVAL` | `2` | Seconds between reads of temperature and mute state. |
| `KEYLIGHTS_COMPACT_TOKENS` | `12288` | Uncached prompt size that counts as large. |
| `KEYLIGHTS_COMPACT_BLINK` | `10` | Flash rate for a large prompt. |
| `KEYLIGHTS_RESET_INTERVAL` | `60` | Seconds between full keyboard resyncs. `0` sends the full sequence with every update. |
| `KEYLIGHTS_POWER_ZONE` | `1` | AlienFX zone of the power button. |
| `KEYLIGHTS_BACKLIGHT_STATE` | `~/.local/state/kbd-light/state` | Optional file holding `on`, `dim` or `off`. |
| `KEYLIGHTS_DIM_BOOST` | `1.6` | Brightness multiplier for ESC and inference keys while that file says `dim`. |

A blink needs two frames, so a cap above half the render rate cannot be drawn.

## How it works

The keyboard is driven with AlienFX v5 feature reports (report ID `0xCC`) through `HIDIOCSFEATURE`
on its hidraw node:

| Report | Purpose |
|---|---|
| `CC 94` | Reset |
| `CC 8C 02 00` followed by up to 15 blocks of `key+1, r, g, b` | Set key colours |
| `CC 8C 13` | Loop |
| `CC 8B 01 FF` | Commit |

On the m15 R2, colours apply from the colour report alone. The daemon sends only the keys that
changed, as one colour report, and sends the full sequence (reset, colour, loop, commit) as a
resync: on the first update, after any failed write, and at most every `KEYLIGHTS_RESET_INTERVAL`
seconds. These writes are volatile: nothing is stored in the keyboard's flash, and the colours do
not survive a power cycle.

The power button is on a separate chip (`187c:0550`, AlienFX API v4) and is programmed with a
one-step static colour. It is written only when its state changes, at most once every 2 seconds.

Polling runs on its own thread. Rendering runs at `KEYLIGHTS_TICK_HZ` and never waits on the
network.

## Troubleshooting

**The keyboard freezes for about 5 seconds.** The controller sometimes stops answering control
transfers, and the kernel's control timeout of 5 seconds cannot be shortened from user space. The
journal shows `Keyboard write failed at <step> after 5.xs`. The same controller handles keystrokes,
and lighting can only be written over control transfers. Stalls become more frequent with more write
traffic and with other traffic on the same USB controller. On the m15 R2 the internal Bluetooth
radio shares it, and stalls rose sharply while a Bluetooth mouse was connected. To reduce them,
lower the blink caps or the render rate, or use a wireless receiver on a different USB controller.

**A key never lights.** Check the key name, and that the user can open the hidraw node
(`ls -l /dev/hidraw*` and the udev rule above).

**An inference key stays red.** The URL is unreachable from this machine. For llama.cpp, check
`curl <url>/v1/models`; for Ollama, `curl <url>/api/ps`.

## Notes

- Never write `/sys/class/leds/dell::kbd_backlight` on Alienware hardware. The SMBIOS path has
  wedged the lighting controller until a full power removal.
- The lights are a status display, not a meter. Rates are approximations from server counters and
  `/proc/net/dev`.
- Key IDs follow the alienfx-tools layout for the `0d62:0a1c` controller.

## Tests

    venv/bin/pytest

## License

MIT. See [LICENSE](LICENSE).
