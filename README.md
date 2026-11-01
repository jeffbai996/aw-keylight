# aw-keylight

Status lights for the keyboard of an Alienware laptop, on Linux. A daemon turns individual keys into indicators for network traffic, CPU and GPU temperature, microphone mute and the state of local llama.cpp and Ollama servers.

I wrote it for an Alienware m15 R2 and have not tried another machine. A model with the same controller IDs (`0d62:0a1c` for the keyboard, `187c:0550` for the power button) may work.

```text
ESC   F1    F2 F3 F4 F5    F6 F7 F8 F9    F10    F11 F12 HOME END DEL
net   gpu   cpu °C         gpu °C         mic    inference servers
```

The power button shows a summary of the servers. This runs the daemon for 30 seconds with a llama.cpp server on DEL and an Ollama server on END (set up the virtualenv first, see Install):

```bash
KEYLIGHTS_LIGHTS="DEL=llama:http://localhost:8080#my-model,END=ollama:http://gpu-box:11434" \
  venv/bin/python -m keylights --seconds 30
```

## What the keys show

| Key | Shows |
|---|---|
| ESC | The network. Blue while the internet is up, dipping to dim blue with traffic. Dark when NetworkManager reports no internet, limited connectivity or a captive portal. |
| F1 | Whether this machine's NVIDIA GPU is doing any work. Green blinking while it works, blue when idle, red when it cannot be read. |
| F2 to F5 | CPU temperature as a bar. One key is lit below 65 °C, and one more at 65, 75 and 85. |
| F6 to F9 | GPU temperature, as the same bar. |
| F10 | Red while the default microphone is muted. |
| Inference keys | One key per server, chosen in `KEYLIGHTS_LIGHTS`. |
| Power button | Green when any server is up, amber while a llama.cpp server reads a prompt, off when none is up. |

Lit bar keys take the colour of the temperature, from green at 45 °C through yellow to red at 90 °C. Keys that are not configured keep the keyboard's own colours.

Every inference key uses the same colours:

| Colour | Meaning |
|---|---|
| Red | The server is unreachable. |
| Dark | Nothing is loaded, or the host has handed its GPU to a game (gamemode). |
| Blue | Idle. A model is loaded and nothing is running. |
| Amber | A prompt is being read. |
| Steady green | llama.cpp only. The prompt is read and the first token is pending. |
| Green flicker | Generating. About one blink for every two tokens. |

On llama.cpp the amber blinks faster as the prompt is read faster. Amber flashes fast once a prompt has run past `KEYLIGHTS_COMPACT_TOKENS` uncached tokens. The server does not label context compaction, and a very large uncached prompt is the closest sign of it. On Ollama the amber blinks until the first token arrives.

## How it works

The keyboard controller takes 64-byte AlienFX v5 feature reports, sent with `HIDIOCSFEATURE` on its hidraw node:

```text
CC 94                       reset
CC 8C 02 00 + key blocks    colours: up to 15 blocks of key+1, r, g, b
CC 8C 13                    loop
CC 8B 01 FF                 commit
```

On the m15 R2 a colour report alone changes a key. The daemon sends only the keys that changed, and the full sequence as a resync on the first update, after a failed write, and at most once a minute. Every transfer saved is one that cannot stall. The power button is on a separate chip that takes 34-byte reports. It is reprogrammed only when its state changes, because that chip has jammed under heavy write load on other Alienware models.

Each server is polled on its own thread over one kept-alive HTTP connection. A new TLS connection for every poll, twice a second, used most of the daemon's CPU. Rendering never waits on the network, so a server that hangs holds up only its own key.

## Install

You need Linux with hidraw and Python 3.10 or later. NetworkManager (`nmcli`) drives ESC, `wpctl` drives F10, and F1 needs an NVIDIA GPU with `nvidia-smi`.

```bash
git clone https://github.com/jeffbai996/aw-keylight.git ~/repos/keylights
cd ~/repos/keylights
python3 -m venv venv
venv/bin/pip install -r requirements.txt
cp .env.example .env            # then edit .env
sudo cp udev/70-aw-keylight.rules /etc/udev/rules.d/
sudo udevadm control --reload && sudo udevadm trigger
```

The udev rules give the active desktop seat and members of `plugdev` write access to both hidraw nodes. A user service that starts at boot, before any seat is active, needs `plugdev`. To run the daemon as a user service:

```bash
ln -s "$PWD"/systemd/keylights.service ~/.config/systemd/user/
systemctl --user enable --now keylights
```

The unit expects the checkout at `~/repos/keylights`. Edit `WorkingDirectory` and `ExecStart` if yours is elsewhere. Settings go in `.env` or in a systemd drop-in. Every one is an environment variable read in `keylights/config.py`, which also holds the defaults.

### Backlight off

With the keyboard backlight off nothing the daemon draws can be seen, so a second pair of units stops it then and starts it again on `on` or `dim`. They follow `~/.local/state/kbd-light/state`, which holds `on`, `dim` or `off`. A separate tool called kbd-light writes that file. It is not part of this repository.

```bash
systemctl --user link ~/repos/keylights/systemd/keylights-follow.path ~/repos/keylights/systemd/keylights-follow.service
systemctl --user enable --now keylights-follow.path keylights-follow.service
```

The daemon reads the same file and, while it says `dim`, brightens ESC, F1 and the inference keys, which would otherwise vanish.

## Inference keys

`KEYLIGHTS_LIGHTS` is a comma-separated list of `KEY=kind:url[#selector]`, where the kind is `llama`, `ollama` or `gpu`. ESC and F1 to F10 are taken. The free keys are F11, F12, HOME, END, DEL, PGUP, PGDOWN, UP, DOWN, LEFT and RIGHT. The first five are checked on hardware. The rest follow the same key table and are untested.

**llama.cpp.** The key reads the server's `/slots`, always with `autoload=false`, because asking a router about an unloaded model loads it. Add `#model` to watch one model of a router. Without it, the key reads every loaded model the router lists at `/models`. It avoids `/v1/models`, which also waits on every upstream the router knows, and one that is unreachable costs seconds. With several slots, any slot reading a prompt counts as reading.

**Ollama.** The key reads `/api/ps` for the loaded models. Any loaded model counts, an embedding model included. Ollama does not say whether a request is running. For the amber and green states, put a proxy in front of it that serves `GET /_gate/activity` on the same base URL:

```json
{"in_flight": 1, "awaiting_first_event": 0, "events_total": 41207}
```

The fields are the open requests, how many of those have streamed nothing yet, and a running count of streamed chunks, about one per token. Without the endpoint the key still shows red, dark and blue. A host that stays silent for 5 seconds shows as unreachable, which covers a proxy that hangs instead of refusing when Ollama is stopped.

**GPU telemetry.** An Ollama key can take a GPU overlay, `#gpu=<status url>@<host>`, so it also flickers for work the proxy cannot see, such as embeddings and speech-to-text. A `gpu` key shows one host's GPU on its own:

```ini
END=ollama:http://gpu-box:11434#gpu=https://status.example@gpu-box
HOME=gpu:https://status.example#gpu-box
```

Both read `GET <status url>/api/telemetry?hosts=<host>`. Lights on the same status server share one request that names all their hosts, because a request without hosts makes the status server probe every machine it knows. A card counts as working when it is out of `P8` and drawing at least a fifth of its power limit, since some cards sit in a high performance state at low power. In gamemode the host has handed its GPU to a game and stopped Ollama on purpose, so the key goes dark instead of red.

## Troubleshooting

**The log says `Keyboard write failed at <step> after N s`.** The controller stopped answering a control transfer, and the write blocked until it gave up. Stalls come in clusters, while other traffic on the same USB controller is heavy, a Bluetooth radio waking up for one. After a stall the daemon sends at most `KEYLIGHTS_STALL_RATE` updates a second for `KEYLIGHTS_STALL_BACKOFF` seconds. Lower blink caps (`KEYLIGHTS_DEL_BLINK_CAP`, `KEYLIGHTS_ESC_BLINK_CAP`) or a lower render rate (`KEYLIGHTS_TICK_HZ`) mean fewer writes, and fewer chances to stall.

**A key never lights.** Check the key name, and that your user can open the hidraw node (`ls -l /dev/hidraw*`).

**An inference key stays red.** The URL is unreachable from this machine. For llama.cpp try `curl <url>/models`, for Ollama `curl <url>/api/ps`.

The lights are a status display, not a meter. Rates are approximations from server counters and `/proc/net/dev`. Key IDs follow the alienfx-tools layout for the `0d62:0a1c` controller.

## Tests

```bash
venv/bin/pytest
```

## License

MIT. See [LICENSE](LICENSE).
