"""Service entrypoint: wires the real readers and hardware into the daemon."""
from __future__ import annotations

import argparse
import logging
import os
import signal
import threading

from dotenv import load_dotenv

from .config import load_config
from .daemon import Daemon, run
from .device import HidrawTransport, Keyboard
from .elc import HidrawElcTransport, PowerButton
from .sources import build_sources


def main() -> None:
    parser = argparse.ArgumentParser(prog="keylights")
    parser.add_argument("--seconds", type=float, help="stop after this many seconds (for testing)")
    args = parser.parse_args()

    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config = load_config(os.environ)

    daemon = Daemon(
        build_sources(config.lights, config.backlight_state_path, config.ollama_ps_interval),
        Keyboard(HidrawTransport(), reset_interval=config.reset_interval),
        PowerButton(HidrawElcTransport(), config.power_zone),
        config,
    )

    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    if args.seconds:
        threading.Timer(args.seconds, stop.set).start()
    run(daemon, config, stop)


if __name__ == "__main__":
    main()
