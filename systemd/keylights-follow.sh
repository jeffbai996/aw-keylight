#!/bin/sh
# Start or stop keylights to match the keyboard backlight state that kbd-light writes (on, dim
# or off). With the backlight off nothing keylights draws can be seen, so it is stopped outright.
#
# systemd ignores changes to the state file while this script runs. A change made during a stop
# or start, such as Fn+F12 pressed quickly, would otherwise be missed and leave keylights in the
# wrong state, so the script keeps acting until the state it reads afterwards is the one it
# acted on.
state_file="$HOME/.local/state/kbd-light/state"
read_state() { cat "$state_file" 2>/dev/null; }

state=$(read_state)
while :; do
    if [ "$state" = off ]; then
        systemctl --user stop keylights.service
    else
        systemctl --user start keylights.service
    fi
    now=$(read_state)
    [ "$now" = "$state" ] && exit 0
    state=$now
done
