"""Key names to AlienFX v5 key IDs.

IDs come from the alienfx-tools layout for the Darfon keyboard controller
(0d62:0a1c). ESC, F1-F12, HOME, END and DEL are verified on an m15 R2; the rest follow the
same table.
"""

_KEYS = {
    "ESC": 0,
    **{f"F{n}": n for n in range(1, 13)},
    "HOME": 13,
    "END": 14,
    "DEL": 15,
    "PGUP": 113,
    "PGDOWN": 115,
    "UP": 114,
    "LEFT": 133,
    "DOWN": 134,
    "RIGHT": 135,
}


class UnknownKeyError(ValueError):
    pass


def key_id(name: str) -> int:
    normalized = name.strip().upper()
    try:
        return _KEYS[normalized]
    except KeyError:
        raise UnknownKeyError(f"Unknown key name: {normalized}") from None
