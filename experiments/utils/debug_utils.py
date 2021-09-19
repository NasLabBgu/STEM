DEBUG_LEVEL = 0


def set_debug_level(level: int) -> bool:
    global DEBUG_LEVEL
    DEBUG_LEVEL = level


def ddprint(debug_level: int, *args, **kwargs):
    if debug_level <= DEBUG_LEVEL:
        print(*args, **kwargs)


def dprint(*args, **kwargs):
    ddprint(1, *args, **kwargs)


def dprint1(*args, **kwargs):
    ddprint(1, *args, **kwargs)


def dprint2(*args, **kwargs):
    ddprint(2, *args, **kwargs)
