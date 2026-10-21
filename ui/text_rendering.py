"""Reuse text surfaces instead of rasterizing the entire editor every frame."""

from functools import lru_cache


@lru_cache(maxsize=2048)
def render_text(font, value, color):
    return font.render(value, True, color)


@lru_cache(maxsize=2048)
def fit_text(font, value, width):
    if width:
        while len(value) > 2 and font.size(value)[0] > width:
            value = value[:-2] + "…"
    return value


@lru_cache(maxsize=512)
def wrap_text(font, value, width):
    lines, line = [], ""
    for word in value.split():
        candidate = f"{line} {word}".strip()
        if line and font.size(candidate)[0] > width:
            lines.append(line)
            line = word
        else:
            line = candidate
    if line:
        lines.append(line)
    return tuple(lines)
