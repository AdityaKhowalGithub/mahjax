"""Compact SVG table renderer for the Hong Kong web UI."""

from html import escape
from typing import List

import numpy as np

from mahjax.hong_kong_mahjong.action import Action
from mahjax.hong_kong_mahjong.meld import EMPTY_MELD, Meld

FLOWERS_JA = ["梅", "蘭", "菊", "竹", "春", "夏", "秋", "冬"]
FLOWERS_EN = ["Plum", "Orchid", "Chrys", "Bamboo", "Spring", "Summer", "Autumn", "Winter"]
HONORS_JA = ["東", "南", "西", "北", "白", "發", "中"]
HONORS_EN = ["E", "S", "W", "N", "Wh", "G", "R"]


def tile_text(tile: int, bilingual: bool = False) -> str:
    if 0 <= tile < 27:
        suit = tile // 9
        number = tile % 9 + 1
        return f"{number}{('M', 'P', 'S')[suit]}" if bilingual else f"{number}{('萬', '筒', '索')[suit]}"
    if 27 <= tile < 34:
        return HONORS_EN[tile - 27] if bilingual else HONORS_JA[tile - 27]
    if 34 <= tile < 42:
        return FLOWERS_EN[tile - 34] if bilingual else FLOWERS_JA[tile - 34]
    return ""


def _tile(x: float, y: float, tile: int, *, hidden: bool = False, small: bool = False, bilingual: bool = False) -> str:
    width, height = ((24, 32) if small else (28, 38))
    fill = "#315d72" if hidden else ("#fff4c9" if tile >= 34 else "#f7f3e8")
    label = "" if hidden else tile_text(tile, bilingual)
    font_size = 8 if small and bilingual else (10 if small else 12)
    color = "#b42318" if tile in (33, 34, 38) else "#18212a"
    return (
        f'<g transform="translate({x:.1f},{y:.1f})">'
        f'<rect width="{width}" height="{height}" rx="3" fill="{fill}" stroke="#17242b" stroke-width="1"/>'
        f'<text x="{width / 2}" y="{height / 2 + 4}" text-anchor="middle" font-size="{font_size}" '
        f'font-family="system-ui,sans-serif" fill="{color}">{escape(label)}</text></g>'
    )


def _hand_tiles(state, player: int) -> List[int]:
    result: List[int] = []
    for tile, count in enumerate(np.array(state.players.hand[player], dtype=int)):
        result.extend([tile] * int(count))
    return result


def _meld_tiles(encoded: int) -> List[int]:
    if encoded == int(EMPTY_MELD):
        return []
    action = int(Meld.action(encoded))
    target = int(Meld.target(encoded))
    if action in (Action.PON,):
        return [target] * 3
    if action == Action.OPEN_KAN or 34 <= action < 68:
        return [target] * 4
    start = target - (action - Action.CHI_L)
    return [start, start + 1, start + 2]


def _row(tiles: List[int], x: float, y: float, *, hidden: bool, bilingual: bool, small: bool = False) -> str:
    step = 20 if small else 22
    return "".join(
        _tile(x + index * step, y, tile, hidden=hidden, small=small, bilingual=bilingual)
        for index, tile in enumerate(tiles)
    )


def render_round_svg(
    state,
    *,
    visible_player: int = 0,
    show_all_hands: bool = True,
    tile_style: str = "standard",
) -> str:
    """Render a stable square table with hands, rivers, melds, and flowers."""
    bilingual = tile_style == "bilingual"
    order = [(visible_player + offset) % 4 for offset in range(4)]
    positions = [(150, 558, 0), (574, 150, 90), (490, 82, 180), (82, 490, 270)]
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 720 720" data-ruleset="hkos-v1">',
        '<rect width="720" height="720" rx="12" fill="#17634f"/>',
        '<rect x="172" y="172" width="376" height="376" rx="8" fill="#124c3e" stroke="#d5b66f"/>',
        '<text x="360" y="326" text-anchor="middle" fill="#f7e7ad" font-size="25" font-family="system-ui,sans-serif">HKOS V1</text>',
        f'<text x="360" y="356" text-anchor="middle" fill="#d8e9e2" font-size="15" font-family="system-ui,sans-serif">3 faan minimum · {144 - int(state.round_state.wall_index)} wall tiles</text>',
        f'<text x="360" y="382" text-anchor="middle" fill="#d8e9e2" font-size="14" font-family="system-ui,sans-serif">Round {int(state.round_state.round) + 1} · East seat P{int(state.round_state.dealer) + 1}</text>',
    ]
    for relative, player in enumerate(order):
        x, y, rotation = positions[relative]
        hidden = relative != 0 and not show_all_hands
        hand = _hand_tiles(state, player)
        melds: List[int] = []
        for encoded in np.array(state.players.melds[player], dtype=np.uint16):
            melds.extend(_meld_tiles(int(encoded)))
        flowers = [34 + i for i, held in enumerate(np.array(state.players.flowers[player], dtype=bool)) if held]
        river = [int(t) for t in np.array(state.players.river[player], dtype=int) if int(t) >= 0]
        parts.append(f'<g transform="rotate({rotation} {x} {y})">')
        parts.append(
            f'<text x="{x}" y="{y - 10}" fill="#fff" font-size="13" font-family="system-ui,sans-serif">'
            f'P{player + 1} · {int(state.round_state.score[player])}</text>'
        )
        parts.append(_row(hand, x, y, hidden=hidden, bilingual=bilingual))
        parts.append(_row(melds, x, y + 43, hidden=False, bilingual=bilingual, small=True))
        parts.append(_row(flowers, x + 190, y + 43, hidden=False, bilingual=bilingual, small=True))
        river_x = 250 if relative in (0, 2) else 235
        river_y = 445 if relative == 0 else (235 if relative == 2 else 340)
        if relative in (1, 3):
            river_x, river_y = x - 45, y + 75
        parts.append(_row(river[:18], river_x, river_y, hidden=False, bilingual=bilingual, small=True))
        parts.append("</g>")
    parts.append("</svg>")
    return "".join(parts)


__all__ = ["FLOWERS_EN", "FLOWERS_JA", "render_round_svg", "tile_text"]
