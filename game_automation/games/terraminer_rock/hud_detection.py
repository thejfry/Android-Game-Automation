"""Read move, time, and match counters from Terraminer Rock's HUD."""

from dataclasses import dataclass
from functools import lru_cache
import logging
import re

import cv2
import numpy as np

logger = logging.getLogger(__name__)

HUD_DIGIT_CONFIDENCE = 0.25
TEMPLATE_SIZE = (24, 36)


@dataclass(frozen=True)
class RockHudStats:
    moves_used: int
    move_limit: int
    seconds_remaining: int
    matches_made: int
    expected_matches: int


class RockHudDetector:
    """Read HUD values from their separate, screen-relative value columns."""

    # Some Android layouts show the status bar, shifting the game HUD down.
    # Try the value row in both fullscreen and status-bar layouts.
    VALUE_ROWS = ((0.145, 0.163), (0.185, 0.207))
    VALUE_COLUMNS = {
        "moves": (0.09, 0.34),
        "time": (0.435, 0.63),
        "matches": (0.70, 0.94),
    }

    @classmethod
    def detect(cls, frame: np.ndarray) -> RockHudStats | None:
        if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("HUD detection requires a BGR image frame.")

        height, width = frame.shape[:2]
        for y_start, y_end in cls.VALUE_ROWS:
            y1, y2 = round(height * y_start), round(height * y_end)
            readings = {}
            for name, (left, right) in cls.VALUE_COLUMNS.items():
                x1, x2 = round(width * left), round(width * right)
                text = cls._read_text(frame[y1:y2, x1:x2])
                logger.debug("HUD %s value recognized as %r.", name, text)
                readings[name] = text

            stats = cls._parse_readings(readings)
            if stats is not None:
                logger.debug("Read Rock HUD stats: %s.", stats)
                return stats
        return None

    @staticmethod
    def _parse_readings(readings: dict[str, str]) -> RockHudStats | None:
        moves = re.fullmatch(r"(\d{1,3})/(\d{1,3})", readings["moves"])
        timer = re.fullmatch(r"(\d{1,3})(?:s|5)?", readings["time"])
        matches = re.fullmatch(r"(\d{1,2})/(\d{1,2})", readings["matches"])
        if moves is None or timer is None or matches is None:
            return None

        moves_used, move_limit = (int(value) for value in moves.groups())
        if moves_used > move_limit and moves.group(1).startswith("7"):
            corrected_moves = int("1" + moves.group(1)[1:])
            if corrected_moves <= move_limit:
                logger.debug(
                    "Correcting HUD moves %d/%d to %d/%d; a leading 1 "
                    "was recognized as 7 and the original exceeds the limit.",
                    moves_used,
                    move_limit,
                    corrected_moves,
                    move_limit,
                )
                moves_used = corrected_moves
        seconds_remaining = int(timer.group(1))
        matches_made, expected_matches = (int(value) for value in matches.groups())
        if (
            move_limit < 1
            or moves_used > move_limit
            or seconds_remaining > 999
            or expected_matches < 1
            or matches_made > expected_matches
        ):
            logger.debug(
                "Rejecting invalid HUD values: moves %d/%d, time %ds, matches %d/%d.",
                moves_used,
                move_limit,
                seconds_remaining,
                matches_made,
                expected_matches,
            )
            return None
        return RockHudStats(
            moves_used,
            move_limit,
            seconds_remaining,
            matches_made,
            expected_matches,
        )

    @classmethod
    def _read_text(cls, region: np.ndarray) -> str:
        if region.size == 0:
            return ""
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        _, binary = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY)

        count, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        minimum_height = max(3, round(binary.shape[0] * 0.23))
        glyphs = []
        for label in range(1, count):
            x, y, glyph_width, glyph_height, area = stats[label]
            if glyph_height < minimum_height or area < 3:
                continue
            glyph = binary[y:y + glyph_height, x:x + glyph_width]
            character, confidence = cls._recognize_glyph(glyph)
            if confidence < HUD_DIGIT_CONFIDENCE:
                logger.debug(
                    "Rejecting HUD character at x=%d: best candidate %r "
                    "has confidence %.3f.",
                    x,
                    character,
                    confidence,
                )
                return ""
            glyphs.append((x, character))
        return "".join(character for _, character in sorted(glyphs))

    @classmethod
    def _recognize_glyph(cls, glyph: np.ndarray) -> tuple[str, float]:
        normalized = _normalize_glyph(glyph)
        best_character = ""
        best_score = -1.0
        for character, templates in _glyph_templates().items():
            for template in templates:
                score = float(
                    cv2.matchTemplate(normalized, template, cv2.TM_CCOEFF_NORMED)[0, 0]
                )
                if score > best_score:
                    best_character = character
                    best_score = score
        return best_character, best_score


def _normalize_glyph(glyph: np.ndarray) -> np.ndarray:
    target_width, target_height = TEMPLATE_SIZE
    height, width = glyph.shape
    scale = min(
        (target_width - 4) / max(width, 1),
        (target_height - 4) / max(height, 1),
    )
    resized = cv2.resize(
        glyph,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
    )
    canvas = np.zeros((target_height, target_width), dtype=np.uint8)
    x = (target_width - resized.shape[1]) // 2
    y = (target_height - resized.shape[0]) // 2
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return canvas


@lru_cache(maxsize=1)
def _glyph_templates() -> dict[str, tuple[np.ndarray, ...]]:
    templates: dict[str, list[np.ndarray]] = {
        character: [] for character in "0123456789/s"
    }
    fonts = (
        cv2.FONT_HERSHEY_SIMPLEX,
        cv2.FONT_HERSHEY_DUPLEX,
        cv2.FONT_HERSHEY_COMPLEX,
        cv2.FONT_HERSHEY_TRIPLEX,
        cv2.FONT_HERSHEY_PLAIN,
    )
    for character in templates:
        for font in fonts:
            for scale in (0.8, 1.0, 1.2):
                for thickness in (1, 2):
                    canvas = np.zeros((56, 48), dtype=np.uint8)
                    (text_width, text_height), baseline = cv2.getTextSize(
                        character,
                        font,
                        scale,
                        thickness,
                    )
                    x = max(0, (canvas.shape[1] - text_width) // 2)
                    y = max(text_height + 1, (canvas.shape[0] + text_height - baseline) // 2)
                    cv2.putText(
                        canvas,
                        character,
                        (x, y),
                        font,
                        scale,
                        255,
                        thickness,
                        cv2.LINE_AA,
                    )
                    points = cv2.findNonZero(canvas)
                    if points is None:
                        continue
                    x, y, glyph_width, glyph_height = cv2.boundingRect(points)
                    templates[character].append(
                        _normalize_glyph(
                            canvas[y:y + glyph_height, x:x + glyph_width]
                        )
                    )
    return {character: tuple(values) for character, values in templates.items()}
