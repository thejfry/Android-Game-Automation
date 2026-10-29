"""In-place terminal progress display that coexists with timestamped logs."""

import logging
import sys
from typing import TextIO

PROGRESS_BAR_WIDTH = 36


def _progress_bar(value: int, total: int) -> str:
    if total <= 0:
        return "-" * PROGRESS_BAR_WIDTH
    filled = round(PROGRESS_BAR_WIDTH * min(max(value, 0), total) / total)
    return "#" * filled + "-" * (PROGRESS_BAR_WIDTH - filled)


def _percentage(value: float) -> int:
    return int(value * 100 + 0.5)


def format_progress_line(
    level: int,
    moves_used: int,
    move_limit: int,
    seconds_remaining: int,
    initial_seconds: int,
    matches_made: int,
    expected_matches: int,
) -> str:
    initial_seconds = max(initial_seconds, 1)
    safe_expected_matches = max(expected_matches, 1)
    elapsed_seconds = max(0, initial_seconds - seconds_remaining)
    moves_percent = min(max(moves_used / max(move_limit, 1), 0.0), 1.0)
    time_percent = min(elapsed_seconds / initial_seconds, 1.0)
    matches_percent = min(max(matches_made / safe_expected_matches, 0.0), 1.0)
    moves_percent_display = _percentage(moves_percent)
    time_percent_display = _percentage(time_percent)
    matches_percent_display = _percentage(matches_percent)

    at_risk = moves_percent > matches_percent or time_percent > matches_percent
    status = "AT RISK" if at_risk else "ON PACE"
    return "\n".join(
        (
            f"Level {level} progress",
            f"Moves used  [{_progress_bar(moves_used, max(move_limit, 1))}] "
            f"{moves_used}/{move_limit} ({moves_percent_display}%)",
            f"Time used   [{_progress_bar(elapsed_seconds, initial_seconds)}] "
            f"{elapsed_seconds}/{initial_seconds}s ({time_percent_display}%)",
            f"Matches     [{_progress_bar(matches_made, safe_expected_matches)}] "
            f"{matches_made}/{expected_matches} ({matches_percent_display}%)",
            f"Pace: {status}",
        )
    )


class LiveProgressDisplay:
    def __init__(self, stream: TextIO = sys.stderr):
        self.stream = stream
        self.enabled = stream.isatty()
        self._text = ""
        self._visible = False

    def update(self, text: str | None) -> None:
        if text is None:
            self.clear()
            self._text = ""
            return
        if text == self._text and self._visible:
            return
        self._text = text
        self._draw()

    def clear(self) -> None:
        if not self.enabled or not self._visible:
            return
        lines = self._text.splitlines()
        if len(lines) > 1:
            self.stream.write(f"\x1b[{len(lines) - 1}A")
        for index in range(len(lines)):
            self.stream.write("\r\x1b[2K")
            if index < len(lines) - 1:
                self.stream.write("\n")
        if len(lines) > 1:
            self.stream.write(f"\x1b[{len(lines) - 1}A")
        self.stream.write("\r")
        self.stream.flush()
        self._visible = False

    def redraw(self) -> None:
        if self._text:
            self._draw()

    def _draw(self) -> None:
        if not self.enabled:
            return
        lines = self._text.splitlines()
        if self._visible:
            if len(lines) > 1:
                self.stream.write(f"\x1b[{len(lines) - 1}A")
            for index, line in enumerate(lines):
                self.stream.write("\r\x1b[2K" + line)
                if index < len(lines) - 1:
                    self.stream.write("\n")
        else:
            self.stream.write("\n".join(lines))
        self.stream.flush()
        self._visible = True


class ProgressStreamHandler(logging.StreamHandler):
    """Keep live progress out of timestamped log lines."""

    def __init__(self, progress: LiveProgressDisplay):
        super().__init__(progress.stream)
        self.progress = progress

    def emit(self, record: logging.LogRecord) -> None:
        self.progress.clear()
        super().emit(record)
        self.progress.redraw()
