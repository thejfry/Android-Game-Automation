from typing import Tuple
import threading
import time
import csv
from datetime import datetime
import statistics

try:
    import msvcrt
except ImportError:
    msvcrt = None

try:
    import ctypes
    import sys
    # On Windows, boost timer resolution to 1ms for more precise sleep
    if sys.platform == "win32":
        winmm = ctypes.windll.winmm
        winmm.timeBeginPeriod(1)
except Exception:
    # If timer resolution boost fails, continue anyway
    pass

from game_automation.core.agent import BaseAgent
from game_automation.core.events import StateBus

DEFAULT_CLICK_OFFSETS = [3.700, 7.000, 10.050, 12.900, 15.580, 18.070, 20.460, 22.775, 24.940, 27.060]

class AtlasGolfAgent(BaseAgent):
    """Agent for the Atlas Golf game that requires no perception.

    Behavior:
    - Waits for Enter presses on stdin.
    - On the first Enter: starts the timer.
    - On each subsequent Enter: taps at the configured bbox center,
      prints elapsed time since the first Enter and since the previous Enter.
    """

    def __init__(self, phone, state_bus: StateBus, bbox: Tuple[int, int, int, int], csv_path: str | None = None, auto: bool = False, click_offsets: list[float] | None = None, adjust_step: float = 0.02, auto_alpha: float = 0.5):
        # tick_hz is unused for this agent but BaseAgent requires it
        super().__init__(phone, state_bus, tick_hz=1)
        self.bbox = bbox
        x1, y1, x2, y2 = bbox
        self.click_point = ((x1 + x2) // 2, (y1 + y2) // 2)

        self._first_time: float | None = None
        self._prev_time: float | None = None
        # Records will store dicts with computed values at each click
        # fields: index(int), iso(str), epoch(float), elapsed(float), gap(float)
        self._records: list[dict] = []
        self._sessions: list[list[dict]] = []
        self.csv_path = csv_path
        self.auto = auto

        if click_offsets is None:
            self.click_offsets = DEFAULT_CLICK_OFFSETS
        else:
            # ensure a list of floats
            self.click_offsets = list(map(float, click_offsets))
            if len(self.click_offsets) != 10:
                self.click_offsets = DEFAULT_CLICK_OFFSETS

        self.adjust_step = float(adjust_step)
        self.auto_alpha = float(auto_alpha)

        self._thread: threading.Thread | None = None

    def start(self):
        """Start the agent's stdin-listener thread."""
        if self._thread and self._thread.is_alive():
            return
        # reset timing state when starting
        self._first_time = None
        self._prev_time = None
        self._records = []
        self.running = True
        self._thread = threading.Thread(target=self._input_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self.running = False
        if self._thread:
            try:
                # joining briefly to allow thread to exit cleanly
                self._thread.join(timeout=0.2)
            except Exception:
                pass

        try:
            self._print_timing_table()
        except Exception:
            pass

    def _input_loop(self):
        print("AtlasGolfAgent ready. Press Enter to start/trigger (Ctrl+C to exit).")
        while self.running:
            try:
                # Before waiting, show scheduled offsets if session not started
                if self._first_time is None:
                    print("Waiting for you to press enter to start the session. Scheduled offsets:", [f"{o:.3f}" for o in self.click_offsets])
                # Wait for an Enter press (empty line). This blocks until user presses Enter.
                line = input()
                _ = line
            except (EOFError, KeyboardInterrupt):
                # Stop on EOF or Ctrl+C
                break

            # If user typed 'N' or 'n', end the current session and start a new one
            if line.strip().lower() == "n":
                if self._records:
                    print("Closing current session and starting a new one...")
                    self._end_session()
                else:
                    print("No active session to close.")
                continue

            now = time.time()
            iso = datetime.fromtimestamp(now).isoformat()
            if self._first_time is None:
                self._first_time = now
                self._prev_time = now
                # first record: elapsed=0, gap=0
                rec = {
                    "index": 0,
                    "iso": iso,
                    "epoch": now,
                    "elapsed": 0.0,
                    "gap": 0.0,
                }
                self._records.append(rec)
                print("Timer started.")
                # If auto mode is enabled, perform an immediate tap at session start
                if self.auto:
                    cx, cy = self.click_point
                    try:
                        # Tap corresponding to the initial press (index 0)
                        self.phone.tap(cx, cy)
                    except Exception as e:
                        print(f"Tap failed: {e}")
                    print(f"Click {rec['index']}: Elapsed since first: {rec['elapsed']:.3f}s; since previous: {rec['gap']:.3f}s")
                    # schedule clicks at specified offsets (seconds after start)
                    start_t = self._first_time
                    # use a temporary copy so interactive adjustments don't persist across sessions
                    offsets = list(self.click_offsets)
                    for idx_offset in range(len(offsets)):
                        target = start_t + float(offsets[idx_offset])
                        self._wait_until(target, idx_offset, offsets)
                        now2 = time.time()
                        rec2 = self._append_record(now2)
                        print(f"Click {rec2['index']}: Elapsed since first: {rec2['elapsed']:.3f}s; since previous: {rec2['gap']:.3f}s")
                        cx, cy = self.click_point
                        try:
                            self.phone.tap(cx, cy)
                        except Exception as e:
                            print(f"Tap failed: {e}")
                    # end session automatically after auto clicks and allow tuning
                    self._end_session()
                    continue
                else:
                    continue

            # Subsequent enter presses: compute elapsed and gap and record
            rec = self._append_record(now)
            print(f"Elapsed since first: {rec['elapsed']:.3f}s; since previous: {rec['gap']:.3f}s")

            cx, cy = self.click_point
            try:
                self.phone.tap(cx, cy)
            except Exception as e:
                print(f"Tap failed: {e}")

        print("AtlasGolfAgent input loop exiting.")

    def _end_session(self):
        """Finalize the current session: print table, store session, reset state."""
        if not self._records:
            return
        # Print session table
        self._print_session_table(self._records)
        # Store session
        self._sessions.append(list(self._records))
        # If we've collected multiple sessions, print aggregated summary
        if len(self._sessions) >= 2:
            self._print_aggregated_summary(self._sessions)
        # Prompt user to tune offsets after each session
        try:
            self._prompt_tune()
        except Exception:
            pass
        # Reset for next session
        self._first_time = None
        self._prev_time = None
        self._records = []

    def tick(self, state):
        # No periodic tick logic required for this agent.
        return

    def _append_record(self, now: float):
        elapsed_since_first = now - (self._first_time or now)
        elapsed_since_prev = now - (self._prev_time or now)
        rec = {
            "index": len(self._records),
            "iso": datetime.fromtimestamp(now).isoformat(),
            "epoch": now,
            "elapsed": elapsed_since_first,
            "gap": elapsed_since_prev,
        }
        self._records.append(rec)
        self._prev_time = now
        return rec

    def _wait_until(self, target_time: float, click_index: int, offsets: list[float] | None = None):
        """Wait until the next click time with high precision, allowing adjustment before second click.

        Uses smaller sleep intervals (2ms) to reduce jitter while minimizing CPU usage.
        If `offsets` is provided, use that list for interactive adjustments so changes
        are session-local and do not persist to `self.click_offsets`.
        """
        offs = offsets if offsets is not None else self.click_offsets
        if click_index == 1 and msvcrt is not None:
            print("Press + or - to adjust the delay before the next click.")

        # Use smaller sleep interval (2ms) for better precision than 50ms
        sleep_interval = 0.002  # 2ms sleep intervals
        
        while True:
            now = time.time()
            wait = target_time - now
            
            if wait <= 0:
                break
            
            # Check for keyboard input (only index 1)
            if click_index == 1 and msvcrt is not None and msvcrt.kbhit():
                ch = msvcrt.getwch()
                if ch == "+":
                    delta = self.adjust_step
                    # Apply delta to all remaining offsets starting from index 1
                    for i in range(1, len(offs)):
                        offs[i] = max(0.0, offs[i] + delta)
                    target_time = self._first_time + offs[1]
                    print(f"Adjusted offsets by +{delta:.3f}s")
                elif ch == "-":
                    delta = -self.adjust_step
                    # Apply delta to all remaining offsets starting from index 1
                    for i in range(1, len(offs)):
                        offs[i] = max(0.0, offs[i] + delta)
                    target_time = self._first_time + offs[1]
                    print(f"Adjusted offsets by {delta:.3f}s")
                continue
            
            # Sleep for the smaller interval or remaining wait time, whichever is smaller
            time.sleep(min(sleep_interval, wait))

    def _print_timing_table(self):
        # Print the current session table
        if not self._records:
            print("No timing events recorded.")
        else:
            self._print_session_table(self._records)

        # Also write CSV for current session if requested
        if self.csv_path and self._records:
            try:
                with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(["index", "iso_timestamp", "epoch", "elapsed_since_first", "gap_from_prev"])
                    for rec in self._records:
                        writer.writerow([
                            rec["index"],
                            rec["iso"],
                            f"{rec['epoch']:.6f}",
                            f"{rec['elapsed']:.6f}",
                            f"{rec['gap']:.6f}",
                        ])
                print(f"Timings written to CSV: {self.csv_path}")
            except Exception as e:
                print(f"Failed to write CSV {self.csv_path}: {e}")

        # Build aggregated sessions list (include current session if present)
        all_sessions = list(self._sessions)
        if self._records:
            all_sessions.append(list(self._records))
        if all_sessions:
            self._print_aggregated_summary(all_sessions)

    def _print_session_table(self, records: list[dict]):
        if not records:
            print("No timing events recorded for this session.")
            return
        print("\nTiming summary (seconds):")
        print(f"{ 'Index':>5} | { 'Elapsed':>10} | { 'Gap':>10} | { 'ISO Timestamp':>25}")
        print("------+-" + "-"*10 + "-+-" + "-"*10 + "-+-" + "-"*25)
        for rec in records:
            print(f"{rec['index']:5d} | {rec['elapsed']:10.3f} | {rec['gap']:10.3f} | {rec['iso']:>25}")
        print("")

    def _print_aggregated_summary(self, sessions: list[list[dict]]):
        # Print number of completed data collections
        print(f"Total sessions collected: {len(sessions)}")
        # Find the maximum index across all sessions
        max_index = 0
        for s in sessions:
            if s:
                max_index = max(max_index, len(s) - 1)
        # Aggregate means and standard deviations for all presses
        print("\nAggregated means and stddevs:")
        print(f"{ 'Index':>5} | { 'Mean Elapsed':>12} | { 'Std Elapsed':>12} | { 'Mean Gap':>12} | { 'Std Gap':>12} | { 'Count':>5}")
        print("------+-" + "-"*12 + "-+-" + "-"*12 + "-+-" + "-"*12 + "-+-" + "-"*12 + "-+-" + "-"*5)
        for i in range(max_index + 1):
            elapsed_vals = []
            gap_vals = []
            for s in sessions:
                if len(s) > i:
                    elapsed_vals.append(s[i]["elapsed"])
                    gap_vals.append(s[i]["gap"])
            if elapsed_vals:
                mean_elapsed = sum(elapsed_vals) / len(elapsed_vals)
                std_elapsed = statistics.stdev(elapsed_vals) if len(elapsed_vals) > 1 else 0.0
                mean_gap = sum(gap_vals) / len(gap_vals) if gap_vals else 0.0
                std_gap = statistics.stdev(gap_vals) if len(gap_vals) > 1 else 0.0
                count = len(elapsed_vals)
                print(f"{i+1:5d} | {mean_elapsed:12.3f} | {std_elapsed:12.3f} | {mean_gap:12.3f} | {std_gap:12.3f} | {count:5d}")
            else:
                print(f"{i+1:5d} | {'-':>12} | {'-':>12} | {'-':>12} | {'-':>12} | {0:5d}")
        print("")

    def _prompt_tune(self):
        """Interactive tuning prompt after a session ends.

        Options presented to the user:
        - Press Enter to skip tuning.
        - Enter a sequence of '+', '-', '=' (one per offset) to adjust each offset by adjust_step.
        - Enter 'auto' to auto-adjust offsets toward measured error using alpha: new = sched + (actual - sched) * alpha
        - Enter comma-separated numeric deltas to apply exact adjustments.
        """
        # Use the most recent session for tuning (last in _sessions)
        if not self._sessions:
            return
        last = self._sessions[-1]
        # Build list of actuals for clicks 1..N (records index 1 onwards)
        actuals = []
        for i in range(1, len(self.click_offsets) + 1):
            if len(last) > i:
                actuals.append(last[i]["elapsed"])
            else:
                actuals.append(None)

        print("\nTuning offsets. Current scheduled offsets (s) vs measured elapsed (s):")
        for i, sched in enumerate(self.click_offsets):
            act = actuals[i] if i < len(actuals) else None
            act_s = f"{act:.3f}" if act is not None else "(missing)"
            print(f"#{i+1}: sched={sched:.3f}, actual={act_s}")

        prompt = (
            f"Enter tuning command: [Enter]=skip, '{len(self.click_offsets)}' char '+-=' seq, 'auto', or comma deltas\n"
            f"  Examples: {'+' * len(self.click_offsets)}  OR  auto  OR  0.01,-0.02{',0' * (len(self.click_offsets) - 2)}\n> "
        )
        try:
            cmd = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            return

        if not cmd:
            print("No tuning applied.")
            return

        if cmd.lower() == "auto":
            # apply proportional correction using alpha
            for i in range(len(self.click_offsets)):
                if i < len(actuals) and actuals[i] is None:
                    continue
                sched = self.click_offsets[i]
                err = (actuals[i] if i < len(actuals) else None) - sched
                if err is not None:
                    delta = err * self.auto_alpha
                    self.click_offsets[i] = max(0.0, sched + delta)
            print("Auto-adjust applied.")
            print("New offsets:", [f"{v:.3f}" for v in self.click_offsets])
            return

        # sequence of +/-/= characters
        if all(ch in "+-=" for ch in cmd) and len(cmd) == len(self.click_offsets):
            for i, ch in enumerate(cmd):
                if ch == "+":
                    self.click_offsets[i] = max(0.0, self.click_offsets[i] + self.adjust_step)
                elif ch == "-":
                    self.click_offsets[i] = max(0.0, self.click_offsets[i] - self.adjust_step)
                # '=' means no change
            print("Offsets updated:", [f"{v:.3f}" for v in self.click_offsets])
            return

        # numeric comma-separated deltas
        parts = [p.strip() for p in cmd.split(",") if p.strip()]
        if len(parts) >= 1:
            try:
                deltas = [float(p) for p in parts]
                for i, d in enumerate(deltas[:len(self.click_offsets)]):
                    self.click_offsets[i] = max(0.0, self.click_offsets[i] + d)
                print("Offsets updated:", [f"{v:.3f}" for v in self.click_offsets])
                return
            except ValueError:
                pass

        print("Unrecognized tuning command; no changes made.")
