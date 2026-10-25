"""Run Terraminer Rock automation against a connected Android device."""

import argparse
import logging
import time

from game_automation.core.device import MyPhone
from game_automation.core.frame_stream import FrameStream
from game_automation.games.terraminer_gold.tap_executor import TapExecutor

from .agent import TerraminerRockAgent
from .controller import ControllerState, TerraminerRockController
from .perception import TerraminerRockPerception


def main():
    parser = argparse.ArgumentParser(
        description="Play Terraminer's rock memory minigame."
    )
    parser.add_argument("--tap-interval", type=float, default=0.5)
    parser.add_argument(
        "--level-time-seconds",
        type=float,
        default=90.0,
        help="Starting level clock duration used to estimate remaining time (default: 90).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Detect the board and report the next card without tapping.",
    )
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Continue after completed boards and automatically detect the next level.",
    )
    parser.add_argument(
        "--next-level-poll-seconds",
        type=float,
        default=5.0,
        help="Seconds between checks for a new board after an ad (default: 5).",
    )
    parser.add_argument(
        "--next-level-timeout-seconds",
        type=float,
        default=240.0,
        help="Stop if no new board appears after this many seconds (default: 120).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    phone = MyPhone()
    phone.connect_device()
    stream = FrameStream(max_queue=1, device=phone.device)
    perception = TerraminerRockPerception()
    tap_executor = TapExecutor(phone, tap_log_level=logging.DEBUG)
    agent = TerraminerRockAgent(
        tap_executor,
        tap_interval=args.tap_interval,
        dry_run=args.dry_run,
    )
    controller = TerraminerRockController(
        agent,
        perception,
        tap_executor,
        continuous=args.continuous,
        level_time_seconds=args.level_time_seconds,
        next_level_poll_interval_seconds=args.next_level_poll_seconds,
        next_level_timeout_seconds=args.next_level_timeout_seconds,
    )
    try:
        stream.start()
        while not controller.finished:
            frame = stream.get_latest_frame()
            if frame is None:
                time.sleep(0.01)
                continue
            controller.process_frame(frame)
            if args.dry_run and controller.state is ControllerState.PLAYING:
                break
    except KeyboardInterrupt:
        logging.info("Terraminer Rock automation interrupted.")
    finally:
        stream.stop()


if __name__ == "__main__":
    main()
