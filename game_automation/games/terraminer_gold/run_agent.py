"""Run Terraminer Gold automation against a connected Android device."""

import argparse
import logging
import time

from game_automation.core.device import MyPhone
from game_automation.core.frame_stream import FrameStream

from .agent import TerraminerGoldAgent
from .controller import TerraminerGoldController
from .perception import TerraminerGoldPerception
from game_automation.core.tap_executor import TapExecutor


def main():
    parser = argparse.ArgumentParser(description="Solve the current Terraminer Gold board.")
    parser.add_argument("--tap-interval", type=float, default=0.3)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Detect the board and report a route without sending device taps.",
    )
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Auto-advance Play Again screens and detect boards after ad transitions.",
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
        help="Stop if no new board appears after this many seconds (default: 240).",
    )
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    phone = MyPhone()
    phone.connect_device()
    stream = FrameStream(max_queue=1, device=phone.device)
    perception = TerraminerGoldPerception()
    tap_executor = TapExecutor(phone)
    agent = TerraminerGoldAgent(
        tap_executor,
        tap_interval=args.tap_interval,
        dry_run=args.dry_run,
    )
    controller = TerraminerGoldController(
        agent,
        perception,
        tap_executor,
        continuous=args.continuous,
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
    except KeyboardInterrupt:
        logging.info("Terraminer Gold automation interrupted.")
    finally:
        stream.stop()


if __name__ == "__main__":
    main()
