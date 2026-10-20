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
        "--dry-run",
        action="store_true",
        help="Detect the board and report the next card without tapping.",
    )
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Wait for manually opened levels and continue after each completed board.",
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
    agent = TerraminerRockAgent(
        TapExecutor(phone),
        tap_interval=args.tap_interval,
        dry_run=args.dry_run,
    )
    controller = TerraminerRockController(
        agent,
        perception,
        agent.tap_executor,
        continuous=args.continuous,
    )
    try:
        stream.start()
        while not controller.finished:
            if controller.state is ControllerState.AWAITING_NEXT_LEVEL:
                input(
                    "Finish the ad or open the next rock board, then press Enter "
                    "(Ctrl+C to stop): "
                )
                controller.acknowledge_next_level()
                continue
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
