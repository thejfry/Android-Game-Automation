"""Run Terraminer Gold automation against a connected Android device."""

import argparse
import logging
import time

from game_automation.core.device import MyPhone
from game_automation.core.events import StateBus
from game_automation.core.frame_stream import FrameStream

from .agent import TerraminerGoldAgent
from .controller import ControllerState, TerraminerGoldController
from .perception import TerraminerGoldPerception


def _coordinate(values):
    if values is None:
        return None
    return tuple(values)


def main():
    parser = argparse.ArgumentParser(description="Solve the current Terraminer Gold board.")
    parser.add_argument("--grid-size", type=int, help="Set grid dimensions when auto-detection is incorrect.")
    parser.add_argument("--start", nargs=2, type=int, metavar=("ROW", "COLUMN"))
    parser.add_argument("--goal", nargs=2, type=int, metavar=("ROW", "COLUMN"))
    parser.add_argument("--tap-interval", type=float, default=0.04)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Detect the board and report a route without sending device taps.",
    )
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Wait for manual confirmation and solve successive levels.",
    )
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    phone = MyPhone()
    phone.connect_device()
    bus = StateBus()
    stream = FrameStream(max_queue=1, device=phone.device)
    perception = TerraminerGoldPerception(
        bus,
        grid_size=args.grid_size,
        start=_coordinate(args.start),
        goal=_coordinate(args.goal),
    )
    agent = TerraminerGoldAgent(
        phone,
        bus,
        tap_interval=args.tap_interval,
        dry_run=args.dry_run,
    )
    controller = TerraminerGoldController(
        agent,
        perception,
        continuous=args.continuous,
    )

    try:
        stream.start()
        while not controller.finished:
            if controller.state is ControllerState.AWAITING_NEXT_LEVEL:
                input(
                    "After any ad, open the next puzzle and press Enter to detect it "
                    "(Ctrl+C to stop): "
                )
                controller.acknowledge_next_level()
                continue
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
