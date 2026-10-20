"""Run Terraminer Rock automation against a connected Android device."""

import argparse
import logging
import time

from game_automation.core.device import MyPhone
from game_automation.core.frame_stream import FrameStream
from game_automation.games.terraminer_gold.tap_executor import TapExecutor

from .agent import TerraminerRockAgent
from .perception import BoardNotDetectedError, TerraminerRockPerception


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
    waiting_for_next_level = False
    candidate_signature = None
    candidate_frames = 0
    try:
        stream.start()
        while True:
            frame = stream.get_latest_frame()
            if frame is None:
                time.sleep(0.01)
                continue
            try:
                observation = perception.process(frame)
            except BoardNotDetectedError as error:
                candidate_signature = None
                candidate_frames = 0
                if not waiting_for_next_level:
                    logging.getLogger(__name__).info(
                        "Waiting for a recognizable memory board: %s", error
                    )
                continue

            if waiting_for_next_level:
                if any(card.face_up for card in observation.board.cards):
                    candidate_signature = None
                    candidate_frames = 0
                    continue
                signature = observation.board.signature
                if signature != candidate_signature:
                    candidate_signature = signature
                    candidate_frames = 1
                else:
                    candidate_frames += 1
                if candidate_frames < 3:
                    continue
                waiting_for_next_level = False
                agent.reset_for_next_level()

            if not agent.process(observation):
                if args.dry_run:
                    break
                continue
            if not args.continuous:
                break

            input(
                "Board complete. Open the next rock level, then press Enter "
                "(Ctrl+C to stop): "
            )
            waiting_for_next_level = True
            candidate_signature = None
            candidate_frames = 0
    except KeyboardInterrupt:
        logging.info("Terraminer Rock automation interrupted.")
    finally:
        stream.stop()


if __name__ == "__main__":
    main()
