"""Runnable launcher for AtlasGolfAgent.

Usage:
  python -m game_automation.games.atlas_golf.run_agent
  python -m game_automation.games.atlas_golf.run_agent --bbox 100 300 300 500
"""

import argparse
import time

from game_automation.core.device import MyPhone
from game_automation.core.events import StateBus
from game_automation.games.atlas_golf.agent import AtlasGolfAgent


def main():
    parser = argparse.ArgumentParser(description="Run AtlasGolfAgent")
    parser.add_argument(
        "--bbox",
        nargs=4,
        type=int,
        metavar=("X1", "Y1", "X2", "Y2"),
        default=[620, 2200, 660, 2230],
        help="Target bounding box for taps (x1 y1 x2 y2)",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Optional path to write timings CSV on stop",
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Enable automatic-click mode: single Enter starts session and agent performs 10 automatic clicks",
    )
    parser.add_argument(
        "--offsets",
        nargs=10,
        type=float,
        metavar=("O1", "O2", "O3", "O4", "O5", "O6", "O7", "O8", "O9", "O10"),
        default=[3.700, 7.000, 10.030, 12.880, 15.560, 18.070, 20.480, 22.775, 24.960, 27.060],
        help="Ten elapsed times (seconds after session start) for automatic clicks",
    )

    args = parser.parse_args()
    bbox = tuple(args.bbox)

    phone = MyPhone()
    phone.connect_device()

    bus = StateBus()

    agent = AtlasGolfAgent(
        phone,
        bus,
        bbox,
        csv_path=args.csv,
        auto=args.auto,
        click_offsets=(list(args.offsets) if args.offsets is not None else None),
    )
    agent.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping AtlasGolfAgent...")
    finally:
        agent.stop()


if __name__ == "__main__":
    main()
