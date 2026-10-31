"""Runnable launcher for AtlasBowlingAgent.

Usage:
  python -m game_automation.games.atlas_bowling.run_agent
  python -m game_automation.games.atlas_bowling.run_agent --bbox 100 300 300 500
"""

import argparse
import time

from game_automation.core.device import MyPhone
from game_automation.core.events import StateBus
from game_automation.games.atlas_bowling.agent import AtlasBowlingAgent


def main():
    parser = argparse.ArgumentParser(description="Run AtlasBowlingAgent")
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
        help="Enable automatic-click mode: single Enter starts session and agent performs 5 automatic clicks",
    )
    parser.add_argument(
        "--auto-interval",
        type=float,
        default=0.2,
        help="Interval in seconds between automatic clicks when --auto is set",
    )
    parser.add_argument(
        "--offsets",
        nargs=4,
        type=float,
        metavar=("O1", "O2", "O3", "O4"),
        default=[6.250, 12.270, 18.180, 24.000],
        help="Four elapsed times (seconds after session start) for automatic clicks",
    )

    args = parser.parse_args()
    bbox = tuple(args.bbox)

    phone = MyPhone()
    phone.connect_device()

    bus = StateBus()

    agent = AtlasBowlingAgent(
        phone,
        bus,
        bbox,
        csv_path=args.csv,
        auto=args.auto,
        auto_interval=args.auto_interval,
        click_offsets=(list(args.offsets) if args.offsets is not None else None),
    )
    agent.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping AtlasBowlingAgent...")
    finally:
        agent.stop()


if __name__ == "__main__":
    main()
