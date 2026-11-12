"""Runnable launcher for AtlasRacerAgent.

Usage:
  python -m game_automation.games.atlas_racer.run_agent
"""

import argparse
import time

from game_automation.core.device import MyPhone
from game_automation.core.events import StateBus
from game_automation.games.atlas_racer.agent import AtlasRacerAgent


def main():
    parser = argparse.ArgumentParser(description="Run AtlasRacerAgent")
    parser.add_argument(
        "--duration",
        type=int,
        default=300,
        help="Swipe duration in milliseconds",
    )
    args = parser.parse_args()

    phone = MyPhone()
    phone.connect_device()
    agent = AtlasRacerAgent(phone, StateBus(), duration_ms=args.duration)
    agent.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping AtlasRacerAgent...")
    finally:
        agent.stop()


if __name__ == "__main__":
    main()
