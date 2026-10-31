Atlas Bowling game (no perception)
=================================

This game module provides `AtlasBowlingAgent`, an agent that does not use
computer-vision perception. Instead it listens for Enter presses on stdin:

- First Enter: starts the timer.
- Each subsequent Enter: the agent taps the center of the configured bounding
  box on the connected device and prints the elapsed time since the first
  Enter and since the previous Enter.

Usage example (run from project root):

```python
from game_automation.core.device import MyPhone
from game_automation.core.events import StateBus
from game_automation.games.atlas_bowling.agent import AtlasBowlingAgent

phone = MyPhone()
phone.connect_device()

bus = StateBus()

# Example bounding box (x1, y1, x2, y2) - adjust to your game's target area
bbox = (100, 300, 300, 500)
Automatic mode accepts `--offsets` as five elapsed times (seconds after the
session start). For example:

```bash
python -m game_automation.games.atlas_bowling.run_agent --auto --offsets 0.39 6.64 12.66 18.57 24.29
```

This will make the agent click at 0.39s, 6.64s, 12.66s, 18.57s and 24.29s after
you press Enter to start the session.

agent = AtlasBowlingAgent(phone, bus, bbox)
agent.start()

# Now press Enter in this console to start and trigger taps.
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    agent.stop()
```

Note: This agent uses `input()` to capture Enter presses. If you run the
framework's `main.py` which also drives frame-based perception, you may prefer
to run `AtlasBowlingAgent` separately to avoid input contention.
