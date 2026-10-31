# Atlas Racer

`AtlasRacerAgent` lets you steer a connected Android device with the keyboard:

- Left arrow: swipe from right to left across the center of the screen.
- Right arrow: swipe from left to right across the center of the screen.

Run it from the project root on Windows with a connected ADB device:

```bash
python -m game_automation.games.atlas_racer.run_agent
```

Optionally set the swipe duration in milliseconds:

```bash
python -m game_automation.games.atlas_racer.run_agent --duration 300
```
