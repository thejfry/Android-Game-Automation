structure of this mobile game automation folder:
`
"""
game_automation/
├─ README.md
├─ pyproject.toml / requirements.txt
├─ game_automation/
│  ├─ __init__.py
│  ├─ config.py
│  ├─ main.py
│  ├─ core/
│  │  ├─ phone.py
│  │  ├─ frame_stream.py
│  │  ├─ perception.py
│  │  ├─ hud.py
│  │  ├─ agent.py
│  │  ├─ events.py
│  │  └─ utils.py
│  ├─ games/
│  │  ├─ __init__.py
│  │  ├─ example_game/
│  │  │  ├─ __init__.py
│  │  │  ├─ config.py
│  │  │  ├─ perception.py
│  │  │  └─ agent.py
│  └─ scripts/
│     ├─ record_session.py
│     └─ replay_session.py
└─ tests/
   ├─ test_phone.py
   ├─ test_perception.py
   └─ test_agent.py
""""
`

## Shared game automation

Game agents send taps through `game_automation.core.tap_executor.TapExecutor`.
It provides bounded position jitter and separately configurable timing jitter;
game-specific agents choose safe target bounds and can disable timing jitter
when precise schedules are required. Atlas Golf and Atlas Bowling jitter taps
inside their configured target box while preserving their scheduled tap times.

Terraminer game controllers inherit shared board-confirmation, reward-screen,
Play Again, level reset, and progress reporting behavior from
`game_automation.games.terraminer_common.TerraminerController`. Individual
Terraminer packages retain their own perception, solver, and game-specific
completion handling. Controllers can receive a callback with the signature
`level_progress_callback(level, status)`; statuses report `started` and
`completed`.