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