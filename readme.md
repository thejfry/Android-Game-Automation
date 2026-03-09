structure of this mobile game automation folder:
`
"""
mobile_automation/
│
├── framework/
│   ├── __init__.py
│   ├── device.py            # ADB wrapper (your MyPhone class, improved)
│   ├── vision.py            # Screenshot capture, pixel sampling, region analysis
│   ├── actions.py           # High-level actions (tap sequences, waits, loops)
│   ├── utils.py             # Helpers: timing, logging, config loading
│   └── config.py            # Central config loader (YAML/JSON)
│
├── games/
│   ├── __init__.py
│   ├── game1/
│   │   ├── config.yaml      # Coordinates, thresholds, colors
│   │   ├── bot.py           # Game-specific logic
│   │   └── regions.png      # Optional: annotated screenshot for reference
│   ├── game2/
│   │   ├── config.yaml
│   │   └── bot.py
│   └── templates/           # Optional: starter configs for new games
│
├── scripts/
│   ├── run_game1.py         # CLI entry point for running a bot
│   ├── calibrate.py         # Tool to help find coordinates/colors
│   └── debug_overlay.py     # Visual debugging tool
│
├── tests/
│   ├── test_device.py
│   ├── test_vision.py
│   └── test_actions.py
│
├── assets/
│   ├── screenshots/         # Saved screenshots for debugging
│   └── icons/               # Optional UI assets for overlays
│
├── requirements.txt
├── README.md
└── pyproject.toml (optional)
""""
`