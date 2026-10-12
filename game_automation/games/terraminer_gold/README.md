# Terraminer Gold

This agent detects the square board and its grid, identifies the miner and gold
pot, classifies the open edges of each road tile, and searches for a route with
the fewest clockwise taps (using route length as a tie-breaker). It then taps
the required tiles and checks the resulting route against a fresh frame.

Open a level on the connected Android device, then run from the project root:

```powershell
python -m game_automation.games.terraminer_gold.run_agent
```

To detect and solve a board without sending any taps, use dry-run mode:

```powershell
python -m game_automation.games.terraminer_gold.run_agent --dry-run --log-level DEBUG
```

To keep the controller running across manually opened levels:

```powershell
python -m game_automation.games.terraminer_gold.run_agent --continuous
```

If automatic board detection needs help, set the square grid size or endpoint
coordinates (row and column, zero-based):

```powershell
python -m game_automation.games.terraminer_gold.run_agent --grid-size 5 --start 0 0 --goal 2 4
```

Use `--log-level DEBUG` to print the detected tile masks, per-port confidence,
selected route, and planned rotations. Port detection requires a narrow
centerline of connected gold pixels, low gold coverage in the surrounding
terrain, and paired direction-aligned edges that persist through each sampled
segment. Color and edge evidence must agree in a sample; textured gold terrain
alone should not count as a road. Reported confidence averages evidence in
the strongest of the two four-sample windows nearest the tile edge, not a
probability. This tolerates a short section obscured by brick seams while
still requiring evidence at the edge and continuity toward the center. The
agent makes small per-tap variations around each tile center (up to 5% of the
cell dimensions) and varies waits between repeated taps by up to 20% of
`--tap-interval`. It verifies commanded tile orientations against a fresh
screenshot and reports route-geometry verification separately; it does not
currently detect a game-level victory indicator.

The runner logs controller state transitions: `WaitingForBoard`, `Planning`,
`Executing`, `Settling`, `Verifying`, `WaitingForCompletion`, `RewardScreen`,
`AwaitingNextLevel`, and `Failed` (dry-run ends in `Stopped`). After sending a
route, it will not re-plan or send more taps while the game animates. It checks
for the reward screen before trying board perception; the detector recognizes
either the bottom action button or the wider `LEVEL UP` / `Watch Ad` panel
that appears between some levels. The latter requires a yellow panel with a
nested orange button, rather than a broad yellow region alone. DEBUG logs
include the matched button/panel bounding boxes. Morphological closing bridges
small text/outline gaps, and the signal must be present in two consecutive frames.
On confirmation, normal mode enters `RewardScreen` and exits without pressing
`Play Again` or `Exit`. In continuous mode, a confirmed bottom `Play Again`
button is tapped at a randomized point inside its detected bounds; the
controller waits briefly, then resets and resumes board detection. It ignores
the still-visible button after tapping so it cannot repeatedly activate it.
A `LEVEL UP` / `Watch Ad` panel is never tapped and instead transitions to
`AwaitingNextLevel`.
If route geometry cannot be confirmed during the transition, it logs that
result and waits for the game's completion screen instead of retrying against
transient frames.

To keep the process running across levels, add `--continuous`. Levels with a
bottom `Play Again` button advance automatically. If an inter-level ad panel
appears, finish the ad, open the next puzzle if needed, and press Enter in the
terminal. The controller then requires three consecutive matching board
detections before planning, preventing it from treating the previous reward
screen or an animation frame as a new level. It does not click the ad panel or
automate ad playback.

The agent captures device screenshots over ADB and uses ADB for control, so an
authorized connected device is required. It waits for a recognizable board to
appear before solving. It is designed for the currently observed straight and
elbow tiles and also handles connected T-junction masks.
