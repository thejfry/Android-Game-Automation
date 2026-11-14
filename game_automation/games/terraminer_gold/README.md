# Terraminer Gold

This agent detects the square board and its grid, identifies the miner and gold
pot, classifies the open edges of each road tile, and searches for a route with
the fewest clockwise taps (using route length as a tie-breaker). It then taps
the required tiles at the same cadence as the Terraminer Rock controller and
checks the resulting route against a fresh frame.

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
cell dimensions) and schedules no more than one tap per controller frame,
waiting a randomized interval around `--tap-interval` (default `0.3` seconds)
between taps. Grid size and both endpoints are detected from the screen. It
verifies commanded tile orientations against a fresh screenshot and reports
route-geometry verification separately; it does not currently detect a
game-level victory indicator.

The runner logs controller state transitions: `WaitingForBoard`, `Planning`,
`Executing`, `Settling`, `Verifying`, `WaitingForCompletion`, `RewardScreen`,
`AdvancingLevel`, `AwaitingNextLevel`, and `Failed` (dry-run ends in `Stopped`).
After sending a route, it will not re-plan while the game animates. Reward UI
recognition lives in a dedicated detector and returns a typed match: either
Gold's lower-screen action button or the wider `LEVEL UP` / `Watch Ad` panel
that appears between some levels. The latter requires a yellow panel
with a nested orange button, rather than a broad yellow region alone. DEBUG
logs include the matched button/panel bounding boxes. Morphological closing
bridges small text/outline gaps, and the signal must be present in two
consecutive frames. On confirmation, normal mode enters `RewardScreen` and
exits without pressing `Play Again` or `Exit`. In continuous mode, a confirmed
bottom `Play Again` button is tapped at a randomized point inside its detected
bounds; its location is intentionally detected in Gold's lower-screen reward
area rather than using Rock's differently positioned button. `AdvancingLevel`
waits asynchronously before board detection resumes.
The controller ignores the still-visible reward UI after tapping, preventing
repeated activation. A `LEVEL UP` / `Watch Ad` panel is never tapped and
instead transitions to `AwaitingNextLevel`.
If route geometry cannot be confirmed during the transition, it logs that
result and waits for the game's completion screen instead of retrying against
transient frames.

To keep the process running across levels, add `--continuous`. Levels with a
lower-screen `Play Again` button advance automatically. If an inter-level ad
panel appears, finish the ad and open the next puzzle if needed; the controller
then checks for a new board automatically, requiring three consecutive
matching detections before planning. It checks every five seconds when no
candidate board is visible, polls faster while confirming a candidate, and
stops after four minutes if no board appears. Use
`--next-level-poll-seconds` and `--next-level-timeout-seconds` to adjust those
defaults. It does not click the ad panel or automate ad playback.

Perception is a small façade over separate board/grid, endpoint, and tile-port
detectors, and returns an immutable, typed board observation directly to the
controller. The solving agent returns an explicit solution/execution result
and exposes route verification without requiring the controller to inspect
agent internals. A shared tap executor owns frame-to-device coordinate
conversion, in-bounds jitter, logging, and device taps for both tile rotations
and reward buttons.

The runner captures device screenshots over ADB and uses ADB for control, so an
authorized connected device is required. It waits for a recognizable board to
appear before solving. It is designed for the currently observed straight and
elbow tiles and also handles connected T-junction masks.
