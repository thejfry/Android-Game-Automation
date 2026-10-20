# Terraminer Rock

This package automates the rock-mine **Memory Match** minigame. Start the game
on the connected Android device, then run this command from the repository root:

```powershell
python -m game_automation.games.terraminer_rock.run_agent
```

The agent detects the card grid from the screenshot, so later levels can have
different row and column counts. It does not assume the initial 3×4 layout or a
fixed number of pairs. To inspect detection and the next planned card without
tapping:

```powershell
python -m game_automation.games.terraminer_rock.run_agent --dry-run --log-level DEBUG
```

By default, it stops after all cards on the current board are matched. Use
`--continuous` to keep running through levels: after each board is completed,
open the next rock level and press Enter in the terminal. The next board must
be detected face-down in three consecutive frames before the agent forgets the
previous level's card locations and starts learning the new board. Press
Ctrl+C to stop. `--tap-interval` controls the minimum wait between taps
(default `0.5` seconds).

## How the solver works

1. **Find the grid.** OpenCV detects the repeated, near-square card outlines
   and groups their centers into rows and columns. Every row/column intersection
   must contain a card, and the card count must be even. Thus grid dimensions
   are inferred from the current screen instead of hard-coded.
2. **Read each reveal.** A card is face-up when its center is bright. For a
   face-up card, the inner part of the image is resized to 32×32 pixels; the
   border is excluded so the ordinary and green matched borders do not affect
   identity.
3. **Remember and compare.** The solver stores each revealed image by its grid
   position. To reduce the effect of the shared card background, it estimates
   that background from the crop's perimeter and extracts a foreground mask.
   Two cards are compared using the overlap of their foreground shapes (75% of
   the score) and the BGR color difference where both masks overlap (25%). A
   combined distance at or below 20 is considered a match. This uses the
   centered artwork rather than treating mostly identical light backgrounds as
   proof that two icons are the same.
4. **Choose a tap.** If one card is open, tap its known partner if one has been
   seen; otherwise reveal an unseen card. With no card open, finish a known
   pair first, then reveal the first still-unknown card. Already matched cards
   are never selected. If two non-matching cards are still visible, wait for the
   game to turn them face-down before choosing again. When they do, that
   position-pair is recorded as a known mismatch and will never be tested
   again, even if their images were initially classified as similar.
5. **Stop on completion.** The game keeps successful pairs face-up with a green
   border, so those cards are permanently removed from the solver's choices.
   When all detected cards have that state, the level is complete.

This is an online strategy: it does not guess hidden pictures. Each card is
first revealed to learn its identity; later flips exploit remembered pairs.
The solver minimizes unnecessary taps given only screenshot feedback, but
cannot guarantee winning within a game's move/time limit on every random
shuffle. Perception and the image-distance threshold are calibrated against the
provided screenshots; substantial changes to the game's art, animation, or
screen layout may require adjusting the detectors.

The game-level board completion is detected from the green matched cards.
Progression to another level is intentionally manual, including with
`--continuous`; the agent does not tap `End Game`, ads, or level-navigation
controls.
