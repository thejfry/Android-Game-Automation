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
`--continuous` to keep running through levels. On the standard results page it
automatically taps the detected **Play Again** button. If the level-up/ad prompt
appears, it does not tap **Play Level**. Instead, it checks the screen every
five seconds for a new face-down board. Once a possible board is detected, it
checks again at a faster cadence (20% of the poll interval, capped at one
second) and resumes after three matching board checks. If a check fails, it
returns to the slower polling cadence. The controller stops if no board appears
within four minutes.
Use `--next-level-poll-seconds` and `--next-level-timeout-seconds` to adjust
those defaults. Press Ctrl+C to stop. `--tap-interval` controls the minimum
wait between card taps (default `0.5` seconds).
At INFO level, the controller reads the Moves, Time, and Matches values from
their separate HUD columns. It prints the progress bar, Moves, and Time
together on one line when the match count, move count, or ten-second time
bucket changes.
The reader handles both screen layouts—with and without the Android status
bar—by checking the two possible value-row locations. It only accepts counters
that parse. The detected complete card grid is authoritative for the expected
pair total, so a misread HUD denominator is corrected from the board while the
current match count is still validated. Controller-level checks also reject
HUD matches beyond the pairs visibly confirmed on the board, move counts beyond
the card taps sent, and counters that move backward or make the timer increase.
This guards against plausible
but incorrect OCR results (for example, misreading `1/8` as `7/8`). Match
progress remains monotonic within each level.
The HUD parser also handles two ambiguities seen in debug output: a leading
move-count `1` misread as `7` when that reading exceeds the move limit, and a
timer suffix `s` misread as `5`.
The completion summary uses a recent, consistent HUD reading when available;
otherwise it estimates time from elapsed time and moves from card taps. If the
results screen appears before the completed board can be observed, that screen
is used to close out the level. Set
`--level-time-seconds` if the game's starting timer differs from the 90-second
fallback. Rolling time and move averages are normalized per matched pair,
making them more comparable as later boards grow.
Individual tap coordinates and card positions are only logged at DEBUG level.
Completed-level summaries include a readable table of the latest five levels.

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
   When only two unmatched cards remain, the game rule that all cards form
   pairs takes precedence over the visual similarity score: the solver always
   selects the other remaining card, even if its image signature looks
   different.
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

The game-level board completion is detected from the green matched cards. A
dedicated lifecycle controller then waits for a stable results control before
stopping, replaying, or awaiting a manually opened board. The agent does not
tap `End Game`, ad prompts, or the level-up **Play Level** button.
