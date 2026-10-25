# Offline match sandbox

Run complete seeded matches locally without connecting to SpacetimeDB or using
the bot credential. The simulator follows the server's bag, starting balance,
initial rack, word scoring, and Vickrey or first-price payment rules.

```sh
npm run sandbox -- --games 20 --seed 42 \
  --seats live-lite,planner,rack,aggressive,value,cheap \
  --rollouts 6 --horizon 12 --bid-step 5 \
  --bid-scale 0.8 --coin-floor 0.35 --early-cap 35 --early-auctions 10 \
  --out logs/sandbox-20.json
```

Policies:

- `cheap`, `value`, `aggressive`: the server's simulated face-value bidders,
  with the same ±2 jitter. They play the longest word of at least three
  letters after winning a tile.
- `rack`: estimates each letter's marginal value with the exact rack evaluator
  and paces spending against expected prices.
- `live-lite`: a local approximation of the live bot's bidding policy. Its
  settings can be swept with `--bid-scale`, `--coin-floor`, `--early-cap`, and
  `--early-auctions` (defaults: 0.8, 0.35, 35, and 10). It does not run the
  native Monte Carlo engine.
- `planner`: a receding-horizon benchmark. For each auction it tests bid
  candidates against sampled future draws and full opponent balances, using
  `live-lite` for its future decisions. This is a semi-optimal benchmark, not
  a game-theoretic optimum; it uses a short horizon and approximate terminal
  rack/cash value.

Seats are randomized per game so one policy does not always receive the same
starting rack position. The seed controls tile draws, initial racks, and ties;
planner rollout randomness has a separate deterministic stream, so changing
planner settings does not also change the actual game's tiles. The JSON output contains the options, per-game standings,
and aggregates for score, placement, balance, tiles won, words, and playable
leftover rack value.

For a paired comparison, run the same seed range and seat list while changing
one policy parameter or version. Keep the game count and rollout settings the
same. Use at least 100 games before treating small differences as meaningful.

## Live-engine and field seats

- `native`: the live bot's decision loop driven through the real native
  engine (`npm run build:native` first). Each auction it posts a snapshot of
  what the live client would know (own start rack, auctioned letters, tiles
  opponents revealed in words, opponent balances), waits for the full result,
  plays a cash-in that beats holding by `CASH_MARGIN`, dumps its plan when
  `usableLeft <= ENDGAME_K`, and bids the paid-tile fair bid (or v/lambda with
  `--native-paid-bid 0`, the same switch as `NATIVE_PAID_BID=0` live). Time
  budgets are lifted so scenario counts, not machine load, bound each pass;
  live passes can be cut short by the 1-second auction. `--opp-bonus`,
  `--opp-slope` and `--opp-jitter` set the engine's opponent model, and
  `--native-threads` its threads. Other `NATIVE_*`/policy env vars apply.
- `profile:<bot>`: bids from that bot's fitted model in
  `data/opponent-profiles.json`: latent bid `a + b*face + c*vowel +
  d*progress + e*endgame + N(0, sigma)`, capped at its balance, fitted with
  censoring from public matches (winning bids are archived; a losing bid is at
  most the runner-up price). Words follow the strategic cash rule above.
- `field`: draws a `profile:<bot>` per seat per game, weighted by how many
  public matches each bot played, to stand in for a lobby or tournament table.

Profiles describe bidding only. Word choice, timing, and anything a bot does
in reaction to its rack are not observed this way, so field seats are an
approximation of real opponents, not replicas.

Paired comparisons with worker threads (each game seed is played by both arms
in the same worker):

```sh
nice -n 19 npm run sandbox:pair -- --jobs 4 --games 200 --seed 42 \
  --seats native,field,field,field,field,field --focus native \
  --base "--native-paid-bid 1" --cand "--native-paid-bid 0" \
  --out logs/claude-evals/e1.json
```

Run it under Node 26 (`/opt/homebrew/bin/node --import tsx src/sandbox-pair.ts`)
if `node` resolves to an older Node whose tsx loader cannot resolve `.js`
imports. Keep `nice` and modest `--jobs` while a live bot shares the machine.

Field calibration (2026-10-24, `logs/claude-evals/calib-*.json`): rack-blind
profiles reproduced live top bids but left runner-up prices 3–4 coins low and
starved the economy. `--field-rack 1` (default) adds the letter's exact
marginal value to the seat's own rack; with it, simulated runner-up prices are
within 1.3 coins of live for every face value and tiles/words match. Known
gaps: the simulated field ends with ~11 more coins and ~4 fewer points than
live (its word rule cashes late). `--field-shift`, `--field-sigma` and
`--field-corr` are kept for sensitivity checks.
