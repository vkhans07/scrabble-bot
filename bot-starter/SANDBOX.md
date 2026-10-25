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
