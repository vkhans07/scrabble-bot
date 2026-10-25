# Tournament bot evaluation

The live strategy is `index.ts` → `policy.ts` / `engine.worker.ts` → native
`engine.cpp` (or JS `rollout.ts`). `strategy.ts` is the starter, not the live policy.
No board is involved: letters are auctioned and words earn score and currency.

From `bot-starter/`:

```sh
npm test
npx tsc --noEmit
npm run build:native
BOT_NAME=evaluation EVAL_MATCHES=3 EVAL_MAX_MS=600000 LOG_DIR=logs/my-run SEED=241024 node --import tsx src/index.ts
npm run evaluate:logs -- logs/my-run
```

Use the existing token file; do not put tokens into commands, reports, or logs.
Check existing processes before launch. An identity lock in this checkout rejects
a second authenticated instance, but does not detect bots on other hosts or in
other checkouts. `EVAL_MATCHES` stops new lobby joins after that many completed
matches and exits after the final archive collection (8 seconds). `EVAL_MAX_MS`
is a hard stop and can leave the last match incomplete. The hard stop is useful
when the endpoint cannot connect. Tournament runs require the intended
`TOURNAMENT_ID`; they never join the ordinary lobby.

Compare runs with the same configuration and record opponent composition and
auction type. A few lobby matches are a smoke test and a first baseline, not an
estimate of tournament strength. Preserve partial runs and report disconnects.
The seed controls bot simulations, not server draws or opponents.

Logs include final score, tied placement (1 + number with strictly higher score),
remaining balance, rack, playable rack value and whether that value is exact;
submitted/acknowledged/failed actions; locally skipped late bids; bid/word
acknowledgment round trips; engine timing and completed/requested evaluations.
`decisionMs` measures local work only. `roundTripMs` measures reducer completion,
not authoritative server execution latency. Initial match snapshots let analysts
identify matches joined partway through. Full results/request counts also include
normal snapshot cancellation; a low ratio is not by itself an engine failure.

Word submissions reserve each occurrence separately until a public WordPlay
confirms consumption. Server SenderError releases its reservation. Transport or
unknown errors quarantine further word submissions for that match. Pending match
IDs are journaled before sending; a restart conservatively quarantines those
matches. No uncertain word request is automatically replayed. This favors safety
over scoring after a disconnect. A new match starts with a fresh ledger.

The native cash-in regression uses a deterministic scenario: cashing a seven-tile
word funds a next-auction purchase, while holding with zero cash cannot. The
baseline no longer automatically cashes before the first bid. Both arms still use
the existing approximate continuation policy after purchases. Its future policy
is not an exact model of the gain-gated live client.

Remaining modeling/engineering limits to measure before strategic changes:

- Real opponents are modeled by configured face-value bids, not learned behavior.
  Future opponent balances remain constant in the simulation.
- Subscriptions use a fixed lower match ID bound and accumulate history during
  long runs. These bounded evaluation runs constrain that growth; long-lived
  tournament service should rotate or narrow subscriptions.
- Main-thread endgame planning is operation-limited, not a hard wall-clock bound.
- Acknowledged submissions and public plays are separate observations; quarantine
  may forgo safe plays after an ambiguous disconnect.
- Unit and native regressions establish specific correctness properties, not
  competitive strength or complete SDK/server integration correctness.

## Stricter bid deadlines

The timing revision removes the receive-time-based clock correction. That
correction incorrectly counted message delivery delay as extra auction time.
On first observation, a server UTC deadline is converted to a fixed monotonic
local deadline, capped at the auction's stated duration. Negative receive-minus-open
observations can only shorten this deadline, accounting conservatively for a local
clock that trails the server. Positive notification delays never extend it.
Notification timestamps cannot independently establish exact clock skew. Later wall-clock adjustments cannot extend that auction.

The live progressive revision submits changed bids while the estimated auction
is open. The earlier 300–750 ms sending cutoff was removed: latency risk alone
should not discard a useful initial bid or improvement. The rolling RTT-based
`deliveryBudgetMs` remains in logs as a delivery-risk estimate, not permission
to send. Confirmed closed/expired auctions and unchanged bids are skipped.

The native engine now publishes three checkpoints:

1. Cached bid immediately from the main thread, then a fast native estimate
   (up to `NATIVE_FAST=300` scenarios).
2. Focused refinement of the current letter and coin value to a cumulative
   `NATIVE_REFINE=3000` scenarios, with up to `NATIVE_REFINE_MS=75` additional ms.
   The additional scenarios continue after the fast batch under the same frozen
   policy. Weighted sample means are merged, then damped once from the same
   prior; the fast samples are neither repeated nor counted twice.
3. Broad evaluation of other letters and cash-in candidates (up to 2,000
   scenarios under the existing budget). This preserves the focused current
   letter and coin estimate. Cash-in execution still waits for a full result.

Scenario counts are targets, not guarantees: time budgets and newer snapshots
can interrupt work. Refinement `ms` includes both focused batches. The JS fallback
keeps its existing fast/full pipeline. More samples reduce Monte Carlo noise;
they do not correct inaccurate opponent or continuation models.

This changes scheduling and sampling, not the bidding formula or endgame rules.
Compare accepted initial bids and auction participation as well as rejection
counts: fewer rejections alone can simply mean fewer attempted bids.


## Price-aware letter bids

For native second-price auctions, the focused current-letter simulation includes
paid acquisition arms. Each arm adds the current letter and subtracts a candidate
runner-up payment from the starting balance before simulating the rest of the
match. Candidate payments are half the modeled average top opposing bid, that
average, and the full available balance. The free-letter arm plus these paid
arms form a coarse expected-score curve; linear interpolation at its zero
crossing supplies the fair bid. Fast and refined checkpoints each publish a
bid estimate, and the refined estimate uses the cumulative scenario batch.

This handles the audited free-tile versus paid-tile budget mismatch directly.
The payment grid and interpolation are still approximations; opponent balances,
price distributions, continuation policy, and scenario sample noise can make the
curve imperfect. First-price auctions continue using their separate bid solver.

## Comparing paid-price searches

Choose `NATIVE_BID_SEARCH=adaptive` to test the coarse curve plus five nearby
prices around its estimated cutoff, or `NATIVE_BID_SEARCH=exhaustive` to simulate
every integer payment from 1 through the current balance. The default `coarse`
mode keeps the three-point interpolation. These modes affect native Vickrey bids;
first-price matches continue using their existing solver. Record the mode from
the bot-level `connected` event when comparing logs.
