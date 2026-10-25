// Belief tracking checked against a mini-simulator that knows every hidden
// tile. Run with `npm test`.

import assert from "node:assert/strict";
import { test } from "node:test";
import { A, BAG, Dict, countsOf, rewardOf } from "./dict.js";
import { makeRng } from "./rng.js";
import { computeBelief, type AuctionRow, type PlayRow, type ResultRow } from "./state.js";

const dict = Dict.load();
const ts = (micros: number) => ({ microsSinceUnixEpoch: BigInt(micros) });

// The server's findBestPlayable: longest word >= 3 that fits, ties alphabetical.
function bestPlayable(rack: Uint8Array): string | undefined {
  let best: string | undefined;
  for (const w of dict.allWords) {
    if (w.length < 3) continue;
    if (best !== undefined && (w.length < best.length || (w.length === best.length && w > best)))
      continue;
    const c = countsOf(w);
    let ok = true;
    for (let i = 0; i < 26 && ok; i++) if (c[i] > rack[i]) ok = false;
    if (ok) best = w;
  }
  return best;
}

interface Truth {
  bag: number[]; // remaining letters, in no particular order
  racks: Map<bigint, Uint8Array>; // everyone's full rack (hidden + won)
  hiddenLeft: Map<bigint, Uint8Array>; // starting tiles not yet revealed
}

function runMatch(seed: number, nBots: number, rackSize: number) {
  const rng = makeRng(seed);
  const bag: number[] = [];
  BAG.forEach((n, i) => {
    for (let k = 0; k < n; k++) bag.push(i);
  });
  const draw = () => bag.splice(Math.floor(rng() * bag.length), 1)[0];
  const me = 1n;
  const bots = Array.from({ length: nBots }, (_, i) => BigInt(i + 1));
  const truth: Truth = { bag, racks: new Map(), hiddenLeft: new Map() };
  for (const b of bots) {
    truth.racks.set(b, new Uint8Array(26));
    truth.hiddenLeft.set(b, new Uint8Array(26));
  }
  for (const b of bots)
    for (let k = 0; k < rackSize; k++) {
      const L = draw();
      truth.racks.get(b)![L]++;
      truth.hiddenLeft.get(b)![L]++;
    }

  const auctions: AuctionRow[] = [];
  const results: ResultRow[] = [];
  const plays: PlayRow[] = [];
  let now = 1_000_000;
  let auctionId = 0n;
  let playId = 0n;
  const checks: { U: Uint8Array; expect: Uint8Array; mismatches: string[] }[] = [];

  const expected = () => {
    const e = new Uint8Array(26);
    for (const L of bag) e[L]++;
    for (const b of bots) if (b !== me) for (let i = 0; i < 26; i++) e[i] += truth.hiddenLeft.get(b)![i];
    return e;
  };

  const playWord = (b: bigint, word: string) => {
    const rack = truth.racks.get(b)!;
    const hid = truth.hiddenLeft.get(b)!;
    for (const ch of word) {
      const L = ch.charCodeAt(0) - A;
      rack[L]--;
    }
    // Tiles are interchangeable: whatever of the rack is no longer there
    // beyond the won tiles has come out of the hidden set. Recompute hidden
    // as min(hidden, rack): the rack always holds hidden tiles first.
    for (let i = 0; i < 26; i++) hid[i] = Math.min(hid[i], rack[i]);
    plays.push({
      id: ++playId,
      matchId: 1n,
      botId: b,
      word,
      totalReward: BigInt(rewardOf(word)),
      playedAt: ts(now),
    });
  };

  while (bag.length > 0) {
    const L = draw();
    auctionId++;
    now += 1_000_000;
    auctions.push({
      id: auctionId,
      matchId: 1n,
      letter: String.fromCharCode(A + L),
      opensAt: ts(now),
      closesAt: ts(now + 1_000_000),
      status: { tag: "Open" },
    });

    // The bot sees the state right after the auction opens (bag already drawn).
    const myRack = truth.racks.get(me)!;
    const belief = computeBelief({
      myBotId: me,
      rackSize,
      bagTotal: bag.length,
      botIds: bots,
      auctions,
      results,
      plays,
      myRack,
    });
    // The open letter has left the bag but is in nobody's rack yet, and U
    // excludes it, so it matches expected() computed the same way.
    checks.push({ U: belief.U, expect: expected(), mismatches: belief.mismatches });

    // Close: a random bot wins; if it is a sim-like bot (not me) it plays its
    // longest word in the same transaction (same timestamp) as the result.
    now += 1_000_000;
    const winner = bots[Math.floor(rng() * bots.length)];
    truth.racks.get(winner)![L]++;
    results.push({
      auctionId,
      matchId: 1n,
      letter: String.fromCharCode(A + L),
      winnerBotId: winner,
      paid: 1n,
      closedAt: ts(now),
    });
    auctions[auctions.length - 1] = { ...auctions[auctions.length - 1], status: { tag: "Closed" } };
    if (winner !== me) {
      const w = bestPlayable(truth.racks.get(winner)!);
      if (w) playWord(winner, w);
    } else if (rng() < 0.2) {
      const w = bestPlayable(myRack);
      if (w) {
        now += 10;
        playWord(me, w);
      }
    }
  }
  return checks;
}

test("belief U equals the true unseen pool at every auction", () => {
  for (const seed of [1, 2, 3, 4, 5]) {
    const checks = runMatch(seed, 6, 5);
    assert.ok(checks.length >= 60, `got ${checks.length} auctions`);
    checks.forEach((c, i) => {
      assert.deepEqual(c.mismatches, [], `seed ${seed} auction ${i}`);
      assert.deepEqual(Array.from(c.U), Array.from(c.expect), `seed ${seed} auction ${i}`);
    });
  }
});

test("belief survives a different roster size and rack size", () => {
  for (const [n, r] of [[2, 7], [4, 3]] as const) {
    const checks = runMatch(9, n, r);
    for (const c of checks) {
      assert.deepEqual(c.mismatches, []);
      assert.deepEqual(Array.from(c.U), Array.from(c.expect));
    }
  }
});

test("a play that lands before its own win in event order is still attributed correctly", () => {
  // Same timestamp for result and play: results sort first.
  const me = 1n;
  const opp = 2n;
  const auctions: AuctionRow[] = [
    { id: 1n, matchId: 1n, letter: "Q", opensAt: ts(1), closesAt: ts(2), status: { tag: "Closed" } },
  ];
  const results: ResultRow[] = [
    { auctionId: 1n, matchId: 1n, letter: "Q", winnerBotId: opp, paid: 1n, closedAt: ts(5) },
  ];
  const plays: PlayRow[] = [
    { id: 1n, matchId: 1n, botId: opp, word: "QI", totalReward: 11n, playedAt: ts(5) },
  ];
  const b = computeBelief({
    myBotId: me,
    rackSize: 1,
    bagTotal: 98 - 2 - 1, // 2 racks of 1 + the auctioned Q
    botIds: [me, opp],
    auctions,
    results,
    plays,
    myRack: countsOf("E"),
  });
  // Q came from the auction; I came from the hidden tile, so hidden is spent.
  assert.equal(b.hidden.get(opp), 0);
  assert.equal(b.U["I".charCodeAt(0) - A], BAG["I".charCodeAt(0) - A] - 1);
  assert.equal(b.U["Q".charCodeAt(0) - A], 0);
  assert.deepEqual(b.mismatches, []);
});

test("an inconsistent table is reported, not thrown", () => {
  const b = computeBelief({
    myBotId: 1n,
    rackSize: 1,
    bagTotal: 10,
    botIds: [1n, 2n],
    auctions: [],
    results: [],
    plays: [],
    myRack: countsOf("E"),
  });
  assert.ok(b.mismatches.length > 0);
});
