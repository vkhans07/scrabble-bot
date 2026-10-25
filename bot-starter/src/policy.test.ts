// Run with `npm test`.

import assert from "node:assert/strict";
import { test } from "node:test";
import { LETTER_VALUE, letterIndex } from "./dict.js";
import { applySafeCap, bidAmount, cashinGain, composeBid, inAllIn, lambdaFor, pacedLambda, shouldDumpPlan } from "./policy.js";
import { makeRng } from "./rng.js";
import { firstPriceBid, oppBid, payment, priceOf, wins, type Opp } from "./simmodel.js";

const flat = { lambdaMode: "flat_allin", allinK: 5, allinMinV: 1, safeMaxBid: null } as const;
const linear = { ...flat, lambdaMode: "linear" } as const;
const mc = { realOppBaseBonus: 2, realOppSlope: 1, realOppJitter: 3 };

// u values that hit each jitter in {-2..2}.
const jitterU = [0.0, 0.2, 0.4, 0.6, 0.8].map((x) => x + 0.01);

test("sim bids match game.ts decideBid for every letter, jitter and balance", () => {
  const vowels = "AEIOU";
  for (let L = 0; L < 26; L++) {
    const ch = String.fromCharCode(65 + L);
    const face = LETTER_VALUE[L];
    const bases = {
      Cheapskate: face - 1,
      ValueBidder: face,
      Aggressive: face + (vowels.includes(ch) ? 4 : 2),
    } as const;
    for (const [kind, base] of Object.entries(bases)) {
      for (const balance of [0, 1, 3, 100]) {
        jitterU.forEach((u, k) => {
          const jitter = k - 2;
          const opp: Opp = { botId: "x", balance, kind: kind as Opp["kind"] };
          const expect = Math.max(Math.min(Math.max(base + jitter, 1), balance), 0);
          assert.equal(oppBid(opp, L, u, mc), expect, `${kind} ${ch} bal=${balance} j=${jitter}`);
        });
      }
    }
  }
});

test("real opponents use the configured base and jitter", () => {
  const opp: Opp = { botId: "r", balance: 100, kind: "Real" };
  const E = letterIndex("E");
  const seen = new Set<number>();
  for (let i = 0; i < 700; i++) seen.add(oppBid(opp, E, i / 700, mc));
  // base 3, jitter -3..+3 gives 0..6, clamped up to the minimum bid of 1.
  assert.deepEqual([...seen].sort((a, b) => a - b), [1, 2, 3, 4, 5, 6]);
});

test("Vickrey payment: runner-up price, reserve of 1, ties at the tied price", () => {
  assert.equal(payment(10, { P: 4, tied: 1 }, false), 4);
  assert.equal(payment(10, { P: 0, tied: 0 }, false), 1);
  assert.equal(payment(7, { P: 7, tied: 2 }, false), 7);
  assert.equal(payment(10, { P: 4, tied: 1 }, true), 10);
});

test("ties are won with probability 1/(tied+1)", () => {
  const pr = { P: 5, tied: 2 };
  assert.equal(wins(6, pr, 0.99), true);
  assert.equal(wins(4, pr, 0), false);
  assert.equal(wins(5, pr, 0.3), true); // 0.3 < 1/3
  assert.equal(wins(5, pr, 0.4), false);
  assert.equal(wins(0, { P: 0, tied: 0 }, 0), false);
});

test("price is the max opposing bid with a tie count", () => {
  const opps: Opp[] = [
    { botId: "a", balance: 100, kind: "ValueBidder" },
    { botId: "b", balance: 100, kind: "ValueBidder" },
    { botId: "c", balance: 0, kind: "Aggressive" },
  ];
  const E = letterIndex("E");
  const pr = priceOf(opps, E, [0.41, 0.41, 0.9], mc); // jitter 0 -> bid 1 each; broke bot bids 0
  assert.deepEqual(pr, { P: 1, tied: 2 });
});

test("lambda: flat is 1; linear runs 1.0 -> 0.1 by usableLeft", () => {
  assert.equal(lambdaFor(50, flat), 1);
  assert.ok(Math.abs(lambdaFor(66, linear) - 1) < 1e-9);
  assert.ok(Math.abs(lambdaFor(0, linear) - 0.1) < 1e-9);
  assert.ok(lambdaFor(33, linear) < lambdaFor(60, linear));
});

test("bid = min(balance, floor(v/lambda)); skip if v <= 0", () => {
  assert.equal(bidAmount(7.9, 100, 30, flat).amount, 7);
  assert.equal(bidAmount(50, 20, 30, flat).amount, 20);
  assert.equal(bidAmount(0, 100, 30, flat).amount, 0);
  assert.equal(bidAmount(-3, 100, 30, flat).amount, 0);
  assert.equal(bidAmount(0.9, 100, 30, flat).amount, 0);
  assert.equal(bidAmount(5, 0, 30, flat).amount, 0);
  // linear: lambda = 0.1 + 0.9*(33/66) = 0.55
  assert.equal(bidAmount(5.5, 100, 33, linear).amount, Math.floor(5.5 / 0.55));
});

test("flat_allin bids the whole balance in the last ALLIN_K usable auctions", () => {
  assert.equal(inAllIn(5, flat), false);
  assert.equal(inAllIn(4, flat), true);
  assert.equal(inAllIn(0, flat), true);
  assert.equal(bidAmount(1, 37, 4, flat).amount, 37);
  assert.equal(bidAmount(0.5, 37, 4, flat).amount, 0); // below ALLIN_MIN_V
  assert.equal(bidAmount(3, 37, 5, flat).amount, 3); // window not open yet
  assert.equal(inAllIn(0, linear), false);
  // first-price: never the whole balance
  assert.equal(bidAmount(6, 37, 2, flat, (value) => Math.floor(value / 2)).amount, 3);
});

test("SAFE_MAX_BID caps a bid", () => {
  assert.equal(applySafeCap(40, { safeMaxBid: 15 }), 15);
  assert.equal(applySafeCap(9, { safeMaxBid: 15 }), 9);
  assert.equal(applySafeCap(40, { safeMaxBid: null }), 40);
});

test("cash-in: money term applies only while there is something to buy", () => {
  const cfg = { moneyWeight: 0.5, lambdaMode: "flat_allin", allinK: 5 } as const;
  assert.equal(cashinGain(10, 30, -10, cfg), 5); // 10 + 0.5*1*10 - 10
  assert.equal(cashinGain(10, 0, -10, cfg), 0);
  assert.ok(cashinGain(10, 30, -20, cfg) < 0);
});

test("endgame: dump the plan on the final auction and when <= ENDGAME_K usable", () => {
  assert.equal(shouldDumpPlan(0, true, { endgameK: 2 }), true);
  assert.equal(shouldDumpPlan(2, false, { endgameK: 2 }), true);
  assert.equal(shouldDumpPlan(3, false, { endgameK: 2 }), false);
});

test("first-price: bids below value, never above it, and higher against stronger rivals", () => {
  const E = letterIndex("E");
  const weak: Opp[] = [{ botId: "a", balance: 100, kind: "Cheapskate" }];
  const strong: Opp[] = [{ botId: "a", balance: 100, kind: "Aggressive" }, { botId: "b", balance: 100, kind: "Aggressive" }];
  const bw = firstPriceBid(weak, E, 8, 100, mc, makeRng(1), 2000);
  const bs = firstPriceBid(strong, E, 8, 100, mc, makeRng(1), 2000);
  assert.ok(bw >= 1 && bw < 8, `weak ${bw}`);
  assert.ok(bs >= bw && bs <= 8, `strong ${bs}`);
  assert.equal(firstPriceBid(weak, E, 0.5, 100, mc, makeRng(1)), 0);
});

test("paced lambda spends the budget on the best value-per-coin letters", () => {
  const v = new Array(26).fill(0);
  const U = new Array(26).fill(0);
  const prices = new Array(26).fill(10);
  // 4 equally common letters worth 8, 4, 2, 1 at a price of 10 each.
  [8, 4, 2, 1].forEach((x, i) => {
    v[i] = x;
    U[i] = 10;
  });
  // 40 auctions, so 10 of each letter would cost 400 in total. With 100
  // coins only the best quarter can be bought: lambda = 8/10.
  assert.ok(Math.abs(pacedLambda(v, U, 100, 40, prices) - 0.8) < 1e-9);
  // 200 coins reaches the second letter: lambda = 4/10.
  assert.ok(Math.abs(pacedLambda(v, U, 200, 40, prices) - 0.4) < 1e-9);
  // Plenty of coins: buy everything, so lambda is the worst letter's ratio.
  assert.ok(Math.abs(pacedLambda(v, U, 100000, 40, prices) - 0.1) < 1e-9);
  // A letter at the cutoff bids its clearing price.
  const lam = pacedLambda(v, U, 100, 40, prices);
  assert.equal(bidAmount(8, 100, 30, { ...flat, lambdaMode: "paced" }, undefined, lam).amount, 10);
});

const composeCfg = { safeMaxBid: null, allinK: 5, nativePaidBid: true, paidBidWarmup: 1, earlyLambdaAuctions: 17, earlyLambdaMin: 0.4, maxBidFraction: 1, maxBidFloor: 20 };
const dec = (amount: number) => ({ amount, reason: "v/lambda" });

test("composeBid uses the paid-tile bid after the warmup", () => {
  assert.equal(composeBid(dec(10), 3, 0.3, 14, 2, 60, 100, { ...composeCfg, earlyLambdaAuctions: 0 }).amount, 14);
});

test("composeBid caps the first auction's paid-tile bid at v/lambda", () => {
  const r = composeBid(dec(10), 3, 0.3, 27, 1, 66, 100, { ...composeCfg, earlyLambdaAuctions: 0 });
  assert.equal(r.amount, 10);
  assert.match(r.reason, /warmup/);
});

test("composeBid applies the early lambda floor as an upper bound only", () => {
  assert.equal(composeBid(dec(10), 3, 0.25, 14, 5, 60, 100, composeCfg).amount, 7); // floor(3 / 0.4)
  assert.equal(composeBid(dec(4), 3, 0.25, 4, 5, 60, 100, composeCfg).amount, 4);
  assert.equal(composeBid(dec(10), 3, 0.25, 14, 18, 60, 100, composeCfg).amount, 14); // after the window
});

test("composeBid leaves the all-in window alone", () => {
  assert.equal(composeBid(dec(50), 3, 0.25, null, 5, 3, 50, composeCfg).amount, 50);
});

test("composeBid with NATIVE_PAID_BID=0 ignores the fair bid", () => {
  assert.equal(composeBid(dec(9), 3, 0.3, 30, 5, 60, 100, { ...composeCfg, nativePaidBid: false, earlyLambdaAuctions: 0 }).amount, 9);
});

test("composeBid caps large bids at max(floor, fraction of balance) outside the all-in window", () => {
  const cfg = { ...composeCfg, earlyLambdaAuctions: 0, maxBidFraction: 0.5 };
  assert.equal(composeBid(dec(10), 9, 0.2, 43, 15, 50, 72, cfg).amount, 36);
  assert.equal(composeBid(dec(10), 9, 0.2, 18, 30, 30, 25, cfg).amount, 18); // under the floor of 20
  assert.equal(composeBid(dec(72), 9, 0.2, null, 64, 3, 72, cfg).amount, 72); // all-in window
});
