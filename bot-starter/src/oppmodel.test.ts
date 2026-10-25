// Opponent priors: the sampler matches the CDF the fitter uses, the in-match
// update moves the right way, and the snapshot / SNAP wiring is unchanged
// unless priors are given. Run with `npm test`.

import assert from "node:assert/strict";
import { test } from "node:test";
import { A } from "./dict.js";
import { oppFields } from "./native.js";
import { OnlineOffsets, bidCdf, parsePriors, sampleBid, type PriorsFile } from "./oppmodel.js";
import { makeRng } from "./rng.js";
import { oppBid, type Opp } from "./simmodel.js";
import { Store, buildSnapshot, derive, matchOffsets, type OppPriorCtx } from "./state.js";

const SPEC = { mu: ["one", "face", "lbal"], sig: ["one"], pi: ["one"] };
const FILE: PriorsFile = {
  version: 1,
  fitted: "2026-10-24T00:00:00Z",
  source: "test",
  spec: SPEC,
  population: { mu: [6, 8, 2], sig: [Math.log(3)], pi: [-3] },
  bots: { "7": { name: "seven", matches: 40, params: { mu: [12, 8, 2], sig: [Math.log(2)], pi: [-4] } } },
  online: { tau: 5, range: 24, step: 1 },
};

test("sampleBid draws from the distribution bidCdf describes", () => {
  const rng = makeRng(3);
  const d = { mu: 13.3, sd: 3.1, pi: 0.08 };
  const cap = 20;
  const n = 200000;
  const hist = new Float64Array(cap + 1);
  for (let i = 0; i < n; i++) hist[sampleBid(d, cap, rng())]++;
  let cum = 0;
  for (let k = 0; k <= cap; k++) {
    cum += hist[k] / n;
    assert.ok(Math.abs(cum - bidCdf(k, d, cap)) < 0.006, `k=${k}: ${cum} vs ${bidCdf(k, d, cap)}`);
  }
  assert.equal(sampleBid(d, 0, 0.9), 0, "no balance, no bid");
});

test("the in-match offset rises for a bot that keeps winning at high prices and falls for one that keeps losing cheaply", () => {
  const on = new OnlineOffsets({ tau: 5, range: 24, step: 1 });
  const d = { mu: 10, sd: 3, pi: 0.05 };
  for (let i = 0; i < 6; i++) {
    on.observe("hi", d, 100, true, 18);
    on.observe("lo", d, 100, false, 4);
  }
  assert.ok(on.offset("hi") > 3, `hi ${on.offset("hi")}`);
  assert.ok(on.offset("lo") < -2, `lo ${on.offset("lo")}`);
  assert.equal(on.offset("unseen"), 0);
  const off = new OnlineOffsets({ tau: 0, range: 24, step: 1 });
  off.observe("hi", d, 100, true, 18);
  assert.equal(off.offset("hi"), 0, "tau 0 disables the update");
});

test("parsePriors rejects a file whose params do not match its spec", () => {
  assert.equal(parsePriors(JSON.stringify(FILE)).known("7"), true);
  const bad = structuredClone(FILE);
  bad.bots["7"].params.mu.pop();
  assert.throws(() => parsePriors(JSON.stringify(bad)), /mu/);
  assert.throws(() => parsePriors(JSON.stringify({ ...FILE, spec: { ...SPEC, mu: ["nope", "face", "lbal"] } })), /unknown feature/);
});

test("oppBid for a Prior opponent uses its per-letter table, other kinds unchanged", () => {
  const cfg = { realOppBaseBonus: 8, realOppSlope: 2.5, realOppJitter: 4 };
  const mu = new Array(26).fill(5);
  mu[25] = 30;
  const o: Opp = { botId: "7", balance: 25, kind: "Prior", prior: { mu, sd: new Array(26).fill(0.05), pi: 0 } };
  assert.equal(oppBid(o, 0, 0.5, cfg), 5);
  assert.equal(oppBid(o, 25, 0.5, cfg), 25, "capped at balance");
  const real: Opp = { botId: "8", balance: 100, kind: "Real" };
  assert.equal(oppBid(real, 0, 0.5, cfg), Math.round(8 + 2.5)); // jitter 0 at u = 0.5
});

test("SNAP fields: kind 4 carries pi and 26 (mu, sd) pairs; old kinds keep their format", () => {
  const cf = { realOppBaseBonus: 8, realOppSlope: 2.5, realOppJitter: 4 };
  const o: Opp = { botId: "7", balance: 42.4, kind: "Prior", prior: { mu: new Array(26).fill(9.5), sd: new Array(26).fill(2), pi: 0.1 } };
  const f = oppFields(o, cf).split(" ");
  assert.equal(f.length, 3 + 52);
  assert.deepEqual(f.slice(0, 5), ["4", "42", "0.1000", "9.50", "2.000"]);
  assert.equal(oppFields({ botId: "8", balance: 50, kind: "Real" }, cf), "3 50 8 2.5 4");
  assert.equal(oppFields({ botId: "9", balance: 50, kind: "Cheapskate" }, cf), "0 50 8 2.5 4");
});

// ---- snapshot wiring --------------------------------------------------------

const ts = (micros: number) => ({ microsSinceUnixEpoch: BigInt(micros) });

function mkStore() {
  const store = new Store();
  const me = 112n;
  const matchId = 1n;
  store.matches.set(matchId, {
    id: matchId,
    status: { tag: "Running" },
    currentAuctionId: 3n,
    bagTotal: 60,
    auctionType: { tag: "Vickrey" },
    auctionDurationMs: 1000n,
    startingRackSize: 5,
  });
  const d = store.match(matchId);
  const bots: [bigint, boolean, string][] = [
    [me, false, "Real"],
    [7n, false, "Real"],
    [8n, false, "Real"],
    [1n, true, "Cheapskate"],
  ];
  bots.forEach(([b, sim, strat], i) => {
    store.bots.set(b, { id: b, name: `b${b}`, isSimulated: sim, strategy: { tag: strat } });
    d.parts.set(BigInt(i), { id: BigInt(i), matchId, botId: b, balance: 100n, score: 0n });
  });
  // Bot 7 won two auctions at a high clearing price.
  const letters = ["E", "S", "Q"];
  letters.forEach((l, i) => {
    d.auctions.set(BigInt(i + 1), {
      id: BigInt(i + 1),
      matchId,
      letter: l,
      opensAt: ts(i * 1000),
      closesAt: ts((i + 1) * 1000),
      status: { tag: i < 2 ? "Closed" : "Open" },
    });
    if (i < 2) d.results.set(BigInt(i + 1), { auctionId: BigInt(i + 1), matchId, letter: l, winnerBotId: 7n, paid: 25n, closedAt: ts((i + 1) * 1000) });
  });
  d.parts.set(1n, { id: 1n, matchId, botId: 7n, balance: 50n, score: 0n });
  return { store, me, matchId, d };
}

test("buildSnapshot without priors is unchanged; with priors real opponents get tables and sims keep their kind", () => {
  const { store, me, matchId } = mkStore();
  const der = derive(store, matchId, me)!;
  const plain = buildSnapshot(store, der, me, false, "k");
  assert.deepEqual(plain.opps.map((o) => o.kind), ["Real", "Real", "Cheapskate"]);
  assert.ok(plain.opps.every((o) => o.prior === undefined));

  const priors = parsePriors(JSON.stringify(FILE));
  const ctx: OppPriorCtx = { priors, online: FILE.online };
  const snap = buildSnapshot(store, der, me, false, "k", ctx);
  assert.deepEqual(snap.opps.map((o) => o.kind), ["Prior", "Prior", "Cheapskate"]);
  const [o7, o8] = snap.opps;
  assert.equal(o7.botId, "7");
  // Bot 7 has its own prior (intercept 12 vs the population's 6) and an
  // upward in-match offset after winning twice at 25.
  assert.ok(o7.priorOffset! > 1, `offset ${o7.priorOffset}`);
  assert.ok(Math.abs(o8.priorOffset!) < 1, `bot 8 lost at a price far above its prior: offset ~0, got ${o8.priorOffset}`);
  const q = "Q".charCodeAt(0) - A;
  const e = "E".charCodeAt(0) - A;
  assert.ok(o7.prior!.mu[q] > o7.prior!.mu[e], "high-value letters priced higher");
  assert.ok(o7.prior!.mu[e] - o8.prior!.mu[e] > 5, "per-bot prior differs from the population");
  // The offset depends only on the match tables, and is cached.
  assert.equal(matchOffsets(der.data, me, ctx), matchOffsets(der.data, me, ctx));
  const noOnline = buildSnapshot(store, der, me, false, "k", { priors, online: { ...FILE.online, tau: 0 } });
  assert.equal(noOnline.opps[0].priorOffset, 0);
});
