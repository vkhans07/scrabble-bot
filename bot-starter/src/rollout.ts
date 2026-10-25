// Monte Carlo value table v[26] and the cash-in comparison.
//
// A scenario is a sequence of H future letters drawn from the unseen pool
// plus pre-drawn opponent jitters. Our own future bids depend only on the
// previous v table and our balance, never on the rack, so the set of letters
// we win is the same in every arm of a scenario. Each scenario therefore
// simulates the wins once, and every arm (rack + L, rack - word, ...) is just
// B() of a different rack on top of those wins: common random numbers.

import type { Config } from "./config.js";
import { Dict, LETTER_VALUE, countsOf, keyOfWord } from "./dict.js";
import { bidAmount, cashinGain, pacedLambda } from "./policy.js";
import { RackEval } from "./rackeval.js";
import type { Rng } from "./rng.js";
import { expectedPrices, firstPriceBid, payment, priceOf, wins, type Opp } from "./simmodel.js";
import type { Snapshot } from "./state.js";

export interface CashinCand {
  word: string;
  reward: number;
  delta: number; // E[B(rack - w + future)] - E[B(rack + future)]
  gain: number;
}

export interface RolloutResult {
  v: Float64Array;
  computed: number[]; // letters whose entry was recomputed
  nScen: number;
  ms: number;
  cashin: CashinCand[];
  evals: number;
}

const tick = () => new Promise<void>((r) => setImmediate(r));

export class Rollouts {
  vPrev = new Float64Array(26);
  private matchId = "";
  totalScenarios = 0;
  totalMs = 0;

  constructor(
    readonly dict: Dict,
    readonly ev: RackEval,
    readonly cfg: Config,
    readonly rng: Rng,
  ) {
    this.resetV();
  }

  private resetV() {
    for (let i = 0; i < 26; i++) this.vPrev[i] = LETTER_VALUE[i] + 1; // cold start
  }

  // Compute v for the current letter only, within a small time budget.
  fast(snap: Snapshot, abort: () => boolean): Promise<RolloutResult> {
    return this.run(snap, abort, { onlyCurrent: true });
  }

  full(snap: Snapshot, abort: () => boolean): Promise<RolloutResult> {
    return this.run(snap, abort, { onlyCurrent: false });
  }

  private async run(
    snap: Snapshot,
    abort: () => boolean,
    mode: { onlyCurrent: boolean },
  ): Promise<RolloutResult> {
    const { cfg, ev, dict, rng } = this;
    const t0 = performance.now();
    if (snap.matchId !== this.matchId) {
      this.matchId = snap.matchId;
      this.resetV();
    }
    const evals0 = ev.hits + ev.misses;
    const budget = mode.onlyCurrent ? cfg.fastBudgetMs : cfg.rolloutBudgetMs;
    const nMax = mode.onlyCurrent ? Math.min(cfg.nRollouts, 200) : cfg.nRollouts;

    // Pool of unseen letters, shuffled in place for each scenario.
    const pool: number[] = [];
    for (let i = 0; i < 26; i++) for (let k = 0; k < snap.U[i]; k++) pool.push(i);
    const H = Math.max(0, Math.min(cfg.horizon, snap.usableLeft, pool.length));

    const arm: number[] = [];
    if (mode.onlyCurrent) {
      if (snap.letter >= 0) arm.push(snap.letter);
    } else {
      for (let i = 0; i < 26; i++) if (snap.U[i] > 0 || i === snap.letter) arm.push(i);
    }

    // Cash-in candidates (full pass only, and only when the rack changed).
    const cands: { word: string; reward: number; counts: Uint8Array }[] = [];
    if (!mode.onlyCurrent && snap.needCashin && RackEval.total(snap.rack) >= 2) {
      const seen = new Set<number>();
      for (const e of ev.topFitting(snap.rack, cfg.cashinK)) {
        seen.add(e);
        cands.push({ word: dict.words[e], reward: dict.rewards[e], counts: countsOf(dict.keys[e]) });
      }
      const plan = ev.plan(snap.rack);
      if (plan.words.length > 0) {
        const w = plan.words[0];
        const e = dict.byKey.get(keyOfWord(w))!;
        if (!seen.has(e)) cands.push({ word: w, reward: dict.rewards[e], counts: countsOf(w) });
      }
    }

    const oppCfg = cfg;
    const nOpp = snap.opps.length;
    const us = new Array<number>(nOpp).fill(0);
    const fpCache = new Map<number, number>();
    const fpFor = (L: number) =>
      snap.firstPrice
        ? (value: number) => {
            const key = L * 100000 + Math.floor(value);
            let b = fpCache.get(key);
            if (b === undefined) {
              b = firstPriceBid(snap.opps, L, value, snap.balance, oppCfg, rng, 200);
              fpCache.set(key, b);
            }
            return b;
          }
        : undefined;

    // Paced mode: one price of a coin for the whole horizon.
    const lam =
      cfg.lambdaMode === "paced"
        ? pacedLambda(this.vPrev, snap.U, snap.balance, snap.usableLeft + 1, expectedPrices(snap.opps, oppCfg))
        : undefined;

    const sumBase = { v: 0 };
    const sumArm = new Float64Array(26);
    const sumCand = new Float64Array(cands.length);
    const withWon = new Uint8Array(26);
    const tmp = new Uint8Array(26);
    let n = 0;

    while (n < nMax) {
      if (abort()) break;
      if (n > 0 && performance.now() - t0 > budget) break;

      // 1. Simulate which future letters we win in this scenario.
      for (let i = 0; i < 26; i++) withWon[i] = snap.rack[i];
      let bal = snap.balance;
      for (let t = 0; t < H; t++) {
        const j = t + Math.floor(rng() * (pool.length - t));
        const L = pool[j];
        pool[j] = pool[t];
        pool[t] = L;
        for (let o = 0; o < nOpp; o++) us[o] = rng();
        const tieU = rng();
        const ul = snap.usableLeft - (t + 1);
        const d = bidAmount(this.vPrev[L], bal, ul, cfg, fpFor(L), lam);
        if (d.amount <= 0) continue;
        const pr = priceOf(snap.opps, L, us, oppCfg);
        if (!wins(d.amount, pr, tieU)) continue;
        const pay = payment(d.amount, pr, snap.firstPrice);
        if (pay > bal) continue;
        bal -= pay;
        withWon[L]++;
      }

      // 2. Evaluate every arm on top of those wins.
      const base = ev.best(withWon);
      sumBase.v += base;
      for (const L of arm) {
        withWon[L]++;
        sumArm[L] += ev.best(withWon) - base;
        withWon[L]--;
      }
      for (let c = 0; c < cands.length; c++) {
        const cc = cands[c].counts;
        for (let i = 0; i < 26; i++) tmp[i] = withWon[i] - cc[i];
        sumCand[c] += ev.best(tmp) - base;
      }
      n++;
      await tick();
    }

    const v = Float64Array.from(this.vPrev);
    if (n > 0) {
      const k = cfg.vShrink;
      for (const L of arm) {
        const est = (sumArm[L] + k * this.vPrev[L]) / (n + k);
        v[L] = this.vPrev[L] + cfg.vDamp * (est - this.vPrev[L]);
        this.vPrev[L] = v[L];
      }
    }
    const cashin: CashinCand[] = [];
    if (n > 0) {
      for (let c = 0; c < cands.length; c++) {
        const delta = sumCand[c] / n;
        cashin.push({
          word: cands[c].word,
          reward: cands[c].reward,
          delta,
          gain: cashinGain(cands[c].reward, snap.usableLeft, delta, cfg, lam),
        });
      }
      cashin.sort((a, b) => b.gain - a.gain);
    }
    const ms = performance.now() - t0;
    this.totalScenarios += n;
    this.totalMs += ms;
    return { v, computed: n > 0 ? arm : [], nScen: n, ms, cashin, evals: ev.hits + ev.misses - evals0 };
  }
}

export type { Opp };
