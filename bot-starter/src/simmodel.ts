// Opponent bid models and the price a future auction clears at.
//
// Sims are modelled exactly (see game.ts decideBid): clamp(base + jitter, 1,
// balance) with jitter uniform on {-2..2}. Real opponents get a configurable
// guess: face + REAL_OPP_BASE_BONUS with uniform +-REAL_OPP_JITTER.

import { A, LETTER_VALUE } from "./dict.js";
import { makeRng, type Rng } from "./rng.js";

export type OppKind = "Cheapskate" | "ValueBidder" | "Aggressive" | "Real";

export interface Opp {
  botId: string;
  balance: number;
  kind: OppKind;
}

export interface OppModelCfg {
  realOppBaseBonus: number;
  realOppSlope: number;
  realOppJitter: number;
}

const VOWELS = new Set([0, 4, 8, 14, 20]); // A E I O U

// `u` is a pre-drawn uniform in [0,1), so scenarios can be reused across arms.
export function oppBid(opp: Opp, letter: number, u: number, cfg: OppModelCfg): number {
  if (opp.balance <= 0) return 0;
  const face = LETTER_VALUE[letter];
  let base: number;
  let jitter: number;
  switch (opp.kind) {
    case "Cheapskate":
      base = face - 1;
      jitter = Math.floor(u * 5) - 2;
      break;
    case "ValueBidder":
      base = face;
      jitter = Math.floor(u * 5) - 2;
      break;
    case "Aggressive":
      base = face + (VOWELS.has(letter) ? 4 : 2);
      jitter = Math.floor(u * 5) - 2;
      break;
    default: {
      const j = Math.max(0, Math.floor(cfg.realOppJitter));
      base = Math.round(cfg.realOppBaseBonus + cfg.realOppSlope * face);
      jitter = Math.floor(u * (2 * j + 1)) - j;
    }
  }
  return Math.max(0, Math.min(Math.max(base + jitter, 1), opp.balance));
}

export interface Price {
  P: number; // highest opposing bid, 0 if nobody bids
  tied: number; // how many opponents bid exactly P
}

export function priceOf(
  opps: Opp[],
  letter: number,
  us: ArrayLike<number>,
  cfg: OppModelCfg,
): Price {
  let P = 0;
  let tied = 0;
  for (let o = 0; o < opps.length; o++) {
    const b = oppBid(opps[o], letter, us[o], cfg);
    if (b > P) {
      P = b;
      tied = 1;
    } else if (b === P && b > 0) tied++;
  }
  return { P, tied };
}

// Does a bid of `bid` beat a price? Ties go to us with probability 1/(tied+1).
export function wins(bid: number, pr: Price, tieU: number): boolean {
  if (bid <= 0) return false;
  if (bid > pr.P) return true;
  if (bid === pr.P) return tieU < 1 / (pr.tied + 1);
  return false;
}

// What we pay on a win. Vickrey: the runner-up's bid, or the reserve of 1.
export function payment(bid: number, pr: Price, firstPrice: boolean): number {
  if (firstPrice) return bid;
  return Math.min(bid, Math.max(pr.P, 1));
}

// First-price: sample the distribution of the best opposing bid for a letter
// and return the integer bid maximising P(win) * (value - bid).
export function firstPriceBid(
  opps: Opp[],
  letter: number,
  value: number,
  balance: number,
  cfg: OppModelCfg,
  rng: Rng,
  samples = 500,
): number {
  const hi = Math.min(balance, Math.floor(value));
  if (hi < 1) return 0;
  const us = new Array<number>(opps.length);
  const prices: Price[] = [];
  for (let s = 0; s < samples; s++) {
    for (let o = 0; o < opps.length; o++) us[o] = rng();
    prices.push(priceOf(opps, letter, us, cfg));
  }
  let bestB = 0;
  let bestEv = 0;
  for (let b = 1; b <= hi; b++) {
    let w = 0;
    for (const pr of prices) {
      if (b > pr.P) w++;
      else if (b === pr.P) w += 1 / (pr.tied + 1);
    }
    const ev = (w / samples) * (value - b);
    if (ev > bestEv) {
      bestEv = ev;
      bestB = b;
    }
  }
  return bestB;
}

// Mean best-opposing-bid per letter under the current model, from a fixed
// seed so the worker and the main thread agree.
export function expectedPrices(opps: Opp[], cfg: OppModelCfg, samples = 40): Float64Array {
  const out = new Float64Array(26);
  const us = new Array<number>(opps.length).fill(0);
  for (let L = 0; L < 26; L++) {
    const rng = makeRng(777 + L);
    let sum = 0;
    for (let s = 0; s < samples; s++) {
      for (let o = 0; o < opps.length; o++) us[o] = rng();
      sum += priceOf(opps, L, us, cfg).P;
    }
    out[L] = sum / samples;
  }
  return out;
}

export const letterOfIndex = (i: number) => String.fromCharCode(A + i);
