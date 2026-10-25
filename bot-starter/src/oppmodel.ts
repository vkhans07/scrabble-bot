// Per-opponent bid model fitted from public match histories (oppfit.ts),
// shared by the offline fitter and the live bot.
//
// An opponent's bid on a letter is
//
//   0                                   with probability pi   (no bid / missed)
//   clamp(round(mu + sd * e), 0, bal)   otherwise, e ~ standard logistic
//
// where mu, log(sd) and logit(pi) are linear in context features (letter,
// phase, balance, the letters the bot is known to hold). Parameters are a
// population mean plus a per-bot deviation shrunk toward it; a bot with no
// history gets the population mean. In a match, `OnlineOffsets` shifts each
// opponent's mu by a per-match offset inferred from the winners and clearing
// prices seen so far.
//
// The live bot hands the engine one (mu, sd) per letter per opponent and a
// pi (engine opponent kind 4, see native.ts / engine.cpp).

import { LETTER_VALUE } from "./dict.js";

export interface ModelSpec {
  mu: string[];
  sig: string[];
  pi: string[];
}

export interface BotParams {
  mu: number[];
  sig: number[];
  pi: number[];
}

export interface OnlineCfg {
  // Prior SD of a bot's per-match offset to mu; 0 disables the update.
  tau: number;
  // Offsets considered: -range..range in steps of `step`.
  range: number;
  step: number;
}

export interface PriorBot {
  name: string;
  matches: number;
  params: BotParams;
}

export interface PriorsFile {
  version: 1;
  fitted: string; // ISO date
  source: string;
  spec: ModelSpec;
  population: BotParams;
  bots: Record<string, PriorBot>; // by bot id
  online: OnlineCfg;
}

export interface Ctx {
  L: number; // letter index
  left: number; // auctions after this one (live: match.bagTotal)
  bal: number;
  known: ArrayLike<number>; // letters the bot is known to hold (won, not played)
}

export const TOTAL_AUCTIONS = 68;
const VOWEL = [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0];

function knownStats(known: ArrayLike<number>) {
  let n = 0;
  let v = 0;
  for (let i = 0; i < 26; i++) {
    n += known[i];
    if (VOWEL[i]) v += known[i];
  }
  return { n, v };
}

// Every feature is roughly unit scale so one shrinkage scale fits all.
export function feature(name: string, c: Ctx): number {
  const face = LETTER_VALUE[c.L];
  const prog = Math.max(0, Math.min(1, (TOTAL_AUCTIONS - 1 - c.left) / (TOTAL_AUCTIONS - 1)));
  switch (name) {
    case "one":
      return 1;
    case "face":
      return face / 4;
    case "vowel":
      return VOWEL[c.L];
    case "hv": // J Q X Z
      return face >= 8 ? 1 : 0;
    case "k5": // K
      return face === 5 ? 1 : 0;
    case "prog":
      return prog;
    case "prog2":
      return prog * prog;
    case "end": // the last few contested auctions
      return c.left > 0 && c.left < 6 ? 1 : 0;
    case "final":
      return c.left === 0 ? 1 : 0;
    case "lbal":
      return Math.log((Math.max(0, c.bal) + 1) / 101);
    case "cpr": // coins per remaining auction, capped
      return Math.min(c.bal / (c.left + 1), 20) / 10;
    case "lowbal":
      return Math.max(0, 30 - c.bal) / 30;
    case "dup":
      return Math.min(c.known[c.L], 2);
    case "nk":
      return knownStats(c.known).n / 10;
    case "vneed": {
      const k = knownStats(c.known);
      // A vowel when the known holdings are consonant-heavy, minus a vowel
      // when they are vowel-heavy (and the mirror for consonants).
      const vf = k.n > 0 ? k.v / k.n : 0.4;
      return (VOWEL[c.L] ? 1 : -1) * (0.4 - vf);
    }
    default:
      if (name.startsWith("L:")) return c.L === name.charCodeAt(2) - 65 ? 1 : 0;
      throw new Error(`unknown feature ${name}`);
  }
}

export function features(names: string[], c: Ctx, out = new Float64Array(names.length)): Float64Array {
  for (let k = 0; k < names.length; k++) out[k] = feature(names[k], c);
  return out;
}

export interface BidDist {
  mu: number;
  sd: number;
  pi: number;
}

const dot = (w: ArrayLike<number>, x: ArrayLike<number>) => {
  let s = 0;
  for (let k = 0; k < w.length; k++) s += w[k] * x[k];
  return s;
};
const sigmoid = (z: number) => 1 / (1 + Math.exp(-z));

export function bidDist(spec: ModelSpec, p: BotParams, c: Ctx, offset = 0): BidDist {
  return {
    mu: dot(p.mu, features(spec.mu, c)) + offset,
    sd: Math.exp(Math.max(-3, Math.min(4, dot(p.sig, features(spec.sig, c))))),
    pi: p.pi.length ? sigmoid(dot(p.pi, features(spec.pi, c))) : 0,
  };
}

// P(bid <= k) for a bid capped at `cap` (the bot's balance).
export function bidCdf(k: number, d: BidDist, cap: number): number {
  if (k < 0) return 0;
  if (k >= cap) return 1;
  return d.pi + (1 - d.pi) * sigmoid((k + 0.5 - d.mu) / d.sd);
}

// Sample a bid from a pre-drawn uniform; matches the engine's kind-4 sampler.
export function sampleBid(d: BidDist, cap: number, u: number): number {
  if (cap <= 0 || u < d.pi) return 0;
  const v = (u - d.pi) / (1 - d.pi);
  const q = Math.max(1e-9, Math.min(1 - 1e-9, v));
  const y = d.mu + d.sd * Math.log(q / (1 - q));
  return Math.max(0, Math.min(Math.round(y), cap));
}

// ---- in-match update --------------------------------------------------------

// Per-opponent posterior over an additive offset to mu, on a grid, from what
// every bot sees live: the winner and the clearing price. A winner bid at
// least the price; everyone else bid at most it. (The exact top bid is not
// visible to bots.)
export class OnlineOffsets {
  private grid: Float64Array;
  private logPost = new Map<string, Float64Array>();

  constructor(private cfg: OnlineCfg) {
    const n = cfg.tau > 0 ? Math.floor(cfg.range / cfg.step) * 2 + 1 : 1;
    this.grid = new Float64Array(n);
    for (let i = 0; i < n; i++) this.grid[i] = (i - (n - 1) / 2) * cfg.step;
  }

  private post(bot: string): Float64Array {
    let lp = this.logPost.get(bot);
    if (!lp) {
      lp = new Float64Array(this.grid.length);
      const t = this.cfg.tau;
      for (let i = 0; i < lp.length; i++) lp[i] = t > 0 ? -0.5 * (this.grid[i] / t) ** 2 : 0;
      this.logPost.set(bot, lp);
    }
    return lp;
  }

  // `d` is the bot's prior-model distribution for this auction (offset 0).
  observe(bot: string, d: BidDist, cap: number, won: boolean, paid: number) {
    if (this.cfg.tau <= 0 || cap <= 0) return;
    const lp = this.post(bot);
    for (let i = 0; i < lp.length; i++) {
      const di = { mu: d.mu + this.grid[i], sd: d.sd, pi: d.pi };
      const pr = won ? 1 - bidCdf(paid - 1, di, cap) : bidCdf(paid, di, cap);
      lp[i] += Math.log(Math.max(pr, 1e-9));
    }
  }

  // Posterior mean offset.
  offset(bot: string): number {
    if (this.cfg.tau <= 0) return 0;
    const lp = this.logPost.get(bot);
    if (!lp) return 0;
    let mx = -Infinity;
    for (const x of lp) mx = Math.max(mx, x);
    let z = 0;
    let s = 0;
    for (let i = 0; i < lp.length; i++) {
      const w = Math.exp(lp[i] - mx);
      z += w;
      s += w * this.grid[i];
    }
    return s / z;
  }
}

// ---- priors lookup ----------------------------------------------------------

export class OppPriors {
  constructor(readonly file: PriorsFile) {}

  params(botId: string): BotParams {
    return this.file.bots[botId]?.params ?? this.file.population;
  }

  known(botId: string): boolean {
    return botId in this.file.bots;
  }

  dist(botId: string, c: Ctx, offset = 0): BidDist {
    return bidDist(this.file.spec, this.params(botId), c, offset);
  }

  // One distribution per letter at the current balance / phase / holdings:
  // what the engine samples for this and every later auction of the rollout.
  letterTable(botId: string, c: Omit<Ctx, "L">, offset = 0): { mu: Float64Array; sd: Float64Array; pi: Float64Array } {
    const mu = new Float64Array(26);
    const sd = new Float64Array(26);
    const pi = new Float64Array(26);
    for (let L = 0; L < 26; L++) {
      const d = this.dist(botId, { ...c, L }, offset);
      mu[L] = d.mu;
      sd[L] = d.sd;
      pi[L] = d.pi;
    }
    return { mu, sd, pi };
  }
}

export function parsePriors(text: string): OppPriors {
  const f = JSON.parse(text) as PriorsFile;
  if (f.version !== 1 || !f.spec || !f.population) throw new Error("not an opponent priors file (version 1)");
  const n = (a: unknown[], b: unknown[], what: string) => {
    if (a.length !== b.length) throw new Error(`priors: ${what} has ${a.length} params for ${b.length} features`);
  };
  const check = (p: BotParams, who: string) => {
    n(p.mu, f.spec.mu, `${who} mu`);
    n(p.sig, f.spec.sig, `${who} sig`);
    n(p.pi, f.spec.pi, `${who} pi`);
  };
  check(f.population, "population");
  for (const [id, b] of Object.entries(f.bots)) check(b.params, id);
  for (const name of [...f.spec.mu, ...f.spec.sig, ...f.spec.pi])
    feature(name, { L: 0, left: 10, bal: 50, known: new Uint8Array(26) });
  f.online ??= { tau: 0, range: 0, step: 1 };
  return new OppPriors(f);
}
