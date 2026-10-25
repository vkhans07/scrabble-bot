// Fit, validate and export per-opponent bid priors (oppmodel.ts) from public
// match histories. Offline only.
//
//   node --import tsx src/oppfit.ts holdout --data <dir>[,<dir>] --split <matchId> [--profiles f] [--spec name]
//   node --import tsx src/oppfit.ts select --data <dir>[,<dir>] --split <matchId>
//   node --import tsx src/oppfit.ts export --data <dir>[,<dir>] --bots bot.json --out priors.json [--spec name]
//
// holdout: fit on matches with id <= split, report held-out metrics on the rest
// for the bot's current global model, the existing linear profiles (if
// given) and the fitted models. select: the same, inside the training set
// (last 30% held out), over model specs and shrinkage strengths.
//
// Likelihood of one auction, bids independent across bots given context:
//   nobody won            -> every bid was 0
//   winner w, top bid t   -> P(b_w = t) *
//     paid p >= 2         -> P(max of the others = p)
//     paid p <= 1         -> P(every other bid <= 1)   (reserve paid)
// with every bid capped at the bidder's balance before the auction.

import * as fs from "node:fs";
import { loadBotNames, loadMatches, type AuctionObs, type MatchObs } from "./oppdata.js";
import {
  OnlineOffsets,
  bidCdf,
  bidDist,
  features,
  type BotParams,
  type Ctx,
  type ModelSpec,
  type OnlineCfg,
  type PriorsFile,
} from "./oppmodel.js";
import { LETTER_VALUE } from "./dict.js";

export const SPECS: Record<string, ModelSpec> = {
  base: { mu: ["one", "face", "vowel", "hv", "prog", "end", "final"], sig: ["one", "face"], pi: ["one", "final"] },
  bal: {
    mu: ["one", "face", "vowel", "hv", "prog", "end", "final", "lbal", "cpr", "lowbal"],
    sig: ["one", "face", "lbal"],
    pi: ["one", "final", "lowbal"],
  },
  rack: {
    mu: ["one", "face", "vowel", "hv", "prog", "end", "final", "dup", "nk", "vneed"],
    sig: ["one", "face"],
    pi: ["one", "final"],
  },
  full: {
    mu: ["one", "face", "vowel", "hv", "k5", "prog", "prog2", "end", "final", "lbal", "cpr", "lowbal", "dup", "nk", "vneed"],
    sig: ["one", "face", "lbal", "prog"],
    pi: ["one", "final", "lowbal"],
  },
};

const EPS = 0.003; // uniform-mixture robustness weight, in fitting and evaluation alike

// ---- dataset ------------------------------------------------------------------

interface Rows {
  spec: ModelSpec;
  nA: number; // auctions
  seat0: Int32Array; // first seat row of auction a
  nSeat: Int8Array;
  winner: Int8Array; // seat index within auction, -1 none
  top: Int32Array;
  paid: Int32Array;
  weight: Float64Array; // per auction
  // per seat row
  bot: Int32Array; // index into bots
  cap: Int32Array;
  xm: Float32Array;
  xs: Float32Array;
  xp: Float32Array;
  ctx: Ctx[];
  match: Int32Array; // per auction: match index
}

function buildRows(ms: MatchObs[], spec: ModelSpec, botIndex: Map<number, number>, weightOf: (m: MatchObs) => number): Rows {
  let nA = 0;
  let nS = 0;
  for (const m of ms) {
    nA += m.auctions.length;
    nS += m.auctions.length * m.bots.length;
  }
  const r: Rows = {
    spec,
    nA,
    seat0: new Int32Array(nA),
    nSeat: new Int8Array(nA),
    winner: new Int8Array(nA),
    top: new Int32Array(nA),
    paid: new Int32Array(nA),
    weight: new Float64Array(nA),
    bot: new Int32Array(nS),
    cap: new Int32Array(nS),
    xm: new Float32Array(nS * spec.mu.length),
    xs: new Float32Array(nS * spec.sig.length),
    xp: new Float32Array(nS * spec.pi.length),
    ctx: new Array(nS),
    match: new Int32Array(nA),
  };
  let a = 0;
  let s = 0;
  const fm = new Float64Array(spec.mu.length);
  const fs_ = new Float64Array(spec.sig.length);
  const fp = new Float64Array(spec.pi.length);
  ms.forEach((m, mi) => {
    const w = weightOf(m);
    for (const au of m.auctions) {
      r.seat0[a] = s;
      r.nSeat[a] = m.bots.length;
      r.winner[a] = au.winner;
      r.top[a] = au.top;
      r.paid[a] = au.paid;
      r.weight[a] = w;
      r.match[a] = mi;
      for (let q = 0; q < m.bots.length; q++) {
        const c: Ctx = { L: au.L, left: au.left, bal: au.bal[q], known: au.known[q] };
        r.ctx[s] = c;
        r.bot[s] = botIndex.get(m.bots[q]) ?? -1;
        r.cap[s] = au.bal[q];
        r.xm.set(features(spec.mu, c, fm), s * spec.mu.length);
        r.xs.set(features(spec.sig, c, fs_), s * spec.sig.length);
        r.xp.set(features(spec.pi, c, fp), s * spec.pi.length);
        s++;
      }
      a++;
    }
  });
  return r;
}

// ---- objective ------------------------------------------------------------------

// Parameter vector: population [P], then a deviation [P] per bot.
interface Layout {
  nm: number;
  ns: number;
  np: number;
  P: number;
  nBots: number;
}

function unpack(x: Float64Array, L: Layout, b: number): BotParams {
  const o = (b + 1) * L.P;
  const p0 = (k: number) => x[k] + (b >= 0 ? x[o + k] : 0);
  const mu = [];
  const sig = [];
  const pi = [];
  for (let k = 0; k < L.nm; k++) mu.push(p0(k));
  for (let k = 0; k < L.ns; k++) sig.push(p0(L.nm + k));
  for (let k = 0; k < L.np; k++) pi.push(p0(L.nm + L.ns + k));
  return { mu, sig, pi };
}

const LS_MIN = -3;
const LS_MAX = 4;

interface Fit {
  x: Float64Array;
  L: Layout;
  value: number;
  iters: number;
}

// Negative penalised log-likelihood and its gradient.
function objective(r: Rows, L: Layout, tau: Float64Array, x: Float64Array, g: Float64Array | null): number {
  const { nm, ns, np, P } = L;
  if (g) g.fill(0);
  // Per seat row: mu, sd, pi, and accumulated dLL/d(mu, logsd, logitpi).
  let nll = 0;
  const MAXS = 8;
  const mu = new Float64Array(MAXS);
  const sd = new Float64Array(MAXS);
  const pi = new Float64Array(MAXS);
  const lsOk = new Uint8Array(MAXS);
  const gm = new Float64Array(MAXS);
  const gl = new Float64Array(MAXS);
  const gp = new Float64Array(MAXS);
  const cA = new Float64Array(MAXS); // C(k1)
  const cB = new Float64Array(MAXS); // C(k1 - 1) or unused
  const dA = new Float64Array(MAXS * 3);
  const dB = new Float64Array(MAXS * 3);

  // C(k) with the epsilon mixture, and its partials into out[o..o+2].
  const cdf = (q: number, k: number, cap: number, out: Float64Array, o: number): number => {
    out[o] = out[o + 1] = out[o + 2] = 0;
    if (k < 0) return 0;
    if (k >= cap) return 1;
    const z = (k + 0.5 - mu[q]) / sd[q];
    const F = 1 / (1 + Math.exp(-z));
    const f = F * (1 - F);
    const p = pi[q];
    const c = p + (1 - p) * F;
    const e = 1 - EPS;
    out[o] = (e * -(1 - p) * f) / sd[q];
    out[o + 1] = lsOk[q] ? e * -(1 - p) * f * z : 0;
    out[o + 2] = e * p * (1 - p) * (1 - F);
    return e * c + (EPS * (k + 1)) / (cap + 1);
  };

  for (let a = 0; a < r.nA; a++) {
    const s0 = r.seat0[a];
    const n = r.nSeat[a];
    const wt = r.weight[a];
    if (wt <= 0) continue;
    for (let q = 0; q < n; q++) {
      const s = s0 + q;
      const b = r.bot[s];
      const o = (b + 1) * P;
      let m = 0;
      for (let k = 0; k < nm; k++) m += (x[k] + (b >= 0 ? x[o + k] : 0)) * r.xm[s * nm + k];
      let ls = 0;
      for (let k = 0; k < ns; k++) ls += (x[nm + k] + (b >= 0 ? x[o + nm + k] : 0)) * r.xs[s * ns + k];
      lsOk[q] = ls > LS_MIN && ls < LS_MAX ? 1 : 0;
      ls = Math.max(LS_MIN, Math.min(LS_MAX, ls));
      let lp = 0;
      for (let k = 0; k < np; k++) lp += (x[nm + ns + k] + (b >= 0 ? x[o + nm + ns + k] : 0)) * r.xp[s * np + k];
      mu[q] = m;
      sd[q] = Math.exp(ls);
      pi[q] = np ? 1 / (1 + Math.exp(-lp)) : 0;
      gm[q] = gl[q] = gp[q] = 0;
    }
    const w = r.winner[a];
    let ll = 0;
    const add = (q: number, scale: number, d: Float64Array) => {
      gm[q] += scale * d[q * 3];
      gl[q] += scale * d[q * 3 + 1];
      gp[q] += scale * d[q * 3 + 2];
    };
    if (w < 0) {
      for (let q = 0; q < n; q++) {
        const c = Math.max(cdf(q, 0, r.cap[s0 + q], dA, q * 3), 1e-300);
        ll += Math.log(c);
        add(q, 1 / c, dA);
      }
    } else {
      const t = r.top[a];
      const p = r.paid[a];
      const capW = r.cap[s0 + w];
      const hi = cdf(w, t, capW, dA, w * 3);
      const lo = cdf(w, t - 1, capW, dB, w * 3);
      const pm = Math.max(hi - lo, 1e-300);
      ll += Math.log(pm);
      add(w, 1 / pm, dA);
      add(w, -1 / pm, dB);
      if (p <= 1) {
        for (let q = 0; q < n; q++) {
          if (q === w) continue;
          const c = Math.max(cdf(q, 1, r.cap[s0 + q], dA, q * 3), 1e-300);
          ll += Math.log(c);
          add(q, 1 / c, dA);
        }
      } else {
        let PA = 1;
        let PB = 1;
        for (let q = 0; q < n; q++) {
          if (q === w) continue;
          cA[q] = cdf(q, p, r.cap[s0 + q], dA, q * 3);
          cB[q] = cdf(q, p - 1, r.cap[s0 + q], dB, q * 3);
          PA *= cA[q];
          PB *= cB[q];
        }
        const D = Math.max(PA - PB, 1e-300);
        ll += Math.log(D);
        if (g) {
          for (let q = 0; q < n; q++) {
            if (q === w) continue;
            let exA = 1;
            let exB = 1;
            for (let r2 = 0; r2 < n; r2++) {
              if (r2 === w || r2 === q) continue;
              exA *= cA[r2];
              exB *= cB[r2];
            }
            add(q, exA / D, dA);
            add(q, -exB / D, dB);
          }
        }
      }
    }
    nll -= wt * ll;
    if (!g) continue;
    for (let q = 0; q < n; q++) {
      const s = s0 + q;
      const b = r.bot[s];
      const o = (b + 1) * P;
      const am = -wt * gm[q];
      const al = -wt * gl[q];
      const ap = -wt * gp[q] * 1;
      for (let k = 0; k < nm; k++) {
        const v = am * r.xm[s * nm + k];
        g[k] += v;
        if (b >= 0) g[o + k] += v;
      }
      for (let k = 0; k < ns; k++) {
        const v = al * r.xs[s * ns + k];
        g[nm + k] += v;
        if (b >= 0) g[o + nm + k] += v;
      }
      for (let k = 0; k < np; k++) {
        const v = ap * r.xp[s * np + k];
        g[nm + ns + k] += v;
        if (b >= 0) g[o + nm + ns + k] += v;
      }
    }
  }
  // Shrinkage of every bot's deviation toward the population.
  for (let b = 0; b < L.nBots; b++) {
    const o = (b + 1) * P;
    for (let k = 0; k < P; k++) {
      const d = x[o + k];
      nll += (0.5 * d * d) / (tau[k] * tau[k]);
      if (g) g[o + k] += d / (tau[k] * tau[k]);
    }
  }
  return nll;
}

// ---- L-BFGS -----------------------------------------------------------------------

function lbfgs(f: (x: Float64Array, g: Float64Array | null) => number, x0: Float64Array, maxIter: number, log = false) {
  const n = x0.length;
  let x = x0.slice();
  let g = new Float64Array(n);
  let fx = f(x, g);
  const S: Float64Array[] = [];
  const Y: Float64Array[] = [];
  const M = 12;
  const d = new Float64Array(n);
  const xn = new Float64Array(n);
  const gn = new Float64Array(n);
  let it = 0;
  for (; it < maxIter; it++) {
    // two-loop recursion
    for (let i = 0; i < n; i++) d[i] = -g[i];
    const alpha: number[] = [];
    for (let j = S.length - 1; j >= 0; j--) {
      const rho = 1 / dotv(Y[j], S[j]);
      const a = rho * dotv(S[j], d);
      alpha[j] = a;
      for (let i = 0; i < n; i++) d[i] -= a * Y[j][i];
    }
    if (S.length) {
      const k = S.length - 1;
      const gamma = dotv(S[k], Y[k]) / dotv(Y[k], Y[k]);
      for (let i = 0; i < n; i++) d[i] *= gamma;
    }
    for (let j = 0; j < S.length; j++) {
      const rho = 1 / dotv(Y[j], S[j]);
      const b = rho * dotv(Y[j], d);
      for (let i = 0; i < n; i++) d[i] += S[j][i] * (alpha[j] - b);
    }
    let gd = dotv(g, d);
    if (!(gd < 0)) {
      S.length = 0;
      Y.length = 0;
      for (let i = 0; i < n; i++) d[i] = -g[i];
      gd = dotv(g, d);
    }
    let step = S.length ? 1 : Math.min(1, 1 / Math.sqrt(dotv(g, g)));
    let fn = Infinity;
    for (let ls = 0; ls < 40; ls++) {
      for (let i = 0; i < n; i++) xn[i] = x[i] + step * d[i];
      fn = f(xn, gn);
      if (Number.isFinite(fn) && fn <= fx + 1e-4 * step * gd) break;
      step *= 0.5;
    }
    if (!(fn < fx)) break;
    const s = new Float64Array(n);
    const y = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      s[i] = xn[i] - x[i];
      y[i] = gn[i] - g[i];
    }
    if (dotv(s, y) > 1e-12) {
      S.push(s);
      Y.push(y);
      if (S.length > M) {
        S.shift();
        Y.shift();
      }
    }
    const rel = (fx - fn) / Math.max(1, Math.abs(fx));
    x = xn.slice();
    g = gn.slice();
    fx = fn;
    if (log && it % 25 === 0) console.error(`  iter ${it} f=${fx.toFixed(2)}`);
    if (rel < 1e-9) break;
  }
  return { x, fx, it };
}

function dotv(a: Float64Array, b: Float64Array) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

// Natural scale of each parameter for shrinkage (multiplied by `c`).
function tauVector(spec: ModelSpec, c: number): Float64Array {
  const scale = (name: string, grp: "mu" | "sig" | "pi") => {
    if (grp === "sig") return 0.5;
    if (grp === "pi") return 1.5;
    if (name === "one") return 6;
    return 3;
  };
  const t: number[] = [];
  for (const n of spec.mu) t.push(scale(n, "mu") * c);
  for (const n of spec.sig) t.push(scale(n, "sig") * c);
  for (const n of spec.pi) t.push(scale(n, "pi") * c);
  return Float64Array.from(t);
}

export function fitModel(ms: MatchObs[], spec: ModelSpec, c: number, opts: { halfLifeH?: number; minMatches?: number; iters?: number; log?: boolean } = {}) {
  const counts = new Map<number, number>();
  for (const m of ms) for (const b of m.bots) counts.set(b, (counts.get(b) ?? 0) + 1);
  const botIds = [...counts.entries()].filter(([, n]) => n >= (opts.minMatches ?? 2)).map(([b]) => b);
  const botIndex = new Map<number, number>(botIds.map((b, i) => [b, i] as const));
  const tEnd = Math.max(...ms.map((m) => m.createdAt));
  const hl = opts.halfLifeH;
  const weightOf = (m: MatchObs) => (hl && hl > 0 ? Math.pow(0.5, (tEnd - m.createdAt) / 3.6e9 / hl) : 1);
  const rows = buildRows(ms, spec, botIndex, weightOf);
  const L: Layout = { nm: spec.mu.length, ns: spec.sig.length, np: spec.pi.length, P: 0, nBots: botIds.length };
  L.P = L.nm + L.ns + L.np;
  const tau = tauVector(spec, c);
  const x0 = new Float64Array((botIds.length + 1) * L.P);
  // Start near the global model: mu = 8 + 2.5 * face, sd ~ 6, pi ~ 5%.
  x0[spec.mu.indexOf("one")] = 8;
  if (spec.mu.includes("face")) x0[spec.mu.indexOf("face")] = 10;
  x0[L.nm + spec.sig.indexOf("one")] = Math.log(4);
  if (L.np) x0[L.nm + L.ns + spec.pi.indexOf("one")] = -3;
  const res = lbfgs((x, g) => objective(rows, L, tau, x, g), x0, opts.iters ?? 400, opts.log);
  const fit: Fit = { x: res.x, L, value: res.fx, iters: res.it };
  return { fit, botIndex, counts, spec, rows };
}

export type Fitted = ReturnType<typeof fitModel>;

export function paramsFor(f: Fitted, botId: number): BotParams {
  return unpack(f.fit.x, f.fit.L, f.botIndex.get(botId) ?? -1);
}

// ---- generic evaluation -----------------------------------------------------------

// A model maps one bot's situation to the CDF of its bid on 0..cap.
export interface EvalModel {
  name: string;
  cdf(botId: number, c: Ctx, cap: number, matchState: OnlineState | null): Float64Array;
  online?: OnlineCfg;
}

export interface OnlineState {
  off: OnlineOffsets;
}

function withEps(C: Float64Array): Float64Array {
  const cap = C.length - 1;
  for (let k = 0; k <= cap; k++) C[k] = (1 - EPS) * C[k] + (EPS * (k + 1)) / (cap + 1);
  return C;
}

export function globalModel(bonus = 8, slope = 2.5, jit = 4): EvalModel {
  return {
    name: `global ${bonus}+${slope}f±${jit}`,
    cdf(_b, c, cap) {
      const C = new Float64Array(cap + 1);
      if (cap <= 0) return C.fill(1);
      const base = Math.round(bonus + slope * LETTER_VALUE[c.L]);
      const n = 2 * jit + 1;
      for (let j = -jit; j <= jit; j++) {
        const b = Math.max(0, Math.min(Math.max(base + j, 1), cap));
        for (let k = b; k <= cap; k++) C[k] += 1 / n;
      }
      C[cap] = 1;
      return withEps(C);
    },
  };
}

interface LinProfile {
  botId: number;
  a: number;
  b_face: number;
  c_vowel: number;
  d_late: number;
  e_endgame: number;
  sigma: number;
}

const erf = (x: number) => {
  const t = 1 / (1 + 0.3275911 * Math.abs(x));
  const y = 1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
  return x >= 0 ? y : -y;
};
const Phi = (z: number) => 0.5 * (1 + erf(z / Math.SQRT2));

export function linearModel(file: string, fallback: EvalModel): EvalModel {
  const raw = JSON.parse(fs.readFileSync(file, "utf8")) as { profiles?: Record<string, LinProfile> };
  const all: Record<string, LinProfile> = raw.profiles ?? (raw as unknown as Record<string, LinProfile>);
  const byId = new Map<number, LinProfile>(Object.values(all).map((p) => [p.botId, p] as const));
  return {
    name: "linear profiles",
    cdf(b, c, cap, st) {
      const p = byId.get(b);
      if (!p) return fallback.cdf(b, c, cap, st);
      const C = new Float64Array(cap + 1);
      const prog = (67 - c.left) / 68;
      const mu = p.a + p.b_face * LETTER_VALUE[c.L] + p.c_vowel * (feature01vowel(c.L)) + p.d_late * prog + p.e_endgame * (c.left < 6 ? 1 : 0);
      for (let k = 0; k < cap; k++) C[k] = Phi((k + 0.5 - mu) / p.sigma);
      C[cap] = 1;
      return withEps(C);
    },
  };
}
const feature01vowel = (L: number) => ([0, 4, 8, 14, 20].includes(L) ? 1 : 0);

export function fittedModel(f: Fitted, name: string, online?: OnlineCfg, perBot = true): EvalModel {
  const cache = new Map<number, BotParams>();
  const pop = unpack(f.fit.x, f.fit.L, -1);
  const pf = (b: number) => {
    if (!perBot) return pop;
    let p = cache.get(b);
    if (!p) cache.set(b, (p = paramsFor(f, b)));
    return p;
  };
  return {
    name,
    online,
    cdf(b, c, cap, st) {
      const d = bidDist(f.spec, pf(b), c, st ? st.off.offset(String(b)) : 0);
      const C = new Float64Array(cap + 1);
      for (let k = 0; k <= cap; k++) C[k] = bidCdf(k, d, cap);
      return withEps(C);
    },
  };
}

// Distribution handed to the online update (offset 0, no epsilon).
function priorDist(m: EvalModel, f: Fitted | null, b: number, c: Ctx) {
  if (!f) return null;
  void m;
  return bidDist(f.spec, paramsFor(f, b), c);
}

interface Acc {
  n: number;
  ll: number; // joint observation log-lik
  llPaid: number;
  paidAE: number;
  paidBias: number;
  topAE: number;
  winLL: number;
  winHit: number;
  // price to beat (max of the other bids), from each seat / from our seat
  ptbN: number;
  ptbLL: number;
  ptbAE: number;
  ptbBias: number;
  usN: number;
  usLL: number;
  usAE: number;
  usBias: number;
  usCover80: number;
  byFace: Map<string, { n: number; ae: number; bias: number; usN: number; usAE: number; usBias: number }>;
}

const newAcc = (): Acc => ({
  n: 0, ll: 0, llPaid: 0, paidAE: 0, paidBias: 0, topAE: 0, winLL: 0, winHit: 0,
  ptbN: 0, ptbLL: 0, ptbAE: 0, ptbBias: 0, usN: 0, usLL: 0, usAE: 0, usBias: 0, usCover80: 0, byFace: new Map(),
});

const faceBucket = (L: number) => {
  const f = LETTER_VALUE[L];
  return f === 1 ? "1" : f <= 3 ? "2-3" : f <= 5 ? "4-5" : "8-10";
};

const at = (C: Float64Array, k: number) => (k < 0 ? 0 : k >= C.length ? 1 : C[k]);
const median = (G: (k: number) => number, K: number) => {
  for (let k = 0; k <= K; k++) if (G(k) >= 0.5) return k;
  return K;
};
const quant = (G: (k: number) => number, K: number, q: number) => {
  for (let k = 0; k <= K; k++) if (G(k) >= q) return k;
  return K;
};
const meanOf = (G: (k: number) => number, K: number) => {
  let s = 0;
  for (let k = 0; k < K; k++) s += 1 - G(k);
  return s;
};

// The final auction's letter cannot be played; `skipFinal` leaves it out.
export function evaluate(ms: MatchObs[], model: EvalModel, us: number, fit: Fitted | null, skipFinal = SKIP_FINAL): Acc {
  const acc = newAcc();
  for (const m of ms) {
    const st: OnlineState | null = model.online && model.online.tau > 0 ? { off: new OnlineOffsets(model.online) } : null;
    for (const au of m.auctions) if (!(skipFinal && au.left === 0)) evalAuction(m, au, model, st, us, acc, fit);
  }
  return acc;
}
const SKIP_FINAL = process.argv.includes("--nofinal");

function evalAuction(m: MatchObs, au: AuctionObs, model: EvalModel, st: OnlineState | null, us: number, acc: Acc, fit: Fitted | null) {
  const n = m.bots.length;
  const ctxs: Ctx[] = m.bots.map((_, q) => ({ L: au.L, left: au.left, bal: au.bal[q], known: au.known[q] }));
  const C = m.bots.map((b, q) => model.cdf(b, ctxs[q], au.bal[q], st));
  const K = Math.max(...au.bal, 1);
  const G1 = (k: number) => C.reduce((p, c) => p * at(c, k), 1);
  const G2 = (k: number) => {
    let s = G1(k);
    for (let q = 0; q < n; q++) {
      let p = 1 - at(C[q], k);
      for (let r = 0; r < n; r++) if (r !== q) p *= at(C[r], k);
      s += p;
    }
    return s;
  };
  // paid: 0 if nobody bid, at least 1 otherwise
  const H = (k: number) => (k <= 0 ? G1(0) : G2(k));
  const w = au.winner;
  const p = au.paid;
  acc.n++;
  // joint log-lik
  let ll: number;
  if (w < 0) ll = C.reduce((s, c) => s + Math.log(Math.max(at(c, 0), 1e-300)), 0);
  else {
    ll = Math.log(Math.max(at(C[w], au.top) - at(C[w], au.top - 1), 1e-300));
    if (p <= 1) for (let q = 0; q < n; q++) { if (q !== w) ll += Math.log(Math.max(at(C[q], 1), 1e-300)); }
    else {
      let A = 1, B = 1;
      for (let q = 0; q < n; q++) if (q !== w) { A *= at(C[q], p); B *= at(C[q], p - 1); }
      ll += Math.log(Math.max(A - B, 1e-300));
    }
  }
  acc.ll += ll;
  acc.llPaid += Math.log(Math.max(H(p) - (p > 0 ? H(p - 1) : 0), 1e-300));
  const pMed = median(H, K);
  const pMean = meanOf(H, K);
  acc.paidAE += Math.abs(pMed - p);
  acc.paidBias += pMean - p;
  acc.topAE += Math.abs(median(G1, K) - au.top);
  // winner
  const pw = new Float64Array(n + 1);
  pw[n] = G1(0);
  for (let q = 0; q < n; q++) {
    let s = 0;
    for (let k = 1; k <= au.bal[q]; k++) {
      const pk = at(C[q], k) - at(C[q], k - 1);
      if (pk <= 0) continue;
      let o = 1;
      for (let r = 0; r < n; r++) if (r !== q) o *= at(C[r], k - 1) + 0.5 * (at(C[r], k) - at(C[r], k - 1));
      s += pk * o;
    }
    pw[q] = s;
  }
  const tot = pw.reduce((a, b) => a + b, 0);
  const wi = w < 0 ? n : w;
  acc.winLL += Math.log(Math.max(pw[wi] / tot, 1e-12));
  let best = 0;
  for (let q = 1; q <= n; q++) if (pw[q] > pw[best]) best = q;
  if (best === wi) acc.winHit++;
  // price to beat for each seat: max of the other bids
  const fb = faceBucket(au.L);
  let fbA = acc.byFace.get(fb);
  if (!fbA) acc.byFace.set(fb, (fbA = { n: 0, ae: 0, bias: 0, usN: 0, usAE: 0, usBias: 0 }));
  fbA.n++;
  fbA.ae += Math.abs(pMed - p);
  fbA.bias += pMean - p;
  for (let q = 0; q < n; q++) {
    const M = (k: number) => {
      let s = 1;
      for (let r = 0; r < n; r++) if (r !== q) s *= at(C[r], k);
      return s;
    };
    // observed: winner sees the runner-up (<=1 when the reserve was paid);
    // everyone else sees the winner's bid; 0 if nobody bid.
    let obs: number;
    let llq: number;
    if (w < 0) {
      obs = 0;
      llq = Math.log(Math.max(M(0), 1e-300));
    } else if (q === w) {
      obs = p <= 1 ? 1 : p;
      llq = p <= 1 ? Math.log(Math.max(M(1), 1e-300)) : Math.log(Math.max(M(p) - M(p - 1), 1e-300));
    } else {
      obs = au.top;
      llq = Math.log(Math.max(M(obs) - M(obs - 1), 1e-300));
    }
    const med = median(M, K);
    const mean = meanOf(M, K);
    acc.ptbN++;
    acc.ptbLL += llq;
    acc.ptbAE += Math.abs(med - obs);
    acc.ptbBias += mean - obs;
    if (m.bots[q] === us) {
      acc.usN++;
      acc.usLL += llq;
      acc.usAE += Math.abs(med - obs);
      acc.usBias += mean - obs;
      const lo = quant(M, K, 0.1);
      const hi = quant(M, K, 0.9);
      if (obs >= lo && obs <= hi) acc.usCover80++;
      fbA.usN++;
      fbA.usAE += Math.abs(med - obs);
      fbA.usBias += mean - obs;
    }
  }
  // online update from what bots see live
  if (st && fit) {
    for (let q = 0; q < n; q++) {
      const d = priorDist(model, fit, m.bots[q], ctxs[q]);
      if (d) st.off.observe(String(m.bots[q]), d, au.bal[q], q === w, w < 0 ? 0 : p);
    }
  }
}

export function summarize(name: string, a: Acc) {
  const f = (x: number, d = 3) => x.toFixed(d);
  const byFace = [...a.byFace.entries()].sort().map(([k, v]) => `${k}:${f(v.ae / v.n, 2)}/${f(v.bias / v.n, 2)}`).join(" ");
  const usFace = [...a.byFace.entries()].sort().map(([k, v]) => `${k}:${f(v.usAE / Math.max(1, v.usN), 2)}/${f(v.usBias / Math.max(1, v.usN), 2)}`).join(" ");
  return {
    model: name,
    auctions: a.n,
    llPerAuction: f(a.ll / a.n),
    llPaid: f(a.llPaid / a.n),
    paidMAE: f(a.paidAE / a.n, 2),
    paidBias: f(a.paidBias / a.n, 2),
    topMAE: f(a.topAE / a.n, 2),
    winnerLogLoss: f(-a.winLL / a.n),
    winnerAcc: f(a.winHit / a.n),
    ptbLL: f(a.ptbLL / a.ptbN),
    ptbMAE: f(a.ptbAE / a.ptbN, 2),
    ptbBias: f(a.ptbBias / a.ptbN, 2),
    usN: a.usN,
    usLL: f(a.usLL / Math.max(1, a.usN)),
    usMAE: f(a.usAE / Math.max(1, a.usN), 2),
    usBias: f(a.usBias / Math.max(1, a.usN), 2),
    usCover80: f(a.usCover80 / Math.max(1, a.usN)),
    paidMAE_bias_byFace: byFace,
    usMAE_bias_byFace: usFace,
  };
}

// ---- CLI ----------------------------------------------------------------------------

function arg(name: string, def?: string): string | undefined {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 ? process.argv[i + 1] : def;
}

function main() {
  const cmd = process.argv[2];
  const data = (arg("data") ?? "").split(",").filter(Boolean);
  if (!data.length) throw new Error("--data <dir>[,<dir>] required");
  const all = loadMatches(data);
  const us = Number(arg("us", "112"));
  const c = Number(arg("c", "1"));
  const hl = Number(arg("halflife", "0"));
  const specName = arg("spec", "base")!;
  const iters = Number(arg("iters", "400"));
  const tauOn = Number(arg("online", "0"));
  console.error(`${all.length} matches (${all[0]?.id}..${all[all.length - 1]?.id})`);
  if (cmd === "export") {
    const out = arg("out");
    if (!out) throw new Error("--out required");
    const names = arg("bots") ? loadBotNames(arg("bots")!) : new Map<number, string>();
    const f = fitModel(all, SPECS[specName], c, { halfLifeH: hl, iters, log: true });
    const bots: PriorsFile["bots"] = {};
    for (const [b] of f.botIndex) {
      if (b === us) continue;
      bots[String(b)] = { name: names.get(b) ?? String(b), matches: f.counts.get(b) ?? 0, params: roundP(paramsFor(f, b)) };
    }
    const pf: PriorsFile = {
      version: 1,
      fitted: new Date().toISOString(),
      source: `${all.length} public matches ${all[0].id}..${all[all.length - 1].id}; spec=${specName} c=${c} halflife=${hl}h`,
      spec: SPECS[specName],
      population: roundP(paramsFor(f, -1)),
      bots,
      online: { tau: tauOn, range: 24, step: 1 },
    };
    fs.writeFileSync(out, JSON.stringify(pf, null, 1));
    console.error(`wrote ${out}: ${Object.keys(bots).length} bots`);
    return;
  }
  const split = Number(arg("split"));
  if (!Number.isFinite(split)) throw new Error("--split <last training match id> required");
  let train = all.filter((m) => m.id <= split);
  let test = all.filter((m) => m.id > split);
  if (cmd === "select") {
    const cut = Math.floor(train.length * 0.7);
    test = train.slice(cut);
    train = train.slice(0, cut);
  }
  console.error(`train ${train.length}, test ${test.length}`);
  const rows: object[] = [];
  const t0 = Date.now();
  if (cmd === "holdout") {
    const g = globalModel();
    rows.push(summarize(g.name, evaluate(test, g, us, null)));
    if (arg("profiles")) {
      const lm = linearModel(arg("profiles")!, g);
      rows.push(summarize(lm.name, evaluate(test, lm, us, null)));
    }
  }
  const specs = (arg("specs") ?? specName).split(",");
  const cs = (arg("cs") ?? String(c)).split(",").map(Number);
  const hls = (arg("halflives") ?? String(hl)).split(",").map(Number);
  const taus = (arg("onlines") ?? String(tauOn)).split(",").map(Number);
  for (const sp of specs)
    for (const cc of cs)
      for (const h of hls) {
        const f = fitModel(train, SPECS[sp], cc, { halfLifeH: h, iters });
        console.error(`fit ${sp} c=${cc} hl=${h}: f=${f.fit.value.toFixed(1)} iters=${f.fit.iters} (${((Date.now() - t0) / 1000).toFixed(0)}s)`);
        if (cmd === "holdout" || arg("pop")) rows.push(summarize(`${sp} population-only`, evaluate(test, fittedModel(f, "pop", undefined, false), us, null)));
        for (const tau of taus) {
          const on = tau > 0 ? { tau, range: 24, step: 1 } : undefined;
          rows.push(summarize(`${sp} c=${cc} hl=${h} online=${tau}`, evaluate(test, fittedModel(f, sp, on), us, f)));
        }
      }
  console.log(JSON.stringify(rows, null, 1));
}

function roundP(p: BotParams): BotParams {
  const r = (a: number[]) => a.map((v) => Math.round(v * 1000) / 1000);
  return { mu: r(p.mu), sig: r(p.sig), pi: r(p.pi) };
}

export const _test = { objective, buildRows, tauVector };

if (process.argv[1] && process.argv[1].endsWith("oppfit.ts")) main();
