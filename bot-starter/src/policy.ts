// Bid rule, lambda modes and endgame rules. Pure functions: the rollouts call
// them to simulate our own future bids, and the main thread calls them to bid.
//
// `usableLeft` everywhere is max(0, bagTotal - 1) read when the auction
// opens. The final auction (bagTotal == 0) is never bid on.

import type { Config } from "./config.js";

// Usable auctions at the start of a standard match (68 draws - the final one).
const FULL_USABLE = 66;

export function lambdaFor(usableLeft: number, cfg: Pick<Config, "lambdaMode" | "allinK">): number {
  if (cfg.lambdaMode === "linear") {
    const frac = Math.max(0, Math.min(1, usableLeft / FULL_USABLE));
    return 0.1 + 0.9 * frac;
  }
  return 1;
}

export function inAllIn(usableLeft: number, cfg: Pick<Config, "lambdaMode" | "allinK">): boolean {
  return cfg.lambdaMode !== "linear" && usableLeft < cfg.allinK;
}

export interface BidDecision {
  amount: number;
  reason: string;
}

type BidCfg = Pick<Config, "lambdaMode" | "allinK" | "allinMinV" | "safeMaxBid">;

// `fp`, when given, maps a value to the first-price-optimal bid.
export function bidAmount(
  v: number,
  balance: number,
  usableLeft: number,
  cfg: BidCfg,
  fp?: (value: number) => number,
  lamOverride?: number,
): BidDecision {
  if (balance < 1) return { amount: 0, reason: "no balance" };
  if (!(v > 0)) return { amount: 0, reason: "v<=0" };
  // All-in is only safe when we pay the runner-up's price (Vickrey); under
  // first-price it would pay the whole balance, so use the solver instead.
  if (!fp && inAllIn(usableLeft, cfg) && v >= cfg.allinMinV) {
    return { amount: balance, reason: `all-in (usableLeft=${usableLeft}, v=${v.toFixed(2)})` };
  }
  const lam = lamOverride ?? lambdaFor(usableLeft, cfg);
  const value = v / lam;
  const raw = fp ? fp(value) : Math.floor(value);
  const amount = Math.min(balance, raw);
  if (amount < 1) return { amount: 0, reason: `bid<1 (v=${v.toFixed(2)}, lambda=${lam.toFixed(2)})` };
  return {
    amount,
    reason: `${fp ? "fp" : "v/lambda"} v=${v.toFixed(2)} lambda=${lam.toFixed(2)}`,
  };
}

export function applySafeCap(amount: number, cfg: Pick<Config, "safeMaxBid">): number {
  return cfg.safeMaxBid === null ? amount : Math.min(amount, cfg.safeMaxBid);
}

type ComposeCfg = Pick<Config, "safeMaxBid" | "allinK" | "nativePaidBid" | "paidBidWarmup" | "earlyLambdaAuctions" | "earlyLambdaMin" | "maxBidFraction" | "maxBidFloor" | "reserveStart" | "reserveUntil" | "highLetterBoost" | "q3Boost">;

// J, K, Q, X, Z: the leaders play twice as many of these as we do.
const HIGH_LETTERS = new Set([9, 10, 16, 23, 25]);

// Third quarter of a standard match, by usable auctions left.
export function inQ3(usableLeft: number): boolean {
  return usableLeft <= FULL_USABLE / 2 && usableLeft > FULL_USABLE / 4;
}

// Coins to hold back with `usableLeft` usable auctions to go.
export function cashReserve(usableLeft: number, cfg: Pick<Config, "reserveStart" | "reserveUntil">): number {
  if (!(cfg.reserveStart > 0) || usableLeft <= cfg.reserveUntil) return 0;
  const frac = Math.min(1, (usableLeft - cfg.reserveUntil) / Math.max(1, FULL_USABLE - cfg.reserveUntil));
  return Math.ceil(cfg.reserveStart * frac);
}

// The live bid from one engine result: the paid-tile fair bid when there is
// one, else v / lambda (`dec`); capped at v / lambda for the first
// PAID_BID_WARMUP auctions seen and at v / EARLY_LAMBDA_MIN early on.
// Shared by the live client and the sandbox's native seat.
export function composeBid(
  dec: BidDecision,
  v: number,
  lam: number | undefined,
  fairBid: number | null,
  seen: number, // auctions seen this match, the open one included
  usableLeft: number,
  balance: number,
  cfg: ComposeCfg,
  letter?: number, // 0-25
): BidDecision {
  const paid = cfg.nativePaidBid && fairBid !== null ? Math.floor(fairBid) : null;
  const warmup = paid !== null && seen <= cfg.paidBidWarmup && dec.amount < paid;
  const base = paid === null ? dec.amount : warmup ? dec.amount : paid;
  const allin = usableLeft < cfg.allinK;
  const boost = allin ? 1 : (letter !== undefined && HIGH_LETTERS.has(letter) ? cfg.highLetterBoost : 1) * (inQ3(usableLeft) ? cfg.q3Boost : 1);
  const requested = boost === 1 ? base : Math.floor(base * boost);
  const early = seen <= cfg.earlyLambdaAuctions && lam !== undefined && lam < cfg.earlyLambdaMin && !(usableLeft < cfg.allinK);
  const earlyCap = early ? Math.floor(v / cfg.earlyLambdaMin) : Infinity;
  const sizeCap = cfg.maxBidFraction < 1 && !(usableLeft < cfg.allinK)
    ? Math.max(cfg.maxBidFloor, Math.floor(cfg.maxBidFraction * balance)) : Infinity;
  const reserve = usableLeft < cfg.allinK ? 0 : cashReserve(usableLeft, cfg);
  const reserveCap = reserve > 0 ? balance - reserve : Infinity;
  const amount = applySafeCap(Math.max(0, Math.min(balance, requested, earlyCap, sizeCap, reserveCap)), cfg);
  const reason = (paid === null ? dec.reason : warmup ? `paid-tile capped at v/lambda (warmup; ${dec.reason})` : "native paid-tile estimate") +
    (boost !== 1 ? `; boost x${boost.toFixed(2)}` : "") +
    (requested > earlyCap ? `; early lambda floor ${cfg.earlyLambdaMin}` : "") +
    (Math.min(requested, earlyCap) > sizeCap ? `; size cap ${sizeCap}` : "") +
    (Math.min(requested, earlyCap, sizeCap) > reserveCap ? `; reserve ${reserve}` : "");
  return { amount, reason };
}

// Paced lambda: coins are worth only the score they buy, so pick the price of
// a coin that spends the whole balance by the end. Rank the letters still to
// come by value per coin (v / expected clearing price), buy from the best
// down, and stop when the expected spend over the remaining auctions equals
// the balance; lambda is the ratio at the cutoff. A letter at the cutoff bids
// exactly its clearing price, better ones bid above it, worse ones below.
export function pacedLambda(
  v: ArrayLike<number>,
  U: ArrayLike<number>,
  balance: number,
  slots: number, // auctions we can still bid in, the open one included
  prices: ArrayLike<number>,
): number {
  let total = 0;
  for (let i = 0; i < 26; i++) total += U[i];
  if (total <= 0 || balance <= 0) return 1;
  const rows: { ratio: number; spend: number }[] = [];
  for (let i = 0; i < 26; i++) {
    if (U[i] <= 0 || !(v[i] > 0)) continue;
    const price = Math.max(1, prices[i]);
    rows.push({ ratio: v[i] / price, spend: slots * (U[i] / total) * price });
  }
  rows.sort((a, b) => b.ratio - a.ratio);
  let spent = 0;
  let lam = 0.05;
  for (const r of rows) {
    spent += r.spend;
    lam = r.ratio;
    if (spent >= balance) break;
  }
  return Math.max(0.05, Math.min(1, lam));
}

// Cash-in: the gain of playing a word now versus holding it.
export function cashinGain(
  reward: number,
  usableLeft: number,
  holdDelta: number, // E[B((rack - w) + future)] - E[B(rack + future)]
  cfg: Pick<Config, "moneyWeight" | "lambdaMode" | "allinK">,
  lam?: number,
): number {
  // Cash funds bids; once no usable auction is left to spend it on it is
  // worth nothing.
  const money = usableLeft > 0 ? cfg.moneyWeight * (lam ?? lambdaFor(usableLeft, cfg)) * reward : 0;
  return reward + money + holdDelta;
}

// With this few usable auctions left, play the whole plan rather than risk it.
export function shouldDumpPlan(usableLeft: number, isFinal: boolean, cfg: Pick<Config, "endgameK">): boolean {
  return isFinal || usableLeft <= cfg.endgameK;
}
