// Env-driven knobs. Everything has a default so `npm start` works unconfigured.

function num(name: string, def: number): number {
  const raw = process.env[name];
  if (raw === undefined || raw === "") return def;
  const n = Number(raw);
  return Number.isFinite(n) ? n : def;
}

function flag(name: string): boolean {
  const raw = process.env[name];
  return raw === "1" || raw === "true";
}

export type LambdaMode = "flat_allin" | "linear" | "paced";

export interface Config {
  dryRun: boolean;
  safeMaxBid: number | null;
  lambdaMode: LambdaMode;
  allinK: number;
  allinMinV: number;
  nRollouts: number;
  horizon: number;
  exactCap: number;
  cashinK: number;
  moneyWeight: number;
  realOppBaseBonus: number;
  realOppSlope: number;
  realOppJitter: number;
  seed: number;
  // Wall-clock cap on one full-table refinement, so a slow machine still
  // produces a table between auctions.
  rolloutBudgetMs: number;
  fastBudgetMs: number;
  // Pseudo-count of the previous table when blending in a new estimate:
  // v = (n * new + k * prev) / (n + k). Damps low-sample updates; 0 disables.
  vShrink: number;
  // Fraction of the way the table moves toward each new estimate. The
  // rollouts bid using the previous table, so an undamped update oscillates.
  vDamp: number;
  // Re-evaluate cash-in at least every this many auctions even if the rack
  // is unchanged (the value of holding falls as the horizon shrinks).
  cashinEvery: number;
  // Play the whole plan once usableLeft is at or below this.
  endgameK: number;
  logDir: string;
  // Native rollout engine (native/engine.cpp); falls back to JS rollouts.
  native: boolean;
  nativeThreads: number;
  nativeExactCap: number;
  nativeScenarios: number;
  nativeFast: number;
  nativeRefine: number;
  nativeRefineMs: number;
  nativeBidSearch: "coarse" | "adaptive" | "exhaustive";
  // A native cash-in must beat holding by at least this many points.
  cashMargin: number;
  // Words of 4 letters or fewer must beat holding by at least this much
  // (and CASH_MARGIN): HO was cashed in at gain 0.57 in 16347. 0 disables.
  shortCashMargin: number;
  // Bid the native paid-tile fair bid when one is available; otherwise
  // (NATIVE_PAID_BID=0) bid v / lambda from the same engine result.
  nativePaidBid: boolean;
  // For the first this-many auctions seen in a match, cap the paid-tile bid at v / lambda.
  // The engine's table is fresh then and its paid arms undervalue coins
  // (live: fair bid 2.6x v/lambda on auction 1, at or below it afterwards).
  paidBidWarmup: number;
  // Over the first earlyLambdaAuctions auctions seen, bid at most
  // v / max(lambda, earlyLambdaMin): we spend ~78 coins in the first quarter
  // versus 24-46 for the leading bots. 0 disables.
  earlyLambdaAuctions: number;
  earlyLambdaMin: number;
  // Outside the all-in window, bid at most max(MAX_BID_FLOOR,
  // MAX_BID_FRACTION * balance): one letter bought with most of the balance
  // (Z for 43 of 72 in match 16132) starved the rest of the match. 1 disables.
  maxBidFraction: number;
  maxBidFloor: number;
  // Keep a cash reserve that tapers linearly from RESERVE_START coins at the
  // first auction to 0 at RESERVE_UNTIL usable auctions left: candidate 2
  // still went broke mid-match (spend 79/0/0/92 by quarter in 16311). 0 disables.
  reserveStart: number;
  reserveUntil: number;
  // Bid multipliers outside the all-in window: J/K/Q/X/Z (we play 0.63 a match
  // against 1.35-1.43 for the top two), and the third quarter (we spend 25
  // there against 50-61 for most leaders). 1 disables.
  highLetterBoost: number;
  q3Boost: number;
}

export function loadConfig(): Config {
  const m = process.env.LAMBDA_MODE;
  const mode: LambdaMode = m === "linear" || m === "flat_allin" ? m : "paced";
  const safe = num("SAFE_MAX_BID", -1);
  return {
    dryRun: flag("DRY_RUN"),
    safeMaxBid: safe >= 0 ? safe : null,
    lambdaMode: mode,
    allinK: num("ALLIN_K", 5),
    allinMinV: num("ALLIN_MIN_V", 1),
    nRollouts: num("N_ROLLOUTS", 400),
    horizon: num("HORIZON", 15),
    exactCap: num("EXACT_CAP", 14),
    cashinK: num("CASHIN_K", 40),
    moneyWeight: num("MONEY_WEIGHT", 0.5),
    // Fitted to ~200 live auctions: clearing price ~ 8 + 2.5 * face value.
    realOppBaseBonus: num("REAL_OPP_BASE_BONUS", 8),
    realOppSlope: num("REAL_OPP_SLOPE", 2.5),
    realOppJitter: num("REAL_OPP_JITTER", 4),
    seed: num("SEED", (Date.now() & 0x7fffffff) >>> 0),
    rolloutBudgetMs: num("ROLLOUT_BUDGET_MS", 700),
    fastBudgetMs: num("FAST_BUDGET_MS", 150),
    vShrink: num("V_SHRINK", 30),
    vDamp: num("V_DAMP", 0.5),
    cashinEvery: num("CASHIN_EVERY", 3),
    endgameK: num("ENDGAME_K", 2),
    logDir: process.env.LOG_DIR ?? "logs",
    native: process.env.NATIVE !== "0" && process.env.NATIVE !== "false",
    nativeThreads: num("NATIVE_THREADS", 8),
    nativeExactCap: num("NATIVE_EXACT_CAP", 16),
    nativeScenarios: num("NATIVE_SCENARIOS", 2000),
    nativeFast: num("NATIVE_FAST", 300),
    nativeRefine: num("NATIVE_REFINE", 3000),
    nativeRefineMs: num("NATIVE_REFINE_MS", 75),
    nativeBidSearch: process.env.NATIVE_BID_SEARCH === "exhaustive" ? "exhaustive"
      : process.env.NATIVE_BID_SEARCH === "adaptive" ? "adaptive" : "coarse",
    cashMargin: num("CASH_MARGIN", 0.3),
    shortCashMargin: Math.max(0, num("SHORT_CASH_MARGIN", 0)),
    nativePaidBid: process.env.NATIVE_PAID_BID !== "0" && process.env.NATIVE_PAID_BID !== "false",
    paidBidWarmup: Math.max(0, Math.floor(num("PAID_BID_WARMUP", 0))),
    earlyLambdaAuctions: Math.max(0, Math.floor(num("EARLY_LAMBDA_AUCTIONS", 0))),
    earlyLambdaMin: Math.max(0, Math.min(1, num("EARLY_LAMBDA_MIN", 0))),
    maxBidFraction: Math.max(0, Math.min(1, num("MAX_BID_FRACTION", 1))),
    maxBidFloor: Math.max(0, num("MAX_BID_FLOOR", 20)),
    reserveStart: Math.max(0, num("RESERVE_START", 0)),
    reserveUntil: Math.max(0, num("RESERVE_UNTIL", 10)),
    highLetterBoost: Math.max(0, num("HIGH_LETTER_BOOST", 1)),
    q3Boost: Math.max(0, num("Q3_BOOST", 1)),
  };
}
