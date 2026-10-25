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
  earlyBidCap: number;
  earlyBidAuctions: number;
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
}

export function loadConfig(): Config {
  const m = process.env.LAMBDA_MODE;
  const mode: LambdaMode = m === "linear" || m === "flat_allin" ? m : "paced";
  const safe = num("SAFE_MAX_BID", -1);
  return {
    dryRun: flag("DRY_RUN"),
    safeMaxBid: safe >= 0 ? safe : null,
    earlyBidCap: Math.max(0, num("EARLY_BID_CAP", 35)),
    earlyBidAuctions: Math.max(0, Math.floor(num("EARLY_BID_AUCTIONS", 10))),
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
  };
}
