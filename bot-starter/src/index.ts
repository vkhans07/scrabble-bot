// ScrabbleBot v1.
//
// First run, with a token from the website:
//   BOT_NAME=<name> BOT_TOKEN=<token> npm start
// (or BOT_NONCE=<code> from the CLI). The token is saved to .token-<BOT_NAME>,
// so later runs need only `BOT_NAME=<name> npm start`.
//
// Entering a tournament: `TOURNAMENT_ID=<id>` registers on connect.
//
// The main thread mirrors the tables, bids from the latest cached value table
// and plays words; a worker thread (engine.worker.ts) does the rollouts. See
// config.ts for every env knob (DRY_RUN, SAFE_MAX_BID, LAMBDA_MODE, ...).

import {
  DbConnection,
  type EventContext,
  type ErrorContext,
} from "./module_bindings/index.js";
import { Identity, SenderError } from "spacetimedb";
import * as fs from "node:fs";
import * as path from "node:path";
import { Worker } from "node:worker_threads";
import { WordReservations, shouldReplaceBid, BidTiming } from "./actions.js";
import { loadConfig } from "./config.js";
import { A, Dict, LETTER_VALUE } from "./dict.js";
import { closeMatchLog, initLog, logEvent } from "./log.js";
import { bidAmount, composeBid, pacedLambda, shouldDumpPlan } from "./policy.js";
import { RackEval } from "./rackeval.js";
import { makeRng } from "./rng.js";
import { expectedPrices, firstPriceBid } from "./simmodel.js";
import {
  Store,
  buildSnapshot,
  derive,
  type ArchiveRow,
  type AuctionRow,
  type Derived,
  type PartRow,
  type RackRow,
  rackCounts,
} from "./state.js";

const cfg = loadConfig();
initLog(cfg.logDir);

const HOST = process.env.STDB_HOST ?? "https://maincloud.spacetimedb.com";
const DB_NAME = process.env.STDB_DB ?? "scrabblebot";
const BOT_NAME = process.env.BOT_NAME ?? "bot";
const BOT_TOKEN = process.env.BOT_TOKEN; // from the website, first run only
const BOT_NONCE = process.env.BOT_NONCE; // from the CLI, first run only
const TOURNAMENT_ID = process.env.TOURNAMENT_ID;
const TOKEN_PATH = path.join(process.cwd(), `.token-${BOT_NAME}`);
const bidTiming = new BidTiming();

const tag = `[${BOT_NAME}]`;
const say = (...a: unknown[]) => console.log(tag, ...a);

function loadToken(): string | undefined {
  try {
    return fs.readFileSync(TOKEN_PATH, "utf8").trim() || undefined;
  } catch {
    return undefined;
  }
}
function saveToken(tok: string) {
  try {
    fs.writeFileSync(TOKEN_PATH, tok, { mode: 0o600 });
  } catch (e) {
    console.warn(tag, "couldn't save token:", e);
  }
}

// A reducer call fails when the module refuses it, often for an expected
// reason (a word played just after its match ended, a bid that missed its
// window). Log it and keep playing; nothing here may kill the process.
function logFailure(what: string, matchId: bigint | null = null) {
  return (err: unknown) => {
    const message = err instanceof Error ? err.message : String(err);
    console.warn(tag, `${what} failed: ${message}`);
    logEvent(matchId, "reducer_error", { what, message });
  };
}

process.on("uncaughtException", (e) => {
  console.error(tag, "uncaughtException:", e);
  logEvent(null, "uncaught", { message: String(e instanceof Error ? e.stack : e) });
});
process.on("unhandledRejection", (e) => {
  console.error(tag, "unhandledRejection:", e);
  logEvent(null, "unhandled_rejection", { message: String(e instanceof Error ? e.stack : e) });
});

// Wrap a callback so anything it throws is logged, not propagated into the
// SDK's dispatch loop.
function safe<Args extends unknown[]>(label: string, fn: (...a: Args) => void) {
  return (...a: Args) => {
    try {
      fn(...a);
    } catch (e) {
      console.error(tag, `${label} threw:`, e);
      logEvent(null, "callback_error", { label, message: String(e instanceof Error ? e.stack : e) });
    }
  };
}

// ---- Shared state ----------------------------------------------------------

const dict = Dict.load();
const mainEv = new RackEval(dict, cfg.exactCap);
const mainRng = makeRng((cfg.seed ^ 0x1234567) >>> 0);
const store = new Store();

let conn: DbConnection | null = null;
let myIdentity: Identity | null = null;
let myBotId: bigint | null = null;

interface MatchCtx {
  id: bigint;
  v: Float64Array | null; // latest value table from the worker
  lam: number | null; // price of a coin from the native rollouts
  vSeq: number;
  nativeBid: { auctionId: string; rackKey: string; amount: number; seq: number } | null;
  auctionSeen: Set<bigint>;
  lastBid: Map<bigint, number>;
  deadlines: Map<bigint, number>; // monotonic auction deadlines
  rackKey: string;
  lastCashinKey: string;
  auctionsSinceCashin: number;
  cashRequests: Map<number, number>;
  submissions: WordReservations;
  finalDumped: boolean;
  mismatchSeen: Set<string>;
  ended: boolean;
}

const matches = new Map<bigint, MatchCtx>();
const reservations = new Map<bigint, WordReservations>();
const journalPath = `${TOKEN_PATH}.pending.json`;
// An interrupted process cannot know whether a non-idempotent call committed.
const quarantined = new Set<string>(fs.existsSync(journalPath)
  ? JSON.parse(fs.readFileSync(journalPath, "utf8")) as string[] : []);
function reservationsFor(id: bigint) {
  let q = reservations.get(id);
  if (!q) {
    q = new WordReservations();
    q.uncertain = quarantined.has(String(id));
    reservations.set(id, q);
  }
  return q;
}
function savePending() {
  const ids = new Set(quarantined);
  for (const [id, q] of reservations) if (q.uncertain || q.pending.size) ids.add(String(id));
  fs.writeFileSync(`${journalPath}.tmp`, JSON.stringify([...ids]), { mode: 0o600 });
  fs.renameSync(`${journalPath}.tmp`, journalPath);
}
const maxMatches = Number(process.env.EVAL_MATCHES ?? 0);
let completedMatches = 0;
let stopping = false;
function stop() {
  if (stopping) return;
  stopping = true;
  logEvent(null, "stopping", { completedMatches });
  conn?.disconnect();
  worker?.postMessage({ type: "shutdown" });
  setTimeout(() => void worker?.terminate(), 200);
  setTimeout(() => process.exit(0), 300);
}
process.on("SIGINT", stop);
process.on("SIGTERM", stop);

function ctxFor(id: bigint): MatchCtx {
  let m = matches.get(id);
  if (!m) {
    m = {
      id,
      v: null,
      lam: null,
      vSeq: 0,
      nativeBid: null,
      auctionSeen: new Set(),
      lastBid: new Map(),
      deadlines: new Map(),
      rackKey: "",
      lastCashinKey: "\0",
      auctionsSinceCashin: 99,
      cashRequests: new Map(),
      submissions: reservationsFor(id),
      finalDumped: false,
      mismatchSeen: new Set(),
      ended: false,
    };
    matches.set(id, m);
  }
  return m;
}

// ---- Worker ----------------------------------------------------------------

let worker: Worker | null = null;

function startWorker() {
  try {
    const w = new Worker(new URL("./engine.worker.ts", import.meta.url), {
      workerData: cfg,
      execArgv: process.execArgv,
    });
    worker = w;
    w.on("message", safe("worker message", onWorkerMessage));
    w.on("error", (e) => {
      console.error(tag, "worker error:", e);
      logEvent(null, "worker_error", { message: String(e instanceof Error ? e.stack : e) });
    });
    w.on("exit", (code) => {
      if (worker === w) worker = null;
      logEvent(null, "worker_exit", { code });
      console.warn(tag, `worker exited (${code}); restarting`);
      if (!stopping) setTimeout(startWorker, 1000);
    });
  } catch (e) {
    console.error(tag, "couldn't start worker:", e);
    setTimeout(startWorker, 3000);
  }
}

interface WorkerResult {
  type: string;
  phase?: string;
  seq: number;
  matchId: string;
  auctionId?: string;
  rackKey?: string;
  v?: Float64Array;
  lam?: number;
  fairBid?: number;
  computed?: number[];
  nScen?: number;
  ms?: number;
  totalMs?: number;
  cashin?: { word: string; reward: number; delta: number; gain: number }[];
  perf?: Record<string, number>;
  message?: string;
}

function onWorkerMessage(m: WorkerResult) {
  if (m.type === "ready") {
    say("engine ready");
    // Re-send state for any live match (after a worker restart).
    for (const mc of matches.values()) if (!mc.ended) evaluate(mc, false);
    return;
  }
  if (m.type === "native_log") {
    say(`native: ${m.message}`);
    logEvent(null, "native", { message: m.message });
    return;
  }
  if (m.type === "error") {
    console.warn(tag, "engine error:", m.message);
    logEvent(null, "engine_error", { message: m.message });
    return;
  }
  if (m.type !== "result" || m.phase === "skip") return;
  const mc = matches.get(BigInt(m.matchId));
  if (!mc || mc.ended || !m.v) return;
  if (m.seq < mc.vSeq) return;
  mc.vSeq = m.seq;
  mc.v = m.v;
  mc.lam = m.lam ?? null;
  if (m.fairBid !== undefined && m.fairBid >= 0 && m.auctionId && m.rackKey) {
    mc.nativeBid = { auctionId: m.auctionId, rackKey: m.rackKey, amount: m.fairBid, seq: m.seq };
  }
  logEvent(mc.id, "engine", {
    phase: m.phase,
    seq: m.seq,
    auctionId: m.auctionId,
    nScen: m.nScen,
    ms: m.ms,
    totalMs: m.totalMs,
    perf: m.perf,
    fairBid: m.fairBid ?? null,
    cashin: m.cashin,
  });
  evaluate(mc, true);
  if (m.phase === "full" && m.rackKey === mc.rackKey && mc.cashRequests.has(m.seq)) {
    mc.lastCashinKey = m.rackKey;
    mc.auctionsSinceCashin = Math.max(0, mc.auctionsSinceCashin - mc.cashRequests.get(m.seq)!);
    mc.cashRequests.delete(m.seq);
    applyCashin(mc, m);
  }
}

function postSnapshot(mc: MatchCtx, d: Derived, rackKey: string) {
  if (!worker || !myBotId) return;
  const need = rackKey !== mc.lastCashinKey || mc.auctionsSinceCashin >= cfg.cashinEvery;
  const snap = buildSnapshot(store, d, myBotId, need, rackKey);
  try {
    worker.postMessage({ type: "snapshot", snap });
    if (need) mc.cashRequests.set(snap.seq, mc.auctionsSinceCashin);
    if (mc.cashRequests.size > 64) mc.cashRequests.delete(mc.cashRequests.keys().next().value!);
    logEvent(mc.id, "engine_request", { seq: snap.seq, needCashin: need, auctionId: snap.auctionId });
  } catch (e) {
    logFailure("posting snapshot", mc.id)(e);
  }
}

// ---- Decisions -------------------------------------------------------------

function bidStep(mc: MatchCtx, d: Derived, a: AuctionRow, rackKey: string, recvAt: number) {
  const L = a.letter.charCodeAt(0) - A;
  const balance = Number(d.me.balance);
  let v: number;
  let src: string;
  if (mc.v) {
    v = mc.v[L];
    src = "table";
  } else {
    // Cold start or no engine yet: myopic marginal value, floored at face+1.
    const r = d.rack.slice();
    const before = mainEv.best(r);
    r[L]++;
    v = Math.max(mainEv.best(r) - before, LETTER_VALUE[L] + 1);
    src = "cold";
  }
  const opps = buildSnapshot(store, d, myBotId!, false, rackKey).opps;
  const fp =
    d.match.auctionType.tag === "FirstPrice"
      ? (value: number) => firstPriceBid(opps, L, value, balance, cfg, mainRng)
      : undefined;
  // Paced mode needs the whole table; before the first one arrives, fall
  // back to lambda = 1.
  const lam =
    mc.lam !== null && mc.v
      ? mc.lam
      : cfg.lambdaMode === "paced" && mc.v
      ? pacedLambda(mc.v, d.belief.U, balance, d.usableLeft + 1, expectedPrices(opps, cfg))
      : undefined;
  const dec = bidAmount(v, balance, d.usableLeft, cfg, fp, lam);
  const paidBid = cfg.nativePaidBid && mc.nativeBid?.auctionId === String(a.id) && mc.nativeBid.rackKey === rackKey
    ? mc.nativeBid : null;
  const composed = composeBid(dec, v, lam, paidBid ? paidBid.amount : null, mc.auctionSeen.size, d.usableLeft, balance, cfg);
  const amount = composed.amount;
  const bidReason = composed.reason;
  const prev = mc.lastBid.get(a.id);
  const left = (mc.deadlines.get(a.id) ?? -Infinity) - performance.now();
  // Send the latest useful bid while the estimated auction is still open.
  // RTT predicts delivery risk; it does not justify discarding an initial bid
  // or a better replacement that still has a chance of acceptance.
  const guardMs = 0;
  const deliveryBudgetMs = bidTiming.guardMs(performance.now());

  if (!shouldReplaceBid(prev, amount, left, guardMs)) {
    if (prev === undefined && amount <= 0) {
      mc.lastBid.set(a.id, 0);
      logEvent(mc.id, "skip_bid", { auctionId: a.id, reason: dec.reason });
    } else if (left <= guardMs && amount !== prev) {
      logEvent(mc.id, "late_action_skipped", { action: "bid", auctionId: a.id, amount, timeLeftMs: left, guardMs });
    }
    return;
  }

  mc.lastBid.set(a.id, amount);
  const latency = performance.now() - recvAt;
  logEvent(mc.id, "bid", {
    auctionId: a.id,
    letter: a.letter,
    amount,
    prev: prev ?? null,
    v,
    src,
    reason: bidReason,
    nativeFairBid: paidBid?.amount ?? null,
    lambda: lam ?? null,
    balance,
    usableLeft: d.usableLeft,
    timeLeftMs: left,
    guardMs,
    deliveryBudgetMs,
    decisionMs: latency,
    dryRun: cfg.dryRun,
    vTable: prev === undefined ? Array.from(mc.v ?? []) : undefined,
  });
  if (cfg.dryRun) {
    say(`DRY: would bid ${amount} on '${a.letter}' (${bidReason}, ${src})`);
    return;
  }
  const sent = performance.now();
  conn!.reducers
    .submitBid({ auctionId: a.id, amount: BigInt(amount) })
    .then(() => {
      const roundTripMs = performance.now() - sent;
      bidTiming.observe(roundTripMs, performance.now());
      logEvent(mc.id, "bid_ack", { auctionId: a.id, amount, roundTripMs });
    })
    .catch((err: unknown) => {
      bidTiming.observe(performance.now() - sent, performance.now());
      logFailure(`bid on auction ${a.id}`, mc.id)(err);
      logEvent(mc.id, "bid_failed", { auctionId: a.id, amount, roundTripMs: performance.now() - sent });
      if (mc.lastBid.get(a.id) === amount) mc.lastBid.delete(a.id);
    });
  say(
    `${prev === undefined ? "bid" : "rebid"} ${amount} on '${a.letter}' ` +
      `(match ${mc.id}, auction ${a.id}, ${dec.reason}, ${latency.toFixed(1)}ms)`,
  );
}

function submitWords(mc: MatchCtx, rackKey: string, words: string[], why: string) {
  const batch = mc.submissions.reserve(rackKey, words);
  if (!batch.length) return;
  // Persist before sending, so a crash cannot cause blind replay on restart.
  savePending();
  for (const { id, word: w } of batch) {
    logEvent(mc.id, "word", { requestId: id, word: w, why, dryRun: cfg.dryRun });
    if (cfg.dryRun) continue;
    const sent = performance.now();
    conn!.reducers.submitWord({ matchId: mc.id, word: w })
      .then(() => logEvent(mc.id, "word_ack", { requestId: id, word: w, roundTripMs: performance.now() - sent }))
      .catch((err: unknown) => {
        // Only a server SenderError establishes that the word was rejected.
        if (err instanceof SenderError) mc.submissions.reject(id);
        else mc.submissions.uncertain = true;
        savePending();
        logEvent(mc.id, "word_failed", { requestId: id, word: w, confirmed: err instanceof SenderError, roundTripMs: performance.now() - sent });
        logFailure(`playing '${w}'`, mc.id)(err);
      });
    say(`match ${mc.id}: playing '${w}' (${why})`);
  }
}

function dumpPlan(mc: MatchCtx, d: Derived, rackKey: string, why: string) {
  if (RackEval.total(d.rack) < 2) return;
  const plan = mainEv.plan(d.rack, 22, 400_000);
  logEvent(mc.id, "plan", { why, value: plan.value, words: plan.words, exact: plan.exact });
  submitWords(mc, rackKey, plan.words, why);
}

function applyCashin(mc: MatchCtx, m: WorkerResult) {
  if (!myBotId || !m.cashin || m.cashin.length === 0) return;
  const d = derive(store, mc.id, myBotId);
  if (!d || d.match.status.tag !== "Running" || d.isFinal) return;
  const rackKey = RackEval.keyOf(d.rack);
  if (rackKey !== m.rackKey) return; // state moved on; a fresher result is coming
  if (shouldDumpPlan(d.usableLeft, false, cfg)) return; // the endgame rule owns this
  const best = m.cashin[0];
  if (best.gain <= (m.perf?.native ? cfg.cashMargin : 0)) return;
  submitWords(mc, rackKey, [best.word], `cash-in gain=${best.gain.toFixed(2)} reward=${best.reward}`);
}

// Re-derive state and act: bid (or rebid), dump words at the end, and tell the
// worker. `fromEngine` skips the snapshot so results don't trigger new work.
function evaluate(mc: MatchCtx, fromEngine: boolean) {
  if (mc.ended || !myBotId || !conn) return;
  const recvAt = performance.now();
  const d = derive(store, mc.id, myBotId);
  if (!d || d.match.status.tag !== "Running") return;

  for (const msg of d.belief.mismatches) {
    if (mc.mismatchSeen.has(msg)) continue;
    mc.mismatchSeen.add(msg);
    console.warn(tag, `belief mismatch in match ${mc.id}: ${msg}`);
    logEvent(mc.id, "belief_mismatch", { message: msg });
  }

  const rackKey = RackEval.keyOf(d.rack);
  const rackChanged = rackKey !== mc.rackKey;
  mc.rackKey = rackKey;

  const a = d.openAuction;
  let isNew = false;
  if (a && !mc.auctionSeen.has(a.id)) {
    mc.auctionSeen.add(a.id);
    mc.auctionsSinceCashin++;
    isNew = true;
    mc.deadlines.set(a.id, bidTiming.deadline(
      Number(a.opensAt.microsSinceUnixEpoch / 1000n),
      Number(a.closesAt.microsSinceUnixEpoch / 1000n),
      Date.now(), performance.now(),
    ));
    logEvent(mc.id, "auction_open", {
      auctionId: a.id,
      letter: a.letter,
      bagTotal: d.match.bagTotal,
      usableLeft: d.usableLeft,
      final: d.isFinal,
      balance: d.me.balance,
      rack: rackKey,
      U: d.belief.U,
      vTable: mc.v ? Array.from(mc.v) : null,
    });
  }

  if (a) {
    if (d.isFinal) {
      // Never bid on the final auction; its window is for the word dump.
      if (!mc.finalDumped) {
        mc.finalDumped = true;
        dumpPlan(mc, d, rackKey, "final dump");
        // A win in the previous auction can reach the rack view a moment
        // after the auction row; look again once it has surely landed.
        setTimeout(
          safe("final redump", () => {
            const d2 = myBotId ? derive(store, mc.id, myBotId) : null;
            if (d2 && d2.match.status.tag === "Running")
              dumpPlan(mc, d2, RackEval.keyOf(d2.rack), "final dump (recheck)");
          }),
          250,
        );
      }
    } else {
      bidStep(mc, d, a, rackKey, recvAt);
    }
  }

  if (!d.isFinal && shouldDumpPlan(d.usableLeft, false, cfg) && (rackChanged || isNew)) {
    dumpPlan(mc, d, rackKey, `endgame (usableLeft=${d.usableLeft})`);
  }

  if (!fromEngine) postSnapshot(mc, d, rackKey);
}

// ---- Table mirroring -------------------------------------------------------

// Callbacks for one transaction fire back to back; evaluate once afterwards.
const dirty = new Set<bigint>();
function markDirty(matchId: bigint) {
  if (dirty.size === 0)
    queueMicrotask(
      safe("evaluate", () => {
        const ids = [...dirty];
        dirty.clear();
        for (const id of ids) {
          const mc = matches.get(id);
          if (mc && !mc.ended) evaluate(mc, false);
        }
      }),
    );
  dirty.add(matchId);
}

function isMine(matchId: bigint): boolean {
  if (myBotId === null) return false;
  for (const p of store.match(matchId).parts.values()) if (p.botId === myBotId) return true;
  return false;
}

// A match worth mirroring: one still running, or one we are already tracking
// (so the rows of its final transaction still land).
function isLive(matchId: bigint): boolean {
  if (matches.has(matchId)) return true;
  return conn?.db.matchState.id.find(matchId)?.status.tag === "Running";
}

// Log only matches this bot plays in; other live matches are just mirrored.
function logMine(matchId: bigint, type: string, data: Record<string, unknown>) {
  if (isMine(matchId)) logEvent(matchId, type, data);
}

function track(matchId: bigint) {
  // Only live matches get a context; history from earlier matches is never
  // acted on.
  if (!isLive(matchId) || !isMine(matchId)) return;
  let mc = matches.get(matchId);
  if (!mc) {
    mc = ctxFor(matchId);
    // Rows that arrived before we knew the match was ours were not logged.
    const d = store.match(matchId);
    logEvent(matchId, "match_snapshot", {
      match: store.matches.get(matchId),
      participants: [...d.parts.values()],
      auctions: [...d.auctions.values()],
      results: [...d.results.values()],
      plays: [...d.plays.values()],
      rack: [...d.rack.values()],
    });
  }
  if (!mc.ended) markDirty(matchId);
}

function wireTables(c: DbConnection) {
  const db = c.db;

  db.bot.onInsert(safe("bot+", (_c: EventContext, r) => void store.bots.set(r.id, r)));
  db.bot.onUpdate(safe("bot~", (_c: EventContext, _o, r) => void store.bots.set(r.id, r)));

  db.matchState.onInsert(
    safe("match+", (_c: EventContext, r) => {
      store.matches.set(r.id, r);
      logMine(r.id, "match_state", { op: "insert", row: r });
      track(r.id);
    }),
  );
  db.matchState.onUpdate(
    safe("match~", (_c: EventContext, old, r) => {
      store.matches.set(r.id, r);
      logMine(r.id, "match_state", { op: "update", row: r });
      track(r.id);
      if (old.status.tag !== "Ended" && r.status.tag === "Ended")
        queueMicrotask(safe("match end", () => onMatchEnded(r.id)));
    }),
  );

  const putPart = (op: string, r: PartRow) => {
    if (!isLive(r.matchId)) return;
    store.match(r.matchId).parts.set(r.id, r);
    logMine(r.matchId, "match_participant", { op, row: r });
    track(r.matchId);
  };
  db.matchParticipant.onInsert(safe("part+", (_c: EventContext, r) => putPart("insert", r)));
  db.matchParticipant.onUpdate(safe("part~", (_c: EventContext, _o, r) => putPart("update", r)));

  const putAuction = (op: string, r: AuctionRow) => {
    if (!isLive(r.matchId)) return;
    store.match(r.matchId).auctions.set(r.id, r);
    logMine(r.matchId, "auction", { op, row: r });
    track(r.matchId);
  };
  db.auction.onInsert(safe("auction+", (_c: EventContext, r) => putAuction("insert", r)));
  db.auction.onUpdate(safe("auction~", (_c: EventContext, _o, r) => putAuction("update", r)));

  db.auctionResult.onInsert(
    safe("result+", (_c: EventContext, r) => {
      if (!isLive(r.matchId)) return;
      store.match(r.matchId).results.set(r.auctionId, r);
      logMine(r.matchId, "auction_result", { row: r });
      track(r.matchId);
    }),
  );
  db.wordPlay.onInsert(
    safe("play+", (_c: EventContext, r) => {
      if (!isLive(r.matchId)) return;
      const fresh = !store.match(r.matchId).plays.has(r.id);
      store.match(r.matchId).plays.set(r.id, r);
      if (fresh && r.botId === myBotId) {
        reservations.get(r.matchId)?.confirm(r.word);
        savePending();
      }
      logMine(r.matchId, "word_play", { row: r });
      track(r.matchId);
    }),
  );

  const putRack = (op: string, r: RackRow) => {
    if (!isLive(r.matchId)) return;
    store.match(r.matchId).rack.set(r.id, r);
    logMine(r.matchId, "my_rack", { op, row: r });
    track(r.matchId);
  };
  db.myRack.onInsert(safe("rack+", (_c: EventContext, r) => putRack("insert", r)));
  db.myRack.onUpdate(safe("rack~", (_c: EventContext, _o, r) => putRack("update", r)));
  db.myRack.onDelete(
    safe("rack-", (_c: EventContext, r) => {
      if (!isLive(r.matchId)) return;
      store.match(r.matchId).rack.delete(r.id);
      logMine(r.matchId, "my_rack", { op: "delete", row: r });
      track(r.matchId);
    }),
  );

  db.botCredential.onInsert(
    safe("cred+", () => {
      if (myBotId === null) myBotId = resolveMyBotId(c);
    }),
  );
}

// Seed the mirror from the subscription's initial rows. Idempotent: the
// insert callbacks may have delivered the same rows already.
function primeStore(c: DbConnection) {
  const db = c.db;
  for (const r of db.bot.iter()) store.bots.set(r.id, r);
  const running = new Set<bigint>();
  for (const r of db.matchState.iter()) {
    store.matches.set(r.id, r);
    if (r.status.tag === "Running") running.add(r.id);
  }
  for (const r of db.matchParticipant.iter())
    if (running.has(r.matchId)) store.match(r.matchId).parts.set(r.id, r);
  for (const r of db.auction.iter())
    if (running.has(r.matchId)) store.match(r.matchId).auctions.set(r.id, r);
  for (const r of db.auctionResult.iter())
    if (running.has(r.matchId)) store.match(r.matchId).results.set(r.auctionId, r);
  for (const r of db.wordPlay.iter())
    if (running.has(r.matchId)) store.match(r.matchId).plays.set(r.id, r);
  for (const r of db.myRack.iter())
    if (running.has(r.matchId)) store.match(r.matchId).rack.set(r.id, r);
  for (const id of running) track(id);
}

// ---- Match end --------------------------------------------------------------

function onMatchEnded(matchId: bigint) {
  const mc = matches.get(matchId);
  if (!mc || mc.ended || !myBotId) return;
  mc.ended = true;
  completedMatches++;
  reservations.delete(matchId);
  quarantined.delete(String(matchId));
  savePending();
  const data = store.match(matchId);
  const parts = [...data.parts.values()];
  const meP = parts.find((p) => p.botId === myBotId);
  const myPlays = [...data.plays.values()].filter((p) => p.botId === myBotId);
  const won = [...data.results.values()].filter((r) => r.winnerBotId === myBotId);
  const place = meP ? 1 + parts.filter((p) => p.score > meP.score).length : null;
  const leftoverRack = rackCounts(data, myBotId);
  const leftoverPlan = mainEv.plan(leftoverRack, cfg.exactCap, 50_000);
  const summary = {
    leftoverRack: RackEval.keyOf(leftoverRack),
    leftoverPlayableValue: leftoverPlan.value,
    leftoverValueExact: leftoverPlan.exact,
    finalScore: meP?.score,
    finalBalance: meP?.balance,
    place,
    participants: parts.length,
    auctions: data.results.size, // one result row per auction, the final one included
    auctionsWon: won.length,
    spentOnWins: won.reduce((s, r) => s + r.paid, 0n),
    wordsPlayed: myPlays.length,
    rewardFromWords: myPlays.reduce((s, p) => s + p.totalReward, 0n),
    scores: parts.map((p) => ({ bot: store.bots.get(p.botId)?.name ?? String(p.botId), score: p.score })),
  };
  logEvent(matchId, "match_summary", summary);
  say(
    `match ${matchId} ended: score ${summary.finalScore}, place ${place}/${parts.length}, ` +
      `leftover balance ${summary.finalBalance}, auctions ${summary.auctions}`,
  );

  // Winning bids only become visible after the match; keep them for opponent
  // modelling. The table can lag the match end, so look twice.
  const dumpArchive = (last: boolean) =>
    safe("archive dump", () => {
      const rows: ArchiveRow[] = [];
      if (conn) for (const r of conn.db.auctionTopBidArchive.iter()) if (r.matchId === matchId) rows.push(r);
      logEvent(matchId, "bid_archive", { last, count: rows.length, rows });
      if (last) {
        closeMatchLog(matchId);
        matches.delete(matchId);
        store.forget(matchId);
        if (maxMatches > 0 && completedMatches >= maxMatches) stop();
      }
    });
  setTimeout(dumpArchive(false), 2000);
  setTimeout(dumpArchive(true), 8000);

  joinLobby();
}

// ---- Lobby / connection ------------------------------------------------------

function resolveMyBotId(c: DbConnection): bigint | null {
  if (!myIdentity) return null;
  const cred = c.db.botCredential.identity.find(myIdentity);
  return cred ? cred.botId : null;
}

function inRunningMatch(c: DbConnection): boolean {
  if (myBotId === null) return false;
  for (const p of c.db.matchParticipant.iter()) {
    if (p.botId !== myBotId) continue;
    if (c.db.matchState.id.find(p.matchId)?.status.tag === "Running") return true;
  }
  return false;
}

let lobbyJoinInFlight = false;
let lobbyRetryTimer: NodeJS.Timeout | null = null;
let lobbyRetryDelay = 1000;
function scheduleLobbyRetry(delay = lobbyRetryDelay) {
  if (lobbyRetryTimer || stopping) return;
  lobbyRetryTimer = setTimeout(() => {
    lobbyRetryTimer = null;
    joinLobby();
  }, delay);
  lobbyRetryDelay = Math.min(lobbyRetryDelay * 2, 30000);
}

function joinLobby() {
  const c = conn;
  if (!c || myBotId === null || TOURNAMENT_ID || stopping || (maxMatches > 0 && completedMatches >= maxMatches)) return;
  if (inRunningMatch(c)) {
    lobbyRetryDelay = 1000;
    return;
  }
  const openLobby = Array.from(c.db.lobby.iter()).find((l) => l.status.tag === "Open");
  const alreadyIn =
    openLobby !== undefined &&
    Array.from(c.db.lobbyMember.iter()).some(
      (lm) => lm.lobbyId === openLobby.id && lm.botId === myBotId,
    );
  if (alreadyIn) {
    lobbyRetryDelay = 1000;
    return;
  }
  if (lobbyJoinInFlight) return;
  lobbyJoinInFlight = true;
  say("joining lobby");
  c.reducers.joinLobby({}).then(() => {
    // Reducer success can precede the local subscription update. Recheck once
    // the cache has had a moment to receive the lobby-member insert.
    scheduleLobbyRetry(1200);
  }).catch((err: unknown) => {
    logFailure("joining the lobby")(err);
    scheduleLobbyRetry();
  }).finally(() => {
    lobbyJoinInFlight = false;
  });
}

function bootstrapActivity(c: DbConnection) {
  if (TOURNAMENT_ID) {
    c.reducers
      .registerForTournament({ tournamentId: BigInt(TOURNAMENT_ID) })
      .then(() => say(`registered for tournament ${TOURNAMENT_ID}`))
      .catch(logFailure(`registering for tournament ${TOURNAMENT_ID}`));
  }
  primeStore(c);
  joinLobby();
}

// Once the tables we need are in the cache: find our bot, then get going.
function onSubscribed(c: DbConnection) {
  say("subscription applied");
  myBotId = resolveMyBotId(c);
  if (myBotId === null) {
    if (BOT_NONCE) {
      say("claiming credential with nonce…");
      c.reducers.claimCredential({ code: BOT_NONCE }).catch(logFailure("claiming the credential"));
      setTimeout(
        safe("claim check", () => {
          myBotId = resolveMyBotId(c);
          if (myBotId === null) {
            console.error(tag, "couldn't claim credential. Bad / expired nonce?");
            process.exit(1);
          }
          say(`claimed credential for bot '${c.db.bot.id.find(myBotId)?.name ?? "?"}' (id ${myBotId})`);
          bootstrapActivity(c);
        }),
        1000,
      );
      return;
    }
    console.error(tag, "no BotCredential for this token. Set BOT_TOKEN or BOT_NONCE and re-run.");
    process.exit(1);
  }
  say(`acting as bot '${c.db.bot.id.find(myBotId)?.name ?? "?"}' (id ${myBotId})`);
  bootstrapActivity(c);
}

// The server holds thousands of finished matches (hundreds of thousands of
// bid-archive rows). Mirroring all of it costs >1GB and a callback storm, so
// subscribe to the small tables first, then to the per-match tables from a
// recent match onward. If the server refuses those queries, reconnect with
// subscribeToAllTables (the two can't be mixed on one connection).
let useAllTables = false;

function subscribe(c: DbConnection) {
  const fallback = (what: string) => (_ctx: unknown, err?: unknown) => {
    console.error(tag, `${what} subscription failed:`, err);
    logEvent(null, "subscription_error", { what, message: String(err) });
    useAllTables = true;
    c.disconnect(); // onDisconnect reconnects, now with everything
  };

  if (useAllTables) {
    c.subscriptionBuilder()
      .onApplied(safe("applied", () => onSubscribed(c)))
      .onError(fallback("full"))
      .subscribeToAllTables();
    return;
  }

  c.subscriptionBuilder()
    .onError(fallback("phase 1"))
    .onApplied(
      safe("phase 1 applied", () => {
        let maxId = 0n;
        let minRunning: bigint | null = null;
        for (const m of c.db.matchState.iter()) {
          if (m.id > maxId) maxId = m.id;
          if (m.status.tag === "Running" && (minRunning === null || m.id < minRunning))
            minRunning = m.id;
        }
        let from = maxId > 5n ? maxId - 5n : 0n;
        if (minRunning !== null && minRunning < from) from = minRunning;
        const q = (t: string) => `SELECT * FROM ${t} WHERE match_id >= ${from}`;
        c.subscriptionBuilder()
          .onError(fallback("phase 2"))
          .onApplied(safe("applied", () => onSubscribed(c)))
          .subscribe([
            q("match_participant"),
            q("auction"),
            q("auction_result"),
            q("word_play"),
            q("auction_top_bid_archive"),
          ]);
      }),
    )
    .subscribe([
      "SELECT * FROM bot",
      "SELECT * FROM bot_credential",
      "SELECT * FROM match_state",
      "SELECT * FROM lobby",
      "SELECT * FROM lobby_member",
      "SELECT * FROM my_rack",
    ]);
}

let lockPath: string | null = null;
function lockIdentity(identity: Identity): void {
  const file = path.join(process.cwd(), `.token-lock-${identity.toHexString()}`);
  if (lockPath === file) return;
  try {
    const fd = fs.openSync(file, "wx", 0o600);
    fs.writeFileSync(fd, String(process.pid));
    fs.closeSync(fd);
  } catch (err) {
    if ((err as NodeJS.ErrnoException).code !== "EEXIST") throw err;
    const pid = Number(fs.readFileSync(file, "utf8"));
    if (!Number.isSafeInteger(pid) || pid <= 0) throw new Error("Invalid bot lock; inspect before removing");
    try { process.kill(pid, 0); }
    catch (e) {
      if ((e as NodeJS.ErrnoException).code === "ESRCH") {
        fs.unlinkSync(file);
        return lockIdentity(identity);
      }
      throw e;
    }
    throw new Error(`Bot identity is already running in process ${pid}`);
  }
  lockPath = file;
}
process.on("exit", () => { if (lockPath) fs.unlinkSync(lockPath); });
function onConnect(c: DbConnection, identity: Identity, token: string) {
  try { lockIdentity(identity); }
  catch (err) { console.error(String(err)); c.disconnect(); process.exit(1); }
  conn = c;
  myIdentity = identity;
  saveToken(token);
  say(`connected as ${identity.toHexString()}`);
  logEvent(null, "connected", { identity: identity.toHexString(), config: cfg, timingPolicy: "paid-tile-price-search-v2", bidSearch: cfg.nativeBidSearch });

  wireTables(c);
  subscribe(c);
}

let reconnectDelay = 1000;
let reconnectTimer: NodeJS.Timeout | null = null;
function scheduleReconnect() {
  if (reconnectTimer || stopping) return;
  const d = reconnectDelay;
  reconnectDelay = Math.min(reconnectDelay * 2, 15000);
  say(`reconnecting in ${d}ms`);
  reconnectTimer = setTimeout(() => {
    reconnectTimer = null;
    connect();
  }, d);
}

function connect() {
  say(`connecting to ${DB_NAME} at ${HOST}`);
  try {
    DbConnection.builder()
      .withUri(HOST)
      .withDatabaseName(DB_NAME)
      .withToken(BOT_TOKEN ?? loadToken())
      .onConnect(
        safe("onConnect", (c: DbConnection, id: Identity, tok: string) => {
          reconnectDelay = 1000;
          onConnect(c, id, tok);
        }),
      )
      .onConnectError((_ctx: ErrorContext, err: Error) => {
        console.error(tag, "connect error:", err.message);
        logEvent(null, "connect_error", { message: err.message });
        scheduleReconnect();
      })
      .onDisconnect(() => {
        say("disconnected");
        logEvent(null, "disconnected");
        conn = null;
        lobbyJoinInFlight = false;
        if (lobbyRetryTimer) clearTimeout(lobbyRetryTimer);
        lobbyRetryTimer = null;
        // The mirror is gone; it is rebuilt from the new subscription.
        for (const q of reservations.values()) q.disconnect();
        savePending();
        worker?.postMessage({ type: "cancel" });
        store.clear();
        for (const mc of matches.values()) mc.ended = true;
        matches.clear();
        scheduleReconnect();
      })
      .build();
  } catch (e) {
    console.error(tag, "connect threw:", e);
    scheduleReconnect();
  }
}

function main() {
  say(
    `config: dryRun=${cfg.dryRun} lambda=${cfg.lambdaMode} safeMaxBid=${cfg.safeMaxBid ?? "off"} ` +
      `rollouts=${cfg.nRollouts} horizon=${cfg.horizon} seed=${cfg.seed}`,
  );
  startWorker();
  connect();
  const duration = Number(process.env.EVAL_MAX_MS ?? 0);
  if (duration > 0) setTimeout(stop, duration);
  // Watchdog: if a lobby join was lost (or the server restarted), try again.
  if (!TOURNAMENT_ID) {
    setInterval(
      safe("watchdog", () => {
        if (conn && myBotId !== null && !inRunningMatch(conn)) joinLobby();
      }),
      5000,
    );
  }
}

main();
