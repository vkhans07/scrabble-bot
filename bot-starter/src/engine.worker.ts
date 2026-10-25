// Heavy computation lives here so the websocket handler never blocks.
//
// Each snapshot from the main thread first gets a fast v entry for the open
// auction's letter, then a full-table refinement (and cash-in evaluation).
// A newer snapshot aborts whatever is running.

import { parentPort, workerData } from "node:worker_threads";
import type { Config } from "./config.js";
import { Dict } from "./dict.js";
import { RackEval } from "./rackeval.js";
import { makeRng } from "./rng.js";
import { Rollouts } from "./rollout.js";
import { NativeEngine, type NativeResult } from "./native.js";
import type { Snapshot } from "./state.js";

const cfg = workerData as Config;
const port = parentPort!;

const dict = Dict.load();
const ev = new RackEval(dict, cfg.exactCap);
const rollouts = new Rollouts(dict, ev, cfg, makeRng((cfg.seed ^ 0x5bd1e995) >>> 0));

let token = 0;

// Native engine, when built. Any failure drops back to the JS rollouts below.
let native: NativeEngine | null = null;
const nativeLog = (msg: string) => port.postMessage({ type: "native_log", message: msg });

function startNative() {
  if (!cfg.native) return;
  native = new NativeEngine(
    cfg,
    (r: NativeResult) => {
      port.postMessage({
        type: "result",
        phase: r.phase,
        seq: r.seq,
        matchId: r.matchId,
        auctionId: r.auctionId,
        rackKey: lastRackKey.get(r.seq),
        v: r.v,
        lam: r.lam,
        fairBid: r.fairBid,
        nScen: r.nScen,
        ms: r.ms,
        cashin: r.cashin,
        perf: { native: 1, base: r.base, memoHitRate: r.memoHitRate, bigCalls: r.bigCalls },
      });
    },
    nativeLog,
    (why) => {
      nativeLog(`native engine down: ${why}; using JS rollouts`);
      native = null;
      setTimeout(startNative, 5000);
    },
  );
  if (!native.start()) native = null;
}

const lastRackKey = new Map<number, string>();

function perf() {
  const calls = ev.hits + ev.misses;
  return {
    rolloutsPerSec: rollouts.totalMs > 0 ? (rollouts.totalScenarios / rollouts.totalMs) * 1000 : 0,
    memoHitRate: calls > 0 ? ev.hits / calls : 0,
    memoSize: ev.memo.size,
    greedyFallbacks: ev.greedyCount,
    exactEvals: ev.exactCount,
  };
}

async function handle(snap: Snapshot) {
  const mine = ++token;
  const abort = () => token !== mine;
  const t0 = performance.now();
  try {
    if (snap.isFinal) {
      port.postMessage({ type: "result", phase: "skip", seq: snap.seq, matchId: snap.matchId });
      return;
    }
    if (native && native.ready && !snap.firstPrice) {
      lastRackKey.set(snap.seq, snap.rackKey);
      if (lastRackKey.size > 64) lastRackKey.delete(lastRackKey.keys().next().value!);
      native.snapshot(snap);
      return;
    }
    if (snap.letter >= 0) {
      const f = await rollouts.fast(snap, abort);
      if (abort()) return;
      port.postMessage({
        type: "result",
        phase: "fast",
        seq: snap.seq,
        matchId: snap.matchId,
        auctionId: snap.auctionId,
        rackKey: snap.rackKey,
        v: f.v,
        computed: f.computed,
        nScen: f.nScen,
        ms: f.ms,
        cashin: [],
        perf: perf(),
      });
    }
    const r = await rollouts.full(snap, abort);
    if (abort()) return;
    port.postMessage({
      type: "result",
      phase: "full",
      seq: snap.seq,
      matchId: snap.matchId,
      auctionId: snap.auctionId,
      rackKey: snap.rackKey,
      v: r.v,
      computed: r.computed,
      nScen: r.nScen,
      ms: r.ms,
      totalMs: performance.now() - t0,
      cashin: r.cashin.slice(0, 8),
      perf: perf(),
    });
  } catch (e) {
    port.postMessage({
      type: "error",
      seq: snap.seq,
      message: e instanceof Error ? (e.stack ?? e.message) : String(e),
    });
  }
}

port.on("message", (msg: { type: string; snap?: Snapshot }) => {
  if (msg.type === "snapshot" && msg.snap) void handle(msg.snap);
  else if (msg.type === "shutdown") {
    token++;
    native?.stop();
    port.close();
  } else if (msg.type === "cancel") {
    token++;
    native?.cancel();
  }
});

startNative();
port.postMessage({ type: "ready", loadMs: performance.now() });
