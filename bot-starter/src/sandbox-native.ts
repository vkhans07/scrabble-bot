// Drives the live native engine (native/engine.cpp) from the offline sandbox.
// One request is in flight at a time: the sandbox posts a snapshot and waits
// for its "full" result, keeping the newest paid-tile fair bid on the way, as
// the live client does between an auction opening and its deadline.

import { loadConfig, type Config } from "./config.js";
import { NativeEngine, type NativeResult } from "./native.js";
import type { Snapshot } from "./state.js";

export interface NativeSettings {
  paidBid: boolean;
  oppBonus: number;
  oppSlope: number;
  oppJitter: number;
  threads: number;
  cfg?: Record<string, number | boolean | string>;
}

export interface NativeEval {
  v: Float64Array;
  lam: number;
  fairBid: number | null;
  cashin: NativeResult["cashin"];
}

// Live config with time budgets lifted, so scenario counts (not machine load)
// bound each pass and paired arms see the same amount of search.
export function sandboxConfig(s: NativeSettings): Config {
  const base = loadConfig();
  for (const k of Object.keys(s.cfg ?? {}))
    if (!(k in base)) throw new Error(`--cfg ${k}: not a Config field`);
  return {
    ...base,
    seed: 241024,
    nativeThreads: s.threads,
    nativePaidBid: s.paidBid,
    realOppBaseBonus: s.oppBonus,
    realOppSlope: s.oppSlope,
    realOppJitter: s.oppJitter,
    fastBudgetMs: 60_000,
    rolloutBudgetMs: 60_000,
    nativeRefineMs: 60_000,
    ...(s.cfg ?? {}),
  } as Config;
}

export class NativeDriver {
  private engine: NativeEngine;
  private seq = 0;
  private pending: {
    seq: number;
    fairBid: number | null;
    resolve: (r: NativeEval) => void;
    reject: (e: Error) => void;
  } | null = null;
  private ready: Promise<void>;

  constructor(readonly cfg: Config) {
    let markReady!: () => void;
    let fail!: (e: Error) => void;
    this.ready = new Promise((res, rej) => { markReady = res; fail = rej; });
    this.engine = new NativeEngine(
      cfg,
      (r) => this.onResult(r),
      (msg) => { if (msg.startsWith("ready")) markReady(); },
      (why) => {
        const e = new Error(`native engine down: ${why}`);
        fail(e);
        this.pending?.reject(e);
        this.pending = null;
      },
    );
    if (!this.engine.start()) fail(new Error("native engine not built; run npm run build:native"));
  }

  private onResult(r: NativeResult) {
    const p = this.pending;
    if (!p || r.seq !== p.seq) return;
    if (r.fairBid >= 0) p.fairBid = r.fairBid;
    if (r.phase !== "full") return;
    this.pending = null;
    p.resolve({ v: r.v, lam: r.lam, fairBid: p.fairBid, cashin: r.cashin });
  }

  async evaluate(snap: Omit<Snapshot, "seq">): Promise<NativeEval> {
    await this.ready;
    const seq = ++this.seq;
    return new Promise((resolve, reject) => {
      this.pending = { seq, fairBid: null, resolve, reject };
      this.engine.snapshot({ ...snap, seq });
    });
  }

  close() {
    this.engine.stop();
  }
}
