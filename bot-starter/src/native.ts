// Client for the native rollout engine (native/engine.cpp), a child process
// speaking a line protocol on stdin/stdout. The caller falls back to the JS
// rollouts whenever this is not ready, so a crash here never costs a bid.

import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import { existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import type { Config } from "./config.js";
import type { Snapshot } from "./state.js";

export interface NativeResult {
  phase: "fast" | "refine" | "search" | "full";
  seq: number;
  matchId: string;
  auctionId: string;
  nScen: number;
  ms: number;
  lam: number;
  base: number;
  fairBid: number;
  v: Float64Array;
  memoHitRate: number;
  bigCalls: number;
  cashin: { word: string; reward: number; delta: number; gain: number }[];
}

const KIND: Record<string, number> = { Cheapskate: 0, ValueBidder: 1, Aggressive: 2, Real: 3 };

export function nativePaths() {
  return {
    bin: fileURLToPath(new URL("../native/engine", import.meta.url)),
    words: fileURLToPath(new URL("../../spacetimedb/wordlist.txt", import.meta.url)),
  };
}

export class NativeEngine {
  private child: ChildProcessWithoutNullStreams | null = null;
  private buf = "";
  ready = false;

  constructor(
    private cfg: Config,
    private onResult: (r: NativeResult) => void,
    private onLog: (msg: string) => void,
    private onDown: (why: string) => void,
  ) {}

  start(): boolean {
    const { bin, words } = nativePaths();
    if (!existsSync(bin)) {
      this.onLog(`native engine not built (${bin})`);
      return false;
    }
    const c = spawn(bin, ["--wordlist", words], { stdio: ["pipe", "pipe", "pipe"] });
    this.child = c;
    c.stdout.setEncoding("utf8");
    c.stdout.on("data", (d: string) => this.onData(d));
    c.stderr.setEncoding("utf8");
    c.stderr.on("data", (d: string) => this.onLog(`stderr: ${d.trim()}`));
    c.stdin.on("error", () => {});
    c.on("error", (e) => this.down(`spawn error: ${e.message}`));
    c.on("exit", (code, sig) => this.down(`exit code=${code} sig=${sig}`));
    const cf = this.cfg;
    this.send(
      `CFG threads ${cf.nativeThreads} seed ${cf.seed} exactCap ${cf.nativeExactCap} allinK ${cf.allinK} bidSearch ${cf.nativeBidSearch === "exhaustive" ? 2 : cf.nativeBidSearch === "adaptive" ? 1 : 0} ` +
        `allinMinV ${cf.allinMinV} endgameK ${cf.endgameK} nMax ${cf.nativeScenarios} nFast ${cf.nativeFast} nRefine ${cf.nativeRefine} refineMs ${cf.nativeRefineMs} ` +
        `${process.env.NATIVE_EXTRA ?? ""} fastMs ${cf.fastBudgetMs} fullMs ${cf.rolloutBudgetMs} damp ${cf.vDamp} cashK ${cf.cashinK > 8 ? 8 : cf.cashinK}`,
    );
    return true;
  }

  private down(why: string) {
    if (!this.child) return;
    this.child = null;
    this.ready = false;
    this.onDown(why);
  }

  private send(line: string) {
    try {
      this.child?.stdin.write(line + "\n");
    } catch {
      /* the exit handler reports it */
    }
  }

  private onData(d: string) {
    this.buf += d;
    let i: number;
    while ((i = this.buf.indexOf("\n")) >= 0) {
      const line = this.buf.slice(0, i);
      this.buf = this.buf.slice(i + 1);
      try {
        this.onLine(line);
      } catch (e) {
        this.onLog(`bad line: ${line.slice(0, 80)} (${e})`);
      }
    }
  }

  private onLine(line: string) {
    const f = line.split(" ");
    if (f[0] === "READY") {
      this.ready = true;
      this.onLog(`ready (${f[1]} words)`);
      return;
    }
    if (f[0] === "ERR") {
      this.onLog(line);
      return;
    }
    if (f[0] !== "RES") return;
    const v = new Float64Array(26);
    for (let i = 0; i < 26; i++) v[i] = Number(f[9 + i]);
    const nc = Number(f[37]);
    const r: NativeResult = {
      phase: f[1] as "fast" | "full",
      seq: Number(f[2]),
      matchId: f[3],
      auctionId: f[4],
      nScen: Number(f[5]),
      ms: Number(f[6]),
      lam: Number(f[7]),
      base: Number(f[8]),
      v,
      memoHitRate: Number(f[35]),
      bigCalls: Number(f[36]),
      fairBid: Number(f[38 + nc * 4]),
      cashin: [],
    };
    for (let k = 0; k < nc; k++) {
      const o = 38 + k * 4;
      const reward = Number(f[o + 1]);
      const gain = Number(f[o + 3]);
      r.cashin.push({ word: f[o], reward, delta: gain - reward, gain });
    }
    if (!Number.isFinite(r.lam) || v.some((x) => !Number.isFinite(x))) throw new Error("non-finite result");
    this.onResult(r);
  }

  snapshot(s: Snapshot) {
    const cf = this.cfg;
    const opps = s.opps
      .map((o) => `${KIND[o.kind] ?? 3} ${Math.max(0, Math.round(o.balance))} ${cf.realOppBaseBonus} ${cf.realOppSlope} ${cf.realOppJitter}`)
      .join(" ");
    this.send(
      `SNAP ${s.seq} ${s.matchId} ${s.auctionId ?? "none"} ${s.letter} ${s.usableLeft} ${Math.round(s.balance)} ` +
        `${Math.round(s.score)} ${s.needCashin ? 1 : 0} ${Array.from(s.rack).join(" ")} ${Array.from(s.U).join(" ")} ` +
        `${s.opps.length} ${opps}`,
    );
  }

  cancel() {
    this.send("CANCEL");
  }

  stop() {
    this.send("QUIT");
    try {
      this.child?.kill();
    } catch {
      /* already gone */
    }
  }
}
