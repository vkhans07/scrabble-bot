// Paired sandbox comparison on worker threads. Every game seed is played by
// both arms in the same worker, so the comparison is paired by construction:
//
//   npm run sandbox:pair -- --jobs 5 --games 200 --seed 42 \
//     --seats live-lite,rack,rack,rack,rack,rack --focus live-lite \
//     --base "--bid-scale 1" --cand "--bid-scale 0.8" --out logs/claude-evals/x.json
//
// Flags outside --base/--cand are shared by both arms.

import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { pathToFileURL } from "node:url";
import { Worker, isMainThread, parentPort, workerData } from "node:worker_threads";
import { Dict } from "./dict.js";
import { RackEval } from "./rackeval.js";
import { parseArgs, runMatch, summarize, type SandboxOptions } from "./sandbox.js";
import { NativeDriver, sandboxConfig } from "./sandbox-native.js";

type Game = Record<string, unknown>;
interface Job { indices: number[]; base: SandboxOptions; cand: SandboxOptions }

const METRICS = ["wonMatch", "lastPlace", "place", "score", "balance", "spent", "tilesWon", "words", "playableRackValue"] as const;

function gameSeed(seed: number, i: number): number {
  return (seed + Math.imul(i, 0x9e3779b9)) >>> 0;
}

function takeFlag(args: string[], name: string): [string | undefined, string[]] {
  const i = args.indexOf(name);
  if (i < 0) return [undefined, args];
  return [args[i + 1], [...args.slice(0, i), ...args.slice(i + 2)]];
}

function focusRows(game: Game, focus: string): Record<string, unknown>[] {
  return (game.standings as Record<string, unknown>[]).filter((r) => r.policy === focus);
}

// Mean over the focus policy's seats in one game.
function gameMetric(game: Game, focus: string, k: string): number {
  const rows = focusRows(game, focus);
  return rows.reduce((s, r) => s + Number(r[k]), 0) / Math.max(1, rows.length);
}

function pairedStats(diffs: number[]) {
  const n = diffs.length;
  const mean = diffs.reduce((s, d) => s + d, 0) / n;
  const sd = Math.sqrt(diffs.reduce((s, d) => s + (d - mean) ** 2, 0) / Math.max(1, n - 1));
  const se = sd / Math.sqrt(n);
  return {
    meanDiff: mean, sd, se, ci95: [mean - 1.96 * se, mean + 1.96 * se],
    candBetter: diffs.filter((d) => d > 0).length,
    candWorse: diffs.filter((d) => d < 0).length,
    ties: diffs.filter((d) => d === 0).length,
  };
}

function quarters(games: Game[], focus: string): number[] {
  const out = [0, 0, 0, 0];
  let n = 0;
  for (const g of games) for (const r of focusRows(g, focus)) {
    (r.spentByQuarter as number[]).forEach((x, q) => { out[q]! += x; });
    n++;
  }
  return out.map((x) => x / Math.max(1, n));
}

async function main(argv: string[]) {
  let [baseFlags, rest] = takeFlag(argv, "--base");
  let candFlags: string | undefined;
  [candFlags, rest] = takeFlag(rest, "--cand");
  let jobsArg: string | undefined, focusArg: string | undefined;
  [jobsArg, rest] = takeFlag(rest, "--jobs");
  [focusArg, rest] = takeFlag(rest, "--focus");
  const split = (s?: string) => (s ?? "").split(/\s+/).filter(Boolean);
  const base = parseArgs([...rest, ...split(baseFlags)]);
  const cand = parseArgs([...rest, ...split(candFlags)]);
  const out = base.out;
  const focus = focusArg ?? "live-lite";
  const jobs = Math.max(1, Math.min(Number(jobsArg ?? Math.max(1, os.availableParallelism() - 2)), base.games));
  const indices = Array.from({ length: base.games }, (_, i) => i);
  const started = Date.now();

  const shards = await Promise.all(Array.from({ length: jobs }, (_, j) => {
    const job: Job = { indices: indices.filter((i) => i % jobs === j), base, cand };
    return new Promise<{ base: Game[]; cand: Game[] }>((resolve, reject) => {
      const w = new Worker(new URL(import.meta.url), { workerData: job, execArgv: process.execArgv });
      w.once("message", resolve);
      w.once("error", reject);
    });
  }));
  const bySeed = (a: Game, b: Game) => Number(a.index) - Number(b.index);
  const baseGames = shards.flatMap((s) => s.base).sort(bySeed);
  const candGames = shards.flatMap((s) => s.cand).sort(bySeed);

  const paired = Object.fromEntries(METRICS.map((k) => [k, pairedStats(baseGames.map((g, i) =>
    gameMetric(candGames[i]!, focus, k) - gameMetric(g, focus, k)))]));
  const result = {
    description: "Paired offline sandbox comparison. Simulation evidence only; diffs are candidate minus base, per game seed.",
    argv, focus, jobs, elapsedMs: Date.now() - started,
    baseOptions: base, candOptions: cand,
    baseSummary: summarize(baseGames, base.seats), candSummary: summarize(candGames, cand.seats),
    spentByQuarter: { base: quarters(baseGames, focus), cand: quarters(candGames, focus) },
    paired,
    baseGames, candGames,
  };
  if (out) {
    fs.mkdirSync(path.dirname(path.resolve(out)), { recursive: true });
    fs.writeFileSync(path.resolve(out), `${JSON.stringify(result, null, 2)}\n`);
  }
  const f = (x: number) => (x >= 0 ? "+" : "") + x.toFixed(3);
  console.log(`${base.games} paired games, ${jobs} workers, ${(result.elapsedMs / 1000).toFixed(1)}s; focus ${focus}`);
  for (const k of METRICS) {
    const p = paired[k]!;
    const b = result.baseSummary.find((r) => r.policy === focus);
    console.log(`${k.padEnd(18)} diff ${f(p.meanDiff)}  95% CI [${f(p.ci95[0]!)}, ${f(p.ci95[1]!)}]  better/worse ${p.candBetter}/${p.candWorse}`);
    void b;
  }
  console.log(`spentByQuarter base ${result.spentByQuarter.base.map((x) => x.toFixed(1)).join("/")} cand ${result.spentByQuarter.cand.map((x) => x.toFixed(1)).join("/")}`);
  for (const [name, s] of [["base", result.baseSummary], ["cand", result.candSummary]] as const)
    for (const r of s) console.log(`${name} ${String(r.policy).padEnd(10)} win ${(100 * r.winRate).toFixed(1)}% last ${(100 * r.lastRate).toFixed(1)}% place ${r.averagePlace.toFixed(2)} score ${r.averageScore.toFixed(1)} cash ${r.averageBalance.toFixed(1)} spent ${r.averageSpent.toFixed(1)} tiles ${r.averageTilesWon.toFixed(1)} words ${r.averageWords.toFixed(2)} rack ${r.averagePlayableRackValue.toFixed(1)}`);
  if (out) console.log(`wrote ${path.resolve(out)}`);
}

async function work(job: Job) {
  const dict = Dict.load();
  const ev = new RackEval(dict, 16);
  const byLength = [...dict.allWords].sort((a, b) => b.length - a.length || a.localeCompare(b));
  // One engine per distinct native setting; paired arms with equal settings share it.
  const drivers = new Map<string, NativeDriver>();
  const driverFor = (o: SandboxOptions) => {
    if (!o.seats.includes("native")) return undefined;
    const key = JSON.stringify(o.native);
    if (!drivers.has(key)) drivers.set(key, new NativeDriver(sandboxConfig(o.native)));
    return drivers.get(key);
  };
  const out: { base: Game[]; cand: Game[] } = { base: [], cand: [] };
  try {
    for (const i of job.indices) {
      out.base.push({ index: i, ...await runMatch(gameSeed(job.base.seed, i), job.base, ev, byLength, driverFor(job.base), "b") });
      out.cand.push({ index: i, ...await runMatch(gameSeed(job.cand.seed, i), job.cand, ev, byLength, driverFor(job.cand), "c") });
    }
  } finally {
    for (const d of drivers.values()) d.close();
  }
  return out;
}

if (!isMainThread) {
  work(workerData as Job).then((r) => parentPort!.postMessage(r), (e) => { throw e; });
} else if (import.meta.url === pathToFileURL(process.argv[1] ?? "").href) {
  main(process.argv.slice(2)).catch((e) => {
    console.error(e instanceof Error ? e.message : String(e));
    process.exitCode = 1;
  });
}
