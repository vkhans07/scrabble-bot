// Counterfactual replay of recorded live matches. Our seat is re-played by
// the live decision loop (native engine + composeBid + cash-in/endgame rules)
// under a candidate config; everything else is what actually happened: the
// letter sequence, our starting rack, our belief U, opponent balances, and the
// highest opposing bid in each auction (the runner-up price when we won, the
// archived winning bid when someone else did). Opponents do not react to the
// candidate, and their final scores are the recorded ones, so a candidate that
// takes letters an opponent actually won is scored against an opponent who
// still had them: place estimates are conservative for changes that win more.
//
//   node --import tsx src/replay.ts --jobs 2 --logs logs/prod-b10334e \
//     --base "" --cand "--cfg paidBidWarmup=1" --out logs/claude-evals/r1.json

import * as fs from "node:fs";
import * as path from "node:path";
import { pathToFileURL } from "node:url";
import { Worker, isMainThread, parentPort, workerData } from "node:worker_threads";
import { A, Dict } from "./dict.js";
import { bidAmount, composeBid, shouldDumpPlan } from "./policy.js";
import { RackEval } from "./rackeval.js";
import { NativeDriver, sandboxConfig, type NativeSettings } from "./sandbox-native.js";
import { fits, parseArgs, wordReward } from "./sandbox.js";

interface Auction {
  id: string;
  letter: number;
  usableLeft: number;
  bagTotal: number;
  rackAtOpen: string;
  U: number[];
  oppMax: number;
  winner: string | null;
  paid: number;
  opps: { botId: string; balance: number }[];
}

export interface RecordedMatch {
  matchId: string;
  file: string;
  start: Uint8Array;
  auctions: Auction[];
  oppFinal: number[];
  actual: { score: number; place: number; balance: number; tiles: number; words: number };
}

const ME = "112";

function rackOf(s: string): Uint8Array {
  const r = new Uint8Array(26);
  for (const c of s) r[c.charCodeAt(0) - A]++;
  return r;
}

// A full-length match with a summary, every auction's result and top bid.
export function loadMatch(file: string): RecordedMatch | null {
  const ev: Record<string, any>[] = [];
  for (const line of fs.readFileSync(file, "utf8").split("\n")) {
    if (!line) continue;
    try { ev.push(JSON.parse(line)); } catch { /* partial last line */ }
  }
  const summary = ev.filter((e) => e.type === "match_summary").at(-1);
  const opens = ev.filter((e) => e.type === "auction_open");
  if (!summary || opens.length < 60 || opens[0].usableLeft !== 66) return null;
  const results = new Map<string, any>(ev.filter((e) => e.type === "auction_result").map((e) => [e.row.auctionId, e.row]));
  const tops = new Map<string, number>();
  for (const e of ev) if (e.type === "bid_archive") for (const r of e.rows) tops.set(r.auctionId, Number(r.topBid));
  const balances = new Map<string, number>();
  const auctions: Auction[] = [];
  let k = 0;
  for (const e of ev) {
    if (e.type === "match_participant" && e.row.botId !== ME) balances.set(e.row.botId, Number(e.row.balance));
    if (e.type !== "auction_open") continue;
    const r = results.get(e.auctionId);
    if (!r) return null;
    const winner = r.winnerBotId ?? null;
    const paid = Number(r.paid);
    let oppMax = 0;
    if (winner === ME) oppMax = paid > 1 ? paid : 0; // the reserve of 1 is paid when nobody else bid
    else if (winner !== null) {
      const top = tops.get(e.auctionId);
      if (top === undefined) return null;
      oppMax = top;
    }
    auctions.push({
      id: e.auctionId, letter: e.letter.charCodeAt(0) - A, usableLeft: e.usableLeft, bagTotal: e.bagTotal,
      rackAtOpen: e.rack, U: e.U, oppMax, winner, paid,
      opps: [...balances].map(([botId, balance]) => ({ botId, balance })),
    });
    k++;
  }
  const scores = (summary.scores as { bot: string; score: string }[]).map((s) => Number(s.score));
  const mine = Number(summary.finalScore);
  const oppFinal = scores.slice();
  oppFinal.splice(oppFinal.indexOf(mine), 1);
  return {
    matchId: path.basename(file).slice(6, -6), file, start: rackOf(opens[0].rack), auctions, oppFinal,
    actual: { score: mine, place: summary.place, balance: Number(summary.finalBalance), tiles: summary.auctionsWon, words: summary.wordsPlayed },
  };
}

interface Seat { rack: Uint8Array; balance: number; score: number; spent: number; won: number; words: string[]; spentByQuarter: number[] }

function cash(seat: Seat, word: string) {
  for (const c of word) seat.rack[c.charCodeAt(0) - A]--;
  const r = wordReward(word);
  seat.balance += r;
  seat.score += r;
  seat.words.push(word);
}

export async function replay(m: RecordedMatch, driver: NativeDriver, ev: RackEval, tag: string) {
  const cfg = driver.cfg;
  const seat: Seat = { rack: m.start.slice(), balance: 100, score: 0, spent: 0, won: 0, words: [], spentByQuarter: [0, 0, 0, 0] };
  let lastCashinKey: string | null = null;
  let sinceCashin = 0;
  let a1 = { bid: 0, won: false, paid: 0 };
  const n = m.auctions.length;
  for (let i = 0; i < n; i++) {
    const a = m.auctions[i]!;
    const isFinal = a.bagTotal === 0;
    if (isFinal || shouldDumpPlan(a.usableLeft, false, cfg)) {
      for (const w of ev.plan(seat.rack, 22, 400_000).words) if (fits(w, seat.rack)) cash(seat, w);
    }
    if (isFinal || seat.balance < 1) continue;
    sinceCashin++;
    let result = null;
    for (let pass = 0; pass < 4; pass++) {
      const rackKey = RackEval.keyOf(seat.rack);
      const needCashin = rackKey !== lastCashinKey || sinceCashin >= cfg.cashinEvery;
      result = await driver.evaluate({
        matchId: `${tag}${m.matchId}`, auctionId: a.id, letter: a.letter, isFinal, usableLeft: a.usableLeft, bagTotal: a.bagTotal,
        firstPrice: false, balance: seat.balance, score: seat.score, rack: seat.rack.slice(), U: Uint8Array.from(a.U), rackKey, needCashin,
        opps: a.opps.map((o) => ({ ...o, kind: "Real" as const })),
      });
      if (!needCashin) break;
      lastCashinKey = rackKey;
      sinceCashin = 0;
      const best = result.cashin[0];
      if (shouldDumpPlan(a.usableLeft, false, cfg) || !best || best.gain <= cfg.cashMargin || !fits(best.word, seat.rack)) break;
      cash(seat, best.word);
    }
    if (!result) continue;
    const v = result.v[a.letter]!;
    const bid = composeBid(bidAmount(v, seat.balance, a.usableLeft, cfg, undefined, result.lam), v, result.lam,
      result.fairBid, i + 1, a.usableLeft, seat.balance, cfg).amount;
    // Ties go to the recorded opponent: the server breaks them at random.
    const win = bid > a.oppMax && bid > 0;
    const pay = win ? Math.max(1, a.oppMax) : 0;
    if (win && pay <= seat.balance) {
      seat.balance -= pay;
      seat.spent += pay;
      seat.won++;
      seat.rack[a.letter]++;
      seat.spentByQuarter[Math.min(3, Math.floor(4 * i / n))]! += pay;
    }
    if (i === 0) a1 = { bid, won: win, paid: pay };
  }
  const place = 1 + m.oppFinal.filter((s) => s > seat.score).length;
  return {
    matchId: m.matchId, score: seat.score, place, wonMatch: place === 1, lastPlace: !m.oppFinal.some((s) => s < seat.score),
    balance: seat.balance, spent: seat.spent, tilesWon: seat.won, words: seat.words.length,
    longWords: seat.words.filter((w) => w.length >= 7).length, wordList: seat.words, spentByQuarter: seat.spentByQuarter,
    playableRackValue: ev.best(seat.rack, 16), a1, actual: m.actual,
  };
}

type Row = Awaited<ReturnType<typeof replay>>;
const METRICS = ["wonMatch", "lastPlace", "place", "score", "balance", "spent", "tilesWon", "words", "longWords", "playableRackValue"] as const;

function stats(diffs: number[]) {
  const n = diffs.length, mean = diffs.reduce((s, d) => s + d, 0) / n;
  const sd = Math.sqrt(diffs.reduce((s, d) => s + (d - mean) ** 2, 0) / Math.max(1, n - 1)), se = sd / Math.sqrt(n);
  return { meanDiff: mean, se, ci95: [mean - 1.96 * se, mean + 1.96 * se], better: diffs.filter((d) => d > 0).length, worse: diffs.filter((d) => d < 0).length };
}

function mean(rows: Row[], k: string) {
  return rows.reduce((s, r) => s + Number((r as any)[k]), 0) / Math.max(1, rows.length);
}

function settingsFrom(flags: string, shared: string[]): NativeSettings {
  return parseArgs([...shared, ...flags.split(/\s+/).filter(Boolean)]).native;
}

async function main(argv: string[]) {
  const take = (name: string) => { const i = argv.indexOf(name); if (i < 0) return undefined; const v = argv[i + 1]; argv.splice(i, 2); return v; };
  const logs: string[] = [];
  for (let v = take("--logs"); v !== undefined; v = take("--logs")) logs.push(v);
  const base = take("--base") ?? "", cand = take("--cand") ?? "", out = take("--out");
  const jobs = Math.max(1, Number(take("--jobs") ?? 2));
  const limit = Number(take("--limit") ?? Infinity);
  const files = logs.flatMap((d) => fs.readdirSync(d).filter((f) => /^match-\d+\.jsonl$/.test(f)).map((f) => path.join(d, f))).sort();
  const matches = files.map(loadMatch).filter((m): m is RecordedMatch => m !== null).slice(0, limit);
  const started = Date.now();
  const shards = await Promise.all(Array.from({ length: Math.min(jobs, matches.length) }, (_, j) =>
    new Promise<{ base: Row[]; cand: Row[] }>((resolve, reject) => {
      const w = new Worker(new URL(import.meta.url), {
        workerData: { files: matches.filter((_, i) => i % jobs === j).map((m) => m.file), base: settingsFrom(base, argv), cand: settingsFrom(cand, argv) },
        execArgv: process.execArgv,
      });
      w.once("message", resolve);
      w.once("error", reject);
    })));
  const byId = (a: Row, b: Row) => a.matchId.localeCompare(b.matchId);
  const B = shards.flatMap((s) => s.base).sort(byId), C = shards.flatMap((s) => s.cand).sort(byId);
  const paired = Object.fromEntries(METRICS.map((k) => [k, stats(B.map((b, i) => Number((C[i] as any)[k]) - Number((b as any)[k])))]));
  const summary = (rows: Row[]) => ({
    n: rows.length, ...Object.fromEntries(METRICS.map((k) => [k, mean(rows, k)])),
    spentByQuarter: [0, 1, 2, 3].map((q) => rows.reduce((s, r) => s + r.spentByQuarter[q]!, 0) / rows.length),
    a1Bid: rows.reduce((s, r) => s + r.a1.bid, 0) / rows.length, a1WinRate: rows.filter((r) => r.a1.won).length / rows.length,
  });
  const actual = { n: B.length, score: B.reduce((s, r) => s + r.actual.score, 0) / B.length, place: B.reduce((s, r) => s + r.actual.place, 0) / B.length };
  const result = { description: "Counterfactual replay of recorded live matches; candidate minus base per match.", argv: process.argv.slice(2), logs, base, cand,
    elapsedMs: Date.now() - started, actual, baseSummary: summary(B), candSummary: summary(C), paired, base_rows: B, cand_rows: C };
  if (out) { fs.mkdirSync(path.dirname(path.resolve(out)), { recursive: true }); fs.writeFileSync(out, JSON.stringify(result, null, 1) + "\n"); }
  const f = (x: number) => (x >= 0 ? "+" : "") + x.toFixed(2);
  console.log(`${B.length} matches, ${(result.elapsedMs / 1000).toFixed(0)}s; recorded actual score ${actual.score.toFixed(1)} place ${actual.place.toFixed(2)}`);
  for (const k of METRICS) {
    const p = paired[k]!;
    console.log(`${k.padEnd(18)} base ${mean(B, k).toFixed(2).padStart(7)} cand ${mean(C, k).toFixed(2).padStart(7)} diff ${f(p.meanDiff)} [${f(p.ci95[0]!)}, ${f(p.ci95[1]!)}] +${p.better}/-${p.worse}`);
  }
  console.log(`spentByQuarter base ${result.baseSummary.spentByQuarter.map((x) => x.toFixed(0)).join("/")} cand ${result.candSummary.spentByQuarter.map((x) => x.toFixed(0)).join("/")}`);
  if (out) console.log(`wrote ${path.resolve(out)}`);
}

async function work(job: { files: string[]; base: NativeSettings; cand: NativeSettings }) {
  const dict = Dict.load();
  const ev = new RackEval(dict, 16);
  const b = new NativeDriver(sandboxConfig(job.base)), c = new NativeDriver(sandboxConfig(job.cand));
  const out = { base: [] as Row[], cand: [] as Row[] };
  try {
    for (const f of job.files) {
      const m = loadMatch(f)!;
      out.base.push(await replay(m, b, ev, "b"));
      out.cand.push(await replay(m, c, ev, "c"));
    }
  } finally { b.close(); c.close(); }
  return out;
}

if (!isMainThread) work(workerData).then((r) => parentPort!.postMessage(r), (e) => { throw e; });
else if (import.meta.url === pathToFileURL(process.argv[1] ?? "").href) {
  main(process.argv.slice(2)).catch((e) => { console.error(e instanceof Error ? e.stack : String(e)); process.exitCode = 1; });
}
