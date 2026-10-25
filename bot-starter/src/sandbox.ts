// Offline, seeded match simulator for comparing bidding policies without
// connecting to SpacetimeDB. The planner is a receding-horizon benchmark, not
// an exact solution: it searches candidate bids using sampled future tiles.

import * as fs from "node:fs";
import * as path from "node:path";
import { pathToFileURL } from "node:url";
import { A, BAG, Dict, LETTER_VALUE } from "./dict.js";
import { bidAmount, pacedLambda, shouldDumpPlan } from "./policy.js";
import { RackEval } from "./rackeval.js";
import { makeRng, type Rng } from "./rng.js";
import type { NativeDriver, NativeSettings } from "./sandbox-native.js";

// `native` drives the live engine; `profile:<bot>` bids from that bot's fitted
// public-data model; `field` draws a profile per game, weighted by matches.
export type Policy = "cheap" | "value" | "aggressive" | "rack" | "live-lite" | "planner" | "native" | "field" | `profile:${string}`;
const POLICIES = ["cheap", "value", "aggressive", "rack", "live-lite", "planner", "native", "field"];

export interface BidProfile {
  matches: number;
  a: number;
  b_face: number;
  c_vowel: number;
  d_late: number;
  e_endgame: number;
  sigma: number;
}

export const DEFAULT_PROFILES = path.resolve(path.dirname(new URL(import.meta.url).pathname), "../data/opponent-profiles.json");
const START_CASH = 100;
const START_RACK = 5;
const ALL_IN_ROUNDS = 5;
const ENDGAME_ROUNDS = 2;
const FLUSH_SIZE = 14;
const PRICE_BASE = 8;
const PRICE_SLOPE = 2.5;

interface Player {
  seat: number;
  policy: Policy;
  profile?: string;
  rack: Uint8Array;
  start: Uint8Array;
  // Won letters not yet seen in a word, and starting tiles a word revealed:
  // what the other seats can infer, as the live belief state does.
  known: Uint8Array;
  revealed: Uint8Array;
  balance: number;
  score: number;
  spent: number;
  spentByQuarter: number[];
  won: number;
  wordCount: number;
  wordRewards: number;
}

interface Scenario {
  letters: number[];
  jitters: number[][];
  tieU: number[];
  currentTieSeed: number;
}

export interface SandboxOptions {
  games: number;
  seed: number;
  seats: Policy[];
  rollouts: number;
  horizon: number;
  bidStep: number;
  bidScale: number;
  minCoinValue: number;
  earlyBidCap: number;
  earlyBidAuctions: number;
  auction: "vickrey" | "first-price";
  native: NativeSettings;
  profiles: string;
  // Share of profile bid noise common to all seats in an auction: bots value
  // the same letter in the same state alike, which tightens the runner-up gap.
  fieldCorr: number;
  // Calibration of profile bids to live prices: added to every latent bid,
  // and a multiplier on each profile's noise.
  fieldShift: number;
  fieldSigma: number;
  // Weight on the letter's exact marginal value to the seat's own rack. The
  // fitted profiles are rack-blind; real bots chase letters their words need.
  fieldRack: number;
  out?: string;
}

function intArg(args: string[], name: string, fallback: number): number {
  const i = args.indexOf(name);
  if (i < 0) return fallback;
  const n = Number(args[i + 1]);
  if (!Number.isFinite(n) || n < 1) throw new Error(`${name} needs a positive number`);
  return Math.floor(n);
}

function numberArg(args: string[], name: string, fallback: number): number {
  const i = args.indexOf(name);
  if (i < 0) return fallback;
  const n = Number(args[i + 1]);
  if (!Number.isFinite(n)) throw new Error(`${name} needs a number`);
  return n;
}

export function parseArgs(args: string[]): SandboxOptions {
  const seatArg = args.includes("--seats") ? args[args.indexOf("--seats") + 1] : undefined;
  const seats = (seatArg ?? "live-lite,planner,rack,aggressive,value,cheap")
    .split(",").map((s) => s.trim()) as Policy[];
  if (seats.some((p) => !POLICIES.includes(p) && !p.startsWith("profile:")))
    throw new Error(`Unknown policy in --seats. Choose: ${POLICIES.join(", ")}, profile:<bot>`);
  if (seats.length < 2 || seats.length > 12) throw new Error("--seats needs 2–12 comma-separated policies");
  const auctionArg = args.includes("--auction") ? args[args.indexOf("--auction") + 1] : "vickrey";
  if (auctionArg !== "vickrey" && auctionArg !== "first-price")
    throw new Error("--auction must be vickrey or first-price");
  const outArg = args.includes("--out") ? args[args.indexOf("--out") + 1] : undefined;
  const seedArg = args.includes("--seed") ? Number(args[args.indexOf("--seed") + 1]) : 42;
  if (!Number.isFinite(seedArg)) throw new Error("--seed needs a number");
  const bidScale = numberArg(args, "--bid-scale", 0.8);
  const minCoinValue = numberArg(args, "--coin-floor", 0.35);
  const earlyBidCap = numberArg(args, "--early-cap", 35);
  const earlyBidAuctions = numberArg(args, "--early-auctions", 10);
  if (bidScale < 0 || bidScale > 1) throw new Error("--bid-scale must be from 0 to 1");
  if (minCoinValue < 0.05 || minCoinValue > 1) throw new Error("--coin-floor must be from 0.05 to 1");
  if (earlyBidCap < 0 || earlyBidAuctions < 0) throw new Error("early bid limits cannot be negative");
  return {
    games: intArg(args, "--games", 12),
    seed: seedArg >>> 0,
    seats,
    rollouts: intArg(args, "--rollouts", 6),
    horizon: intArg(args, "--horizon", 12),
    bidStep: intArg(args, "--bid-step", 5),
    bidScale,
    minCoinValue,
    earlyBidCap,
    earlyBidAuctions: Math.floor(earlyBidAuctions),
    auction: auctionArg,
    native: {
      paidBid: numberArg(args, "--native-paid-bid", 1) !== 0,
      oppBonus: numberArg(args, "--opp-bonus", 8),
      oppSlope: numberArg(args, "--opp-slope", 2.5),
      oppJitter: numberArg(args, "--opp-jitter", 4),
      threads: intArg(args, "--native-threads", 2),
    },
    fieldCorr: numberArg(args, "--field-corr", 0),
    fieldShift: numberArg(args, "--field-shift", 0),
    fieldSigma: numberArg(args, "--field-sigma", 1),
    fieldRack: numberArg(args, "--field-rack", 1),
    profiles: args.includes("--profiles") ? path.resolve(args[args.indexOf("--profiles") + 1]!) : DEFAULT_PROFILES,
    out: outArg,
  };
}

function shuffle<T>(a: T[], rng: Rng): void {
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [a[i], a[j]] = [a[j]!, a[i]!];
  }
}

function total(rack: Uint8Array): number {
  let n = 0;
  for (const x of rack) n += x;
  return n;
}

function withLetter(rack: Uint8Array, letter: number): Uint8Array {
  const out = rack.slice();
  out[letter]++;
  return out;
}

function removeWord(rack: Uint8Array, word: string): void {
  for (const c of word) rack[c.charCodeAt(0) - A]--;
}

function wordReward(word: string): number {
  let base = 0;
  for (const c of word) base += LETTER_VALUE[c.charCodeAt(0) - A]!;
  if (word.length <= 3) return base;
  if (word.length === 4) return Math.floor(base * 3 / 2);
  if (word.length === 5) return base * 2;
  if (word.length === 6) return Math.floor(base * 5 / 2);
  return base * 3;
}

function cashWord(p: Player, word: string): void {
  const reward = wordReward(word);
  removeWord(p.rack, word);
  for (const c of word) {
    const l = c.charCodeAt(0) - A;
    if (p.known[l]! > 0) p.known[l]!--;
    else p.revealed[l]!++;
  }
  p.balance += reward;
  p.score += reward;
  p.wordCount++;
  p.wordRewards += reward;
}

function playFaceWord(p: Player, byLength: string[]): void {
  const key = RackEval.keyOf(p.rack);
  const cached = longestCache.get(key);
  const word = cached === undefined
    ? byLength.find((w) => w.length >= 3 && fits(w, p.rack)) ?? null
    : cached;
  longestCache.set(key, word);
  if (word) cashWord(p, word);
}

const longestCache = new Map<string, string | null>();

function fits(word: string, rack: Uint8Array): boolean {
  const need = new Uint8Array(26);
  for (const c of word) {
    const i = c.charCodeAt(0) - A;
    if (i < 0 || i >= 26 || ++need[i]! > rack[i]!) return false;
  }
  return true;
}

function applyStrategicWords(p: Player, ev: RackEval, final = false): void {
  // Simulate the live/native continuation rule: cash long words; flush a
  // crowded rack; dump every remaining word at match end.
  for (;;) {
    const plan = ev.plan(p.rack, 16, 30_000);
    if (!plan.words.length) return;
    const choices = plan.words
      .map((word) => ({ word, reward: wordReward(word) }))
      .sort((a, b) => b.reward - a.reward);
    const chosen = choices.find((x) => final || x.word.length >= 7 || total(p.rack) > FLUSH_SIZE);
    if (!chosen) return;
    cashWord(p, chosen.word);
  }
}

function fixedBid(policy: Policy, letter: number, balance: number, jitter: number): number {
  const face = LETTER_VALUE[letter]!;
  let base: number;
  if (policy === "cheap") base = face - 1;
  else if (policy === "value") base = face;
  else if (policy === "aggressive") base = face + ([0, 4, 8, 14, 20].includes(letter) ? 4 : 2);
  else return 0;
  return Math.max(0, Math.min(balance, Math.max(1, base + jitter)));
}

function unseenCounts(letters: number[]): Uint8Array {
  const counts = new Uint8Array(26);
  for (const l of letters) counts[l]++;
  return counts;
}

function rackValues(p: Player, ev: RackEval): Float64Array {
  const base = ev.best(p.rack, 16);
  const values = new Float64Array(26);
  for (let l = 0; l < 26; l++) {
    const next = p.rack.slice();
    next[l]++;
    values[l] = Math.max(0, ev.best(next, 16) - base);
  }
  return values;
}

function pacedRackBid(
  p: Player,
  letter: number,
  lettersLeft: number[],
  round: number,
  auctionsLeft: number,
  ev: RackEval,
  policy: Policy,
  options: SandboxOptions,
): number {
  if (p.balance <= 0 || auctionsLeft <= 1) return 0;
  const v = rackValues(p, ev);
  const prices = Float64Array.from(LETTER_VALUE, (x) => Math.max(1, PRICE_BASE + PRICE_SLOPE * x));
  const lam = Math.max(options.minCoinValue,
    pacedLambda(v, unseenCounts(lettersLeft), p.balance, auctionsLeft - 1, prices));
  let bid = Math.floor(v[letter]! / lam);
  if (auctionsLeft - 1 < ALL_IN_ROUNDS && v[letter]! > 0) bid = p.balance;
  else bid = Math.floor(bid * (policy === "live-lite" ? options.bidScale : 0.9));
  if (policy === "live-lite" && round <= options.earlyBidAuctions)
    bid = Math.min(bid, options.earlyBidCap);
  return Math.max(0, Math.min(p.balance, bid));
}

const profileCache = new Map<string, Record<string, BidProfile>>();

export function loadProfiles(file: string): Record<string, BidProfile> {
  let out = profileCache.get(file);
  if (!out) {
    out = JSON.parse(fs.readFileSync(file, "utf8")).profiles as Record<string, BidProfile>;
    profileCache.set(file, out);
  }
  return out;
}

// A fitted latent bid, capped at the balance. `z` is a standard normal draw.
function profileBid(
  prof: BidProfile,
  letter: number,
  balance: number,
  round: number,
  auctions: number,
  auctionsLeft: number,
  z: number,
): number {
  if (auctionsLeft <= 1 || balance <= 0) return 0; // the final letter cannot be played
  const vowel = [0, 4, 8, 14, 20].includes(letter) ? 1 : 0;
  const y = prof.a + prof.b_face * LETTER_VALUE[letter]! + prof.c_vowel * vowel +
    prof.d_late * (round - 1) / auctions + prof.e_endgame * (auctionsLeft - 1 < 6 ? 1 : 0) + prof.sigma * z;
  return Math.max(0, Math.min(balance, Math.round(y)));
}

function gaussian(rng: Rng): number {
  const u = Math.max(1e-12, rng());
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rng());
}

function isFace(p: Player): boolean {
  return p.policy === "cheap" || p.policy === "value" || p.policy === "aggressive";
}

function bidFor(
  p: Player,
  letter: number,
  remaining: number[],
  round: number,
  auctionsLeft: number,
  ev: RackEval,
  jitter: number,
  options: SandboxOptions,
  z = 0,
  auctions = 0,
): number {
  if (isFace(p)) return fixedBid(p.policy, letter, p.balance, jitter);
  if (p.profile) {
    const prof = loadProfiles(options.profiles)[p.profile]!;
    const need = options.fieldRack > 0 && auctionsLeft > 1
      ? options.fieldRack * Math.max(0, ev.best(withLetter(p.rack, letter), 16) - ev.best(p.rack, 16))
      : 0;
    return profileBid({ ...prof, a: prof.a + options.fieldShift + need, sigma: prof.sigma * options.fieldSigma },
      letter, p.balance, round, auctions || round + auctionsLeft - 1, auctionsLeft, z);
  }
  return pacedRackBid(p, letter, remaining, round, auctionsLeft, ev,
    p.policy === "planner" || p.policy === "native" ? "live-lite" : p.policy, options);
}

function settleAuction(
  players: Player[],
  bids: number[],
  letter: number,
  rng: Rng,
  auction: SandboxOptions["auction"],
): number | null {
  const order = players.map((_, i) => i);
  shuffle(order, rng);
  order.sort((a, b) => bids[b]! - bids[a]!);
  const winner = order[0]!;
  const top = bids[winner]!;
  if (top <= 0) return null;
  const second = order.length > 1 ? bids[order[1]!]! : 0;
  const paid = auction === "first-price" ? top : (second > 0 ? second : 1);
  const p = players[winner]!;
  if (p.balance < paid) return null;
  p.balance -= paid;
  p.spent += paid;
  p.won++;
  p.rack[letter]++;
  p.known[letter]++;
  return winner;
}

function copyPlayers(players: Player[]): Player[] {
  return players.map((p) => ({ ...p, rack: p.rack.slice(), known: p.known.slice(), revealed: p.revealed.slice() }));
}

function makeScenarios(
  count: number,
  remaining: number[],
  players: number,
  horizon: number,
  rng: Rng,
): Scenario[] {
  const out: Scenario[] = [];
  for (let s = 0; s < count; s++) {
    const letters = remaining.slice();
    shuffle(letters, rng);
    const selected = letters.slice(0, horizon);
    out.push({
      letters: selected,
      jitters: selected.map(() => Array.from({ length: players }, () => Math.floor(rng() * 5) - 2)),
      tieU: selected.map(() => rng()),
      currentTieSeed: Math.floor(rng() * 0xffffffff),
    });
  }
  return out;
}

function plannerBid(
  plannerSeat: number,
  currentLetter: number,
  players: Player[],
  currentBids: number[],
  remaining: number[],
  round: number,
  auctionsLeft: number,
  ev: RackEval,
  rng: Rng,
  options: SandboxOptions,
  byLength: string[],
): number {
  const cap = players[plannerSeat]!.balance;
  if (cap <= 0 || auctionsLeft <= 1) return 0; // final draw is reserved for word play
  const candidates = new Set<number>([0, cap]);
  for (let b = options.bidStep; b < cap; b += options.bidStep) candidates.add(b);
  const scenarios = makeScenarios(options.rollouts, remaining, players.length,
    Math.min(options.horizon, Math.max(0, auctionsLeft - 1)), rng);
  let bestBid = 0;
  let bestValue = -Infinity;
  for (const candidate of candidates) {
    let sum = 0;
    for (const scenario of scenarios) {
      const simulated = copyPlayers(players);
      const firstBids = currentBids.slice();
      firstBids[plannerSeat] = candidate;
      const currentWinner = settleAuction(simulated, firstBids, currentLetter,
        makeRng(scenario.currentTieSeed), options.auction);
      for (let i = 0; i < simulated.length; i++) {
        const p = simulated[i]!;
        if (isFace(p)) {
          if (currentWinner === p.seat) playFaceWord(p, byLength);
        }
        else applyStrategicWords(p, ev);
      }
      for (let t = 0; t < scenario.letters.length; t++) {
        const l = scenario.letters[t]!;
        const rem = scenario.letters.slice(t + 1);
        const nextRound = round + t + 1;
        const left = auctionsLeft - t - 1;
        const bids = simulated.map((p, i) => bidFor(p, l, rem, nextRound,
          left, ev, scenario.jitters[t]![i]!, options));
        const futureWinner = settleAuction(simulated, bids, l, makeRng(Math.floor(scenario.tieU[t]! * 0xffffffff)), options.auction);
        for (const p of simulated) {
          if (isFace(p)) {
            if (futureWinner === p.seat) playFaceWord(p, byLength);
          }
          else applyStrategicWords(p, ev, left <= ENDGAME_ROUNDS);
        }
      }
      const me = simulated[plannerSeat]!;
      const rackPotential = ev.best(me.rack, 16);
      sum += me.score + rackPotential + me.balance * options.minCoinValue;
    }
    const value = sum / scenarios.length;
    // Prefer the cheaper action if rollout noise makes two bids look equal.
    if (value > bestValue + 1e-9 || (Math.abs(value - bestValue) <= 1e-9 && candidate < bestBid)) {
      bestValue = value;
      bestBid = candidate;
    }
  }
  return bestBid;
}

function initialPlayers(seats: Policy[]): Player[] {
  return seats.map((policy, seat) => ({
    seat, policy, rack: new Uint8Array(26), start: new Uint8Array(26),
    known: new Uint8Array(26), revealed: new Uint8Array(26), balance: START_CASH,
    score: 0, spent: 0, spentByQuarter: [0, 0, 0, 0], won: 0, wordCount: 0, wordRewards: 0,
  }));
}

function playAllFinal(players: Player[], ev: RackEval, byLength: string[]): void {
  for (const p of players) {
    if (isFace(p)) playFaceWord(p, byLength);
    else if (p.policy !== "native") applyStrategicWords(p, ev, true);
  }
}

interface NativeState {
  lastCashinKey: string | null;
  sinceCashin: number;
}

// One auction for a native seat, mirroring the live client (config at
// b10334e): cash-ins that beat holding by CASH_MARGIN outside the endgame
// window, then the engine's paid-tile fair bid, else v / lambda.
async function nativeBid(
  p: Player,
  players: Player[],
  letter: number,
  bagAfter: number,
  auctioned: Uint8Array,
  matchId: string,
  round: number,
  st: NativeState,
  driver: NativeDriver,
  ev: RackEval,
  options: SandboxOptions,
): Promise<number> {
  const cfg = driver.cfg;
  const usableLeft = Math.max(0, bagAfter - 1);
  const isFinal = bagAfter === 0;
  if (isFinal || shouldDumpPlan(usableLeft, false, cfg)) dumpPlan(p, ev);
  if (isFinal || p.balance < 1) return 0;
  st.sinceCashin++;
  let result = null;
  for (let pass = 0; pass < 4; pass++) {
    const rackKey = RackEval.keyOf(p.rack);
    const needCashin = rackKey !== st.lastCashinKey || st.sinceCashin >= cfg.cashinEvery;
    const U = new Uint8Array(26);
    for (let l = 0; l < 26; l++) {
      let u = BAG[l]! - p.start[l]! - auctioned[l]!;
      for (const q of players) if (q !== p) u -= q.revealed[l]!;
      U[l] = Math.max(0, u);
    }
    result = await driver.evaluate({
      matchId, auctionId: String(round), letter, isFinal, usableLeft, bagTotal: bagAfter,
      firstPrice: options.auction === "first-price", balance: p.balance, score: p.score,
      rack: p.rack.slice(), U, rackKey, needCashin,
      opps: players.filter((q) => q !== p).map((q) => ({ botId: String(q.seat), balance: q.balance, kind: "Real" as const })),
    });
    if (!needCashin) break;
    st.lastCashinKey = rackKey;
    st.sinceCashin = 0;
    const best = result.cashin[0];
    if (shouldDumpPlan(usableLeft, false, cfg) || !best || best.gain <= cfg.cashMargin || !fits(best.word, p.rack)) break;
    cashWord(p, best.word);
  }
  if (!result) return 0;
  if (cfg.nativePaidBid && result.fairBid !== null) return Math.max(0, Math.min(p.balance, Math.floor(result.fairBid)));
  return bidAmount(result.v[letter]!, p.balance, usableLeft, cfg, undefined, result.lam).amount;
}

function dumpPlan(p: Player, ev: RackEval): void {
  if (total(p.rack) < 2) return;
  for (const word of ev.plan(p.rack, 22, 400_000).words) if (fits(word, p.rack)) cashWord(p, word);
}

export async function runMatch(
  gameSeed: number,
  options: SandboxOptions,
  ev: RackEval,
  byLength: string[],
  driver?: NativeDriver,
  tag = "",
): Promise<Record<string, unknown>> {
  const rng = makeRng(gameSeed);
  // Keep planner sampling independent of the real deck/tie stream so changes
  // to rollout settings remain paired on the same game seed.
  const plannerRng = makeRng((gameSeed ^ 0xa5a5a5a5) >>> 0);
  // Profile bid noise has its own stream, drawn for every seat every round.
  const profileRng = makeRng((gameSeed ^ 0x0ddba11) >>> 0);
  // Seat order decides who is dealt which starting rack. A separate stream
  // keeps the deck unchanged and gives paired runs the same assignment.
  const seatRng = makeRng((gameSeed ^ 0x5eed5eed) >>> 0);
  const seatOrder = options.seats.slice();
  shuffle(seatOrder, seatRng);
  const bag: number[] = [];
  for (let l = 0; l < BAG.length; l++) for (let n = 0; n < BAG[l]!; n++) bag.push(l);
  shuffle(bag, rng);
  const players = initialPlayers(seatOrder);
  if (players.some((p) => p.policy === "field" || p.policy.startsWith("profile:"))) {
    const pool = Object.entries(loadProfiles(options.profiles));
    const weight = pool.reduce((sum, [, v]) => sum + v.matches, 0);
    for (const p of players) {
      if (p.policy.startsWith("profile:")) p.profile = p.policy.slice(8);
      else if (p.policy === "field") {
        let u = seatRng() * weight;
        p.profile = pool.find(([, v]) => (u -= v.matches) < 0)?.[0] ?? pool[0]![0];
      }
      if (p.profile && !loadProfiles(options.profiles)[p.profile]) throw new Error(`no profile for ${p.profile}`);
    }
  }
  if (players.some((p) => p.policy === "native") && !driver) throw new Error("native seat needs a NativeDriver");
  let cursor = 0;
  for (const p of players) for (let n = 0; n < START_RACK; n++) p.rack[bag[cursor++]! ]++;
  for (const p of players) p.start.set(p.rack);
  const remaining = bag.slice(cursor);
  const auctions = remaining.length;
  const auctioned = new Uint8Array(26);
  const matchId = `${tag}${gameSeed}`;
  const nativeState = players.map((): NativeState => ({ lastCashinKey: null, sinceCashin: 0 }));
  const auctionLog: { letter: string; winner: number | null; paid: number; top: number }[] = [];
  longestCache.clear();
  for (let round = 1; round <= auctions; round++) {
    const letter = remaining.shift()!;
    auctioned[letter]++;
    const auctionsLeft = auctions - round + 1;
    const jitter = players.map(() => Math.floor(rng() * 5) - 2);
    const common = gaussian(profileRng);
    const rho = options.fieldCorr;
    const z = players.map(() => rho * common + Math.sqrt(1 - rho * rho) * gaussian(profileRng));
    const bids = players.map((p, i) => bidFor(p, letter, remaining, round,
      auctionsLeft, ev, jitter[i]!, options, z[i]!, auctions));
    for (let i = 0; i < players.length; i++) {
      const p = players[i]!;
      if (p.policy === "native") {
        bids[i] = await nativeBid(p, players, letter, remaining.length, auctioned, matchId,
          round, nativeState[i]!, driver!, ev, options);
      }
    }
    for (let i = 0; i < players.length; i++) {
      if (players[i]!.policy !== "planner") continue;
      bids[i] = plannerBid(i, letter, players, bids, remaining, round,
        auctionsLeft, ev, plannerRng, options, byLength);
    }
    const spentBefore = players.map((p) => p.spent);
    const winner = settleAuction(players, bids, letter, rng, options.auction);
    auctionLog.push({
      letter: String.fromCharCode(A + letter), winner,
      paid: winner === null ? 0 : players[winner]!.spent - spentBefore[winner]!, top: Math.max(...bids),
    });
    const quarter = Math.min(3, Math.floor(4 * (round - 1) / auctions));
    for (let i = 0; i < players.length; i++)
      players[i]!.spentByQuarter[quarter]! += players[i]!.spent - spentBefore[i]!;
    for (const p of players) {
      if (isFace(p)) {
        if (winner === p.seat) playFaceWord(p, byLength);
      } else if (p.policy !== "native") applyStrategicWords(p, ev, auctionsLeft <= ENDGAME_ROUNDS);
    }
  }
  // The server ends the match when the last auction settles, so nobody plays
  // after it: the native seat's final dump already ran during that auction.
  playAllFinal(players, ev, byLength);
  const maxScore = Math.max(...players.map((p) => p.score));
  const standings = players.map((p) => ({
    seat: p.seat,
    policy: p.policy,
    profile: p.profile ?? null,
    score: p.score,
    place: 1 + players.filter((q) => q.score > p.score).length,
    wonMatch: p.score === maxScore,
    lastPlace: !players.some((q) => q.score < p.score),
    balance: p.balance,
    spent: p.spent,
    spentByQuarter: p.spentByQuarter,
    tilesWon: p.won,
    words: p.wordCount,
    wordRewards: p.wordRewards,
    rack: RackEval.keyOf(p.rack),
    playableRackValue: ev.best(p.rack, 16),
  }));
  return { seed: gameSeed, auctions, auctionType: options.auction, standings, auctionLog };
}

export function summarize(games: Record<string, unknown>[], seats: Policy[]) {
  const rows = games.flatMap((g) => g.standings as Record<string, unknown>[]);
  return [...new Set(seats)].map((policy) => {
    const ps = rows.filter((r) => r.policy === policy);
    const avg = (k: string) => ps.reduce((s, r) => s + Number(r[k]), 0) / Math.max(1, ps.length);
    return {
      policy,
      matches: ps.length,
      wins: ps.filter((r) => r.wonMatch).length,
      winRate: ps.filter((r) => r.wonMatch).length / Math.max(1, ps.length),
      lastRate: ps.filter((r) => r.lastPlace).length / Math.max(1, ps.length),
      averagePlace: avg("place"),
      averageScore: avg("score"),
      averageBalance: avg("balance"),
      averageSpent: avg("spent"),
      averageTilesWon: avg("tilesWon"),
      averageWords: avg("words"),
      averagePlayableRackValue: avg("playableRackValue"),
    };
  });
}

export async function runSandbox(options: SandboxOptions) {
  const dict = Dict.load();
  const ev = new RackEval(dict, 16);
  const byLength = [...dict.allWords].sort((a, b) => b.length - a.length || a.localeCompare(b));
  const games: Record<string, unknown>[] = [];
  const driver = options.seats.includes("native")
    ? new (await import("./sandbox-native.js")).NativeDriver((await import("./sandbox-native.js")).sandboxConfig(options.native))
    : undefined;
  try {
    for (let i = 0; i < options.games; i++) {
      const seed = (options.seed + Math.imul(i, 0x9e3779b9)) >>> 0;
      games.push(await runMatch(seed, options, ev, byLength, driver));
    }
  } finally {
    driver?.close();
  }
  const result = {
    description: "Offline seeded game. planner is a short-horizon Monte Carlo best-response benchmark, not exact optimal play.",
    options,
    summary: summarize(games, options.seats),
    games,
  };
  if (options.out) {
    const outfile = path.resolve(options.out);
    fs.mkdirSync(path.dirname(outfile), { recursive: true });
    fs.writeFileSync(outfile, `${JSON.stringify(result, null, 2)}\n`);
  }
  return result;
}

function format(result: Awaited<ReturnType<typeof runSandbox>>): string {
  const lines = [
    `Offline sandbox: ${result.options.games} games, seed ${result.options.seed}, ${result.options.seats.length} seats, ${result.options.auction}`,
    "policy      win%  last%   avg place   avg score   cash left   tiles   words   rack value",
  ];
  for (const r of result.summary) {
    lines.push(
      `${String(r.policy).padEnd(11)} ${(100 * Number(r.winRate)).toFixed(1).padStart(5)}%` +
      ` ${(100 * Number(r.lastRate)).toFixed(1).padStart(5)}%` +
      ` ${Number(r.averagePlace).toFixed(2).padStart(10)}` +
      ` ${Number(r.averageScore).toFixed(1).padStart(11)}` +
      ` ${Number(r.averageBalance).toFixed(1).padStart(11)}` +
      ` ${Number(r.averageTilesWon).toFixed(1).padStart(7)}` +
      ` ${Number(r.averageWords).toFixed(1).padStart(7)}` +
      ` ${Number(r.averagePlayableRackValue).toFixed(1).padStart(12)}`,
    );
  }
  if (result.options.out) lines.push(`wrote ${path.resolve(result.options.out)}`);
  return lines.join("\n");
}

if (import.meta.url === pathToFileURL(process.argv[1] ?? "").href) {
  runSandbox(parseArgs(process.argv.slice(2))).then((result) => console.log(format(result)), (e) => {
    console.error(e instanceof Error ? e.message : String(e));
    process.exitCode = 1;
  });
}
