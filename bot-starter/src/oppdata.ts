// Load public match histories (the JSON written by the public-SQL fetch
// scripts: match_participant, auction_result, auction_top_bid_archive,
// word_play, match_state) and replay each match into per-auction rows with
// every bot's reconstructed balance and known rack before the auction.
//
// Offline only (oppfit.ts); the live bot never reads these files.

import * as fs from "node:fs";
import * as path from "node:path";
import { A } from "./dict.js";

export interface AuctionObs {
  i: number; // auction index in the match, 0-based
  left: number; // auctions after this one
  L: number; // letter index
  winner: number; // seat index of the winner, -1 if nobody bid
  top: number; // winner's bid (archived top bid), 0 if none
  paid: number; // second-highest bid, or the reserve (1) if only one bid
  bal: Int32Array; // per seat, before this auction (upper bound: includes words played before close)
  known: Uint8Array[]; // per seat: letters won and not yet played (starting rack is hidden)
}

export interface Play {
  seat: number;
  word: string;
  reward: number;
  auctionIdx: number; // number of auctions closed before this play
}

export interface MatchObs {
  id: number;
  createdAt: number;
  bots: number[]; // bot id per seat
  finalScore: number[];
  finalBalance: number[];
  auctions: AuctionObs[];
  plays: Play[];
  balanceMismatch: number; // sum |reconstructed - final| over seats
}

const some = (v: unknown): number | null =>
  Array.isArray(v) && v[0] === 0 ? (Array.isArray(v[1]) ? Number(v[1][0]) : Number(v[1])) : null;

interface RawMatch {
  match_participant: { bot_id: number; balance: number; score: number }[];
  auction_result: { auction_id: number; letter: string; winner_bot_id: unknown; paid: number; closed_at: number }[];
  auction_top_bid_archive: { auction_id: number; top_bid: number }[];
  word_play: { bot_id: number; word: string; total_reward: number; played_at: number }[];
  match_state: { id: number; created_at: number }[];
}

export function parseMatch(raw: RawMatch): MatchObs | null {
  const st = raw.match_state?.[0];
  if (!st || raw.auction_result.length < 60 || raw.match_participant.length < 2) return null;
  const bots = raw.match_participant.map((p) => p.bot_id);
  const seatOf = new Map<number, number>(bots.map((b, s) => [b, s] as const));
  const top = new Map<number, number>(raw.auction_top_bid_archive.map((a) => [a.auction_id, a.top_bid] as const));
  const res = [...raw.auction_result].sort((a, b) => a.closed_at - b.closed_at);
  const plays = [...raw.word_play].sort((a, b) => a.played_at - b.played_at);
  const n = bots.length;
  const bal = new Int32Array(n).fill(100);
  const known = bots.map(() => new Uint8Array(26));
  const auctions: AuctionObs[] = [];
  const outPlays: Play[] = [];
  let wi = 0;
  const applyPlay = (p: (typeof plays)[number], idx: number) => {
    const s = seatOf.get(p.bot_id);
    if (s === undefined) return;
    bal[s] += p.total_reward;
    for (const ch of p.word) {
      const L = ch.charCodeAt(0) - A;
      if (known[s][L] > 0) known[s][L]--;
    }
    outPlays.push({ seat: s, word: p.word, reward: p.total_reward, auctionIdx: idx });
  };
  for (let i = 0; i < res.length; i++) {
    const r = res[i];
    while (wi < plays.length && plays[wi].played_at < r.closed_at) applyPlay(plays[wi++], i);
    const w = some(r.winner_bot_id);
    const ws = w === null ? -1 : (seatOf.get(w) ?? -1);
    const t = top.get(r.auction_id) ?? 0;
    auctions.push({
      i,
      left: res.length - 1 - i,
      L: r.letter.charCodeAt(0) - A,
      winner: ws,
      top: ws >= 0 ? Math.max(t, r.paid) : 0,
      paid: r.paid,
      bal: bal.slice(),
      known: known.map((k) => k.slice()),
    });
    if (ws >= 0 && bal[ws] >= r.paid) {
      bal[ws] -= r.paid;
      known[ws][r.letter.charCodeAt(0) - A]++;
    }
  }
  while (wi < plays.length) applyPlay(plays[wi++], res.length);
  let mismatch = 0;
  raw.match_participant.forEach((p, s) => (mismatch += Math.abs(p.balance - bal[s])));
  return {
    id: st.id,
    createdAt: st.created_at,
    bots,
    finalScore: raw.match_participant.map((p) => p.score),
    finalBalance: raw.match_participant.map((p) => p.balance),
    auctions,
    plays: outPlays,
    balanceMismatch: mismatch,
  };
}

// Every *.json in the given directories, deduplicated by match id, oldest first.
export function loadMatches(dirs: string[]): MatchObs[] {
  const byId = new Map<number, MatchObs>();
  for (const dir of dirs)
    for (const f of fs.readdirSync(dir)) {
      if (!f.endsWith(".json")) continue;
      const m = parseMatch(JSON.parse(fs.readFileSync(path.join(dir, f), "utf8")) as RawMatch);
      if (m && !byId.has(m.id)) byId.set(m.id, m);
    }
  return [...byId.values()].sort((a, b) => a.createdAt - b.createdAt || a.id - b.id);
}

export function loadBotNames(file: string): Map<number, string> {
  const rows = JSON.parse(fs.readFileSync(file, "utf8")) as { id: number; name: string }[];
  return new Map<number, string>(rows.map((r) => [r.id, r.name] as const));
}
