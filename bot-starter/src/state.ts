// Mirror of the subscribed tables (indexed by match, so history from old
// matches never slows a lookup) and the belief state derived from it.
//
// The belief is a pure function of the tables: it is recomputed from scratch
// on every change, replaying the match in order. That makes it immune to the
// order callbacks fire in (a sim's word play lands in the same transaction as
// the auction result that gave it the letters) and correct after a restart in
// the middle of a match.

import { A, BAG, LETTER_VALUE } from "./dict.js";
import type { OppKind, Opp } from "./simmodel.js";

interface Ts {
  microsSinceUnixEpoch: bigint;
}
export interface AuctionRow {
  id: bigint;
  matchId: bigint;
  letter: string;
  opensAt: Ts;
  closesAt: Ts;
  status: { tag: string };
}
export interface ResultRow {
  auctionId: bigint;
  matchId: bigint;
  letter: string;
  winnerBotId?: bigint | null;
  paid: bigint;
  closedAt: Ts;
}
export interface PlayRow {
  id: bigint;
  matchId: bigint;
  botId: bigint;
  word: string;
  totalReward: bigint;
  playedAt: Ts;
}
export interface PartRow {
  id: bigint;
  matchId: bigint;
  botId: bigint;
  balance: bigint;
  score: bigint;
}
export interface MatchRow {
  id: bigint;
  status: { tag: string };
  currentRound: number;
  currentAuctionId?: bigint | null;
  bagTotal: number;
  auctionType: { tag: string };
  auctionDurationMs: bigint;
  startingRackSize: number;
}
export interface RackRow {
  id: bigint;
  matchId: bigint;
  botId: bigint;
  letter: string;
  count: number;
}
export interface BotRow {
  id: bigint;
  name: string;
  isSimulated: boolean;
  strategy: { tag: string };
}
export interface ArchiveRow {
  auctionId: bigint;
  matchId: bigint;
  topBid: bigint;
}

export class MatchData {
  auctions = new Map<bigint, AuctionRow>();
  results = new Map<bigint, ResultRow>();
  plays = new Map<bigint, PlayRow>();
  parts = new Map<bigint, PartRow>();
  rack = new Map<bigint, RackRow>();
}

export class Store {
  bots = new Map<bigint, BotRow>();
  matches = new Map<bigint, MatchRow>();
  private data = new Map<bigint, MatchData>();

  match(id: bigint): MatchData {
    let d = this.data.get(id);
    if (!d) {
      d = new MatchData();
      this.data.set(id, d);
    }
    return d;
  }

  clear() {
    this.bots.clear();
    this.matches.clear();
    this.data.clear();
  }

  // Drop the heavy per-match data once a match is done with.
  forget(id: bigint) {
    this.data.delete(id);
  }
}

const sum = (a: ArrayLike<number>) => {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i];
  return s;
};

export interface Belief {
  U: Uint8Array;
  myStart: Uint8Array;
  hidden: Map<bigint, number>;
  known: Map<bigint, Uint8Array>;
  mismatches: string[];
}

export interface BeliefInput {
  myBotId: bigint;
  rackSize: number;
  bagTotal: number;
  botIds: bigint[]; // every participant, me included
  auctions: Iterable<AuctionRow>;
  results: Iterable<ResultRow>;
  plays: Iterable<PlayRow>;
  myRack: Uint8Array;
}

export function computeBelief(inp: BeliefInput): Belief {
  const mismatches: string[] = [];
  const known = new Map<bigint, Uint8Array>();
  const hidden = new Map<bigint, number>();
  for (const b of inp.botIds) {
    known.set(b, new Uint8Array(26));
    hidden.set(b, inp.rackSize);
  }

  type Ev = { t: bigint; order: number; id: bigint; res?: ResultRow; play?: PlayRow };
  const events: Ev[] = [];
  for (const r of inp.results)
    events.push({ t: r.closedAt.microsSinceUnixEpoch, order: 0, id: r.auctionId, res: r });
  for (const p of inp.plays)
    events.push({ t: p.playedAt.microsSinceUnixEpoch, order: 1, id: p.id, play: p });
  events.sort((a, b) =>
    a.t < b.t ? -1 : a.t > b.t ? 1 : a.order - b.order || (a.id < b.id ? -1 : a.id > b.id ? 1 : 0),
  );

  // Pass 1: my own wins and plays, to recover the starting rack.
  const myWon = new Uint8Array(26);
  const myPlayed = new Uint8Array(26);
  for (const e of events) {
    if (e.res && e.res.winnerBotId === inp.myBotId) myWon[e.res.letter.charCodeAt(0) - A]++;
    if (e.play && e.play.botId === inp.myBotId)
      for (const ch of e.play.word) myPlayed[ch.charCodeAt(0) - A]++;
  }
  const myStart = new Uint8Array(26);
  for (let i = 0; i < 26; i++) {
    const v = inp.myRack[i] + myPlayed[i] - myWon[i];
    if (v < 0) mismatches.push(`myStart[${String.fromCharCode(A + i)}]=${v}`);
    myStart[i] = Math.max(0, v);
  }

  // Signed while counting down, so an underflow shows up as a mismatch.
  const U = new Int16Array(26);
  for (let i = 0; i < 26; i++) U[i] = BAG[i] - myStart[i];
  for (const a of inp.auctions) U[a.letter.charCodeAt(0) - A]--;

  // Pass 2: opponents. Letters played come out of known inventory first; any
  // remainder was a hidden starting tile, which is now revealed.
  for (const e of events) {
    if (e.res) {
      const w = e.res.winnerBotId;
      if (w !== undefined && w !== null && w !== inp.myBotId && known.has(w))
        known.get(w)![e.res.letter.charCodeAt(0) - A]++;
    } else if (e.play) {
      const x = e.play.botId;
      if (x === inp.myBotId || !known.has(x)) continue;
      const inv = known.get(x)!;
      for (const ch of e.play.word) {
        const L = ch.charCodeAt(0) - A;
        if (inv[L] > 0) inv[L]--;
        else {
          U[L]--;
          const h = hidden.get(x)! - 1;
          hidden.set(x, h);
          if (h < 0) mismatches.push(`bot ${x} hidden went negative`);
        }
      }
    }
  }
  const Uout = new Uint8Array(26);
  for (let i = 0; i < 26; i++) {
    if (U[i] < 0) mismatches.push(`U[${String.fromCharCode(A + i)}]=${U[i]}`);
    Uout[i] = Math.max(0, U[i]);
  }
  for (const [b, h] of hidden) if (h < 0) hidden.set(b, 0);

  let hid = 0;
  for (const [b, h] of hidden) if (b !== inp.myBotId) hid += h;
  const lhs = sum(Uout);
  const rhs = inp.bagTotal + hid;
  if (lhs !== rhs) mismatches.push(`invariant sum(U)=${lhs} != bagTotal+hidden=${rhs}`);

  return { U: Uout, myStart, hidden, known, mismatches };
}

// ---- Snapshot handed to the engine worker ---------------------------------

export interface Snapshot {
  seq: number;
  matchId: string;
  auctionId: string | null;
  letter: number; // letter index of the open auction, -1 if none
  isFinal: boolean;
  usableLeft: number;
  bagTotal: number;
  firstPrice: boolean;
  balance: number;
  score: number;
  rack: Uint8Array;
  U: Uint8Array;
  opps: Opp[];
  rackKey: string;
  needCashin: boolean;
}

export function oppKind(bot: BotRow | undefined): OppKind {
  if (!bot || !bot.isSimulated) return "Real";
  switch (bot.strategy.tag) {
    case "Cheapskate":
      return "Cheapskate";
    case "ValueBidder":
      return "ValueBidder";
    case "Aggressive":
      return "Aggressive";
    default:
      return "Real";
  }
}

export function rackCounts(d: MatchData, botId: bigint): Uint8Array {
  const c = new Uint8Array(26);
  for (const r of d.rack.values()) {
    if (r.botId !== botId) continue;
    const i = r.letter.charCodeAt(0) - A;
    if (i >= 0 && i < 26) c[i] += r.count;
  }
  return c;
}

export function usableLeftOf(bagTotal: number): number {
  return Math.max(0, bagTotal - 1);
}

export interface Derived {
  match: MatchRow;
  data: MatchData;
  me: PartRow;
  belief: Belief;
  rack: Uint8Array;
  openAuction: AuctionRow | null;
  usableLeft: number;
  isFinal: boolean;
}

// Everything the policy needs, or null if the match isn't fully mirrored yet.
export function derive(store: Store, matchId: bigint, myBotId: bigint): Derived | null {
  const match = store.matches.get(matchId);
  if (!match) return null;
  const data = store.match(matchId);
  let me: PartRow | undefined;
  for (const p of data.parts.values()) if (p.botId === myBotId) me = p;
  if (!me) return null;
  const rack = rackCounts(data, myBotId);
  const botIds = [...data.parts.values()].map((p) => p.botId);
  const belief = computeBelief({
    myBotId,
    rackSize: match.startingRackSize,
    bagTotal: match.bagTotal,
    botIds,
    auctions: data.auctions.values(),
    results: data.results.values(),
    plays: data.plays.values(),
    myRack: rack,
  });
  let openAuction: AuctionRow | null = null;
  if (match.currentAuctionId !== undefined && match.currentAuctionId !== null) {
    const a = data.auctions.get(match.currentAuctionId);
    if (a && a.status.tag === "Open") openAuction = a;
  }
  return {
    match,
    data,
    me,
    belief,
    rack,
    openAuction,
    usableLeft: usableLeftOf(match.bagTotal),
    isFinal: match.bagTotal === 0 && openAuction !== null,
  };
}

let seqCounter = 0;

export function buildSnapshot(
  store: Store,
  d: Derived,
  myBotId: bigint,
  needCashin: boolean,
  rackKey: string,
): Snapshot {
  const opps: Opp[] = [];
  const roundsSeen = Math.max(1, d.match.currentRound - 1);
  for (const p of d.data.parts.values()) {
    if (p.botId === myBotId) continue;
    const incomePerAuction = Math.max(0, Math.min(3,
      (Number(p.score) + 1.2 * 8) / (roundsSeen + 8),
    ));
    opps.push({
      botId: String(p.botId),
      balance: Number(p.balance),
      kind: oppKind(store.bots.get(p.botId)),
      incomePerAuction,
    });
  }
  return {
    seq: ++seqCounter,
    matchId: String(d.match.id),
    auctionId: d.openAuction ? String(d.openAuction.id) : null,
    letter: d.openAuction ? d.openAuction.letter.charCodeAt(0) - A : -1,
    isFinal: d.isFinal,
    usableLeft: d.usableLeft,
    bagTotal: d.match.bagTotal,
    firstPrice: d.match.auctionType.tag === "FirstPrice",
    balance: Number(d.me.balance),
    score: Number(d.me.score),
    rack: d.rack,
    U: d.belief.U,
    opps,
    rackKey,
    needCashin,
  };
}

export { LETTER_VALUE };
