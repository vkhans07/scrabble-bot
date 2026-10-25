// B(rack): the best total reward from any set of disjoint words in the rack.
//
//   B(r) = max( B(r - e_i), max over keys k <= r with k[i] >= 1 of reward(k) + B(r - k) )
//
// where i is the lowest letter present. Keys are enumerated by walking the
// dictionary trie, so only prefixes that begin real words are visited.
// Memoised on the sorted-letter string, which is the same information as the
// 26 counts but shorter to build.

import { A, Dict } from "./dict.js";

class BudgetExceeded extends Error {}

export interface Plan {
  value: number;
  words: string[];
  exact: boolean;
}

export class RackEval {
  memo = new Map<string, number>();
  hits = 0;
  misses = 0;
  greedyCount = 0;
  exactCount = 0;
  private ops = 0;
  private opLimit = Infinity;
  private static readonly MEMO_MAX = 3_000_000;

  constructor(
    readonly dict: Dict,
    readonly exactCap = 14,
  ) {}

  static total(c: Uint8Array): number {
    let n = 0;
    for (let i = 0; i < 26; i++) n += c[i];
    return n;
  }

  static keyOf(c: Uint8Array): string {
    let s = "";
    for (let i = 0; i < 26; i++) {
      const n = c[i];
      if (n) s += String.fromCharCode(A + i).repeat(n);
    }
    return s;
  }

  // Value of a rack. Exact up to exactCap tiles (or `cap` if given), greedy
  // beyond. `c` is restored before returning.
  best(c: Uint8Array, cap = this.exactCap): number {
    const n = RackEval.total(c);
    if (n < 2) return 0;
    if (n > cap) {
      this.greedyCount++;
      return this.greedy(c).value;
    }
    this.exactCount++;
    this.opLimit = Infinity;
    return this.exact(c);
  }

  // Exact with a work limit; falls back to greedy when it runs out.
  bestBudgeted(c: Uint8Array, maxOps: number): { value: number; exact: boolean } {
    const n = RackEval.total(c);
    if (n < 2) return { value: 0, exact: true };
    this.ops = 0;
    this.opLimit = maxOps;
    try {
      // A throw mid-walk leaves the counts half-consumed, so never hand it
      // the caller's array.
      const v = this.exact(c.slice());
      this.exactCount++;
      return { value: v, exact: true };
    } catch (e) {
      if (!(e instanceof BudgetExceeded)) throw e;
      this.greedyCount++;
      return { value: this.greedy(c).value, exact: false };
    } finally {
      this.opLimit = Infinity;
    }
  }

  private exact(c: Uint8Array): number {
    let i = 0;
    while (i < 26 && c[i] === 0) i++;
    if (i === 26) return 0;
    const key = RackEval.keyOf(c);
    const m = this.memo.get(key);
    if (m !== undefined) {
      this.hits++;
      return m;
    }
    this.misses++;
    if (key.length === 1) {
      this.memo.set(key, 0);
      return 0;
    }
    const { dict } = this;
    c[i]--;
    let best = this.exact(c);
    c[i]++;
    dict.forEachFitting(c, i, (e) => {
      if (++this.ops > this.opLimit) throw new BudgetExceeded();
      // forEachFitting has already removed the key's letters from c.
      const v = dict.rewards[e] + this.exact(c);
      if (v > best) best = v;
    });
    if (this.memo.size >= RackEval.MEMO_MAX) this.memo.clear();
    this.memo.set(key, best);
    return best;
  }

  // Exact plan, or greedy when the rack is above the cap.
  plan(c: Uint8Array, cap = this.exactCap, maxOps = Infinity): Plan {
    const n = RackEval.total(c);
    if (n < 2) return { value: 0, words: [], exact: true };
    if (n > cap) {
      this.greedyCount++;
      return { ...this.greedy(c), exact: false };
    }
    const r = this.bestBudgeted(c, maxOps);
    if (!r.exact) return { ...this.greedy(c), exact: false };
    const work = c.slice();
    const words: string[] = [];
    const { dict } = this;
    for (;;) {
      let i = 0;
      while (i < 26 && work[i] === 0) i++;
      if (i === 26) break;
      work[i]--;
      let best = this.exact(work);
      work[i]++;
      let pick = -1;
      dict.forEachFitting(work, i, (e) => {
        const v = dict.rewards[e] + this.exact(work);
        if (v > best) {
          best = v;
          pick = e;
        }
      });
      if (pick === -1) {
        work[i]--;
      } else {
        words.push(dict.words[pick]);
        const k = dict.keys[pick];
        for (let j = 0; j < k.length; j++) work[k.charCodeAt(j) - A]--;
      }
    }
    return { value: r.value, words, exact: true };
  }

  // Repeatedly take the highest-reward word that fits.
  greedy(c: Uint8Array): { value: number; words: string[] } {
    const work = c.slice();
    const words: string[] = [];
    let value = 0;
    const { dict } = this;
    for (;;) {
      let bestE = -1;
      let bestR = 0;
      dict.forEachFitting(work, -1, (e) => {
        const r = dict.rewards[e];
        if (r > bestR) {
          bestR = r;
          bestE = e;
        }
      });
      if (bestE === -1) break;
      words.push(dict.words[bestE]);
      value += bestR;
      const k = dict.keys[bestE];
      for (let j = 0; j < k.length; j++) work[k.charCodeAt(j) - A]--;
    }
    return { value, words };
  }

  // The top-k index entries that fit in the rack, by reward.
  topFitting(c: Uint8Array, k: number): number[] {
    const found: number[] = [];
    this.dict.forEachFitting(c.slice(), -1, (e) => found.push(e));
    found.sort((a, b) => this.dict.rewards[b] - this.dict.rewards[a] || a - b);
    return found.slice(0, k);
  }
}
