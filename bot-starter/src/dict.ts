// Wordlist, rewards, and the anagram index.
//
// Reward depends only on a word's letter multiset, so one entry per sorted-
// letter key is enough. The keys are also stored in a trie (first-child /
// next-sibling arrays) so that "which keys fit inside this rack" is a pruned
// walk rather than an enumeration of every sub-multiset.

import * as fs from "node:fs";
import * as path from "node:path";

export const A = 65;
export const BAG: readonly number[] = [
  9, 2, 2, 4, 12, 2, 3, 2, 9, 1, 1, 4, 2, 6, 8, 2, 1, 6, 4, 6, 4, 2, 2, 1, 2, 1,
];

const VALUES = "AEIOULNSTR".split("").reduce(
  (acc, c) => {
    acc[c.charCodeAt(0) - A] = 1;
    return acc;
  },
  new Array<number>(26).fill(0),
);
for (const c of "DG") VALUES[c.charCodeAt(0) - A] = 2;
for (const c of "BCMP") VALUES[c.charCodeAt(0) - A] = 3;
for (const c of "FHVWY") VALUES[c.charCodeAt(0) - A] = 4;
VALUES["K".charCodeAt(0) - A] = 5;
for (const c of "JX") VALUES[c.charCodeAt(0) - A] = 8;
for (const c of "QZ") VALUES[c.charCodeAt(0) - A] = 10;
export const LETTER_VALUE: readonly number[] = VALUES;

export function letterIndex(c: string): number {
  return c.charCodeAt(0) - A;
}

// floor(sum of letter values * multiplier), done in integers like the server.
export function rewardOf(word: string): number {
  let base = 0;
  for (let i = 0; i < word.length; i++) base += VALUES[word.charCodeAt(i) - A];
  const n = word.length;
  if (n <= 3) return base;
  if (n === 4) return Math.floor((base * 3) / 2);
  if (n === 5) return base * 2;
  if (n === 6) return Math.floor((base * 5) / 2);
  return base * 3;
}

export function keyOfWord(word: string): string {
  return word.split("").sort().join("");
}

export function defaultWordlistPath(): string {
  return path.join(
    path.dirname(new URL(import.meta.url).pathname),
    "..",
    "..",
    "spacetimedb",
    "wordlist.txt",
  );
}

export class Dict {
  // Per index entry: the representative word, its reward, and its key.
  words: string[] = [];
  rewards: number[] = [];
  keys: string[] = [];
  byKey = new Map<string, number>();
  allWords = new Set<string>();

  // Trie over sorted keys. Node 0 is the root.
  firstChild: number[] = [-1];
  nextSib: number[] = [-1];
  letter: number[] = [-1];
  entry: number[] = [-1]; // index entry ending at this node, or -1

  constructor(lines: string[]) {
    for (const raw of lines) {
      const w = raw.trim().toUpperCase();
      if (w.length < 2 || !/^[A-Z]+$/.test(w)) continue;
      this.allWords.add(w);
      const key = keyOfWord(w);
      const r = rewardOf(w);
      const ex = this.byKey.get(key);
      if (ex === undefined) {
        this.byKey.set(key, this.words.length);
        this.words.push(w);
        this.rewards.push(r);
        this.keys.push(key);
      } else if (w < this.words[ex]) {
        this.words[ex] = w; // deterministic representative
      }
    }
    for (let i = 0; i < this.keys.length; i++) this.insertTrie(this.keys[i], i);
  }

  static load(file = defaultWordlistPath()): Dict {
    return new Dict(fs.readFileSync(file, "utf8").split("\n"));
  }

  private insertTrie(key: string, idx: number) {
    let node = 0;
    for (let i = 0; i < key.length; i++) {
      const L = key.charCodeAt(i) - A;
      let c = this.firstChild[node];
      while (c !== -1 && this.letter[c] !== L) c = this.nextSib[c];
      if (c === -1) {
        c = this.letter.length;
        this.letter.push(L);
        this.firstChild.push(-1);
        this.nextSib.push(this.firstChild[node]);
        this.entry.push(-1);
        this.firstChild[node] = c;
      }
      node = c;
    }
    this.entry[node] = idx;
  }

  isWord(w: string): boolean {
    return this.allWords.has(w);
  }

  // Every index entry whose key fits inside `counts`, optionally restricted
  // to keys whose first (lowest) letter is `first`.
  forEachFitting(
    counts: Uint8Array,
    first: number,
    cb: (entryIdx: number) => void,
  ) {
    const walk = (node: number) => {
      for (let c = this.firstChild[node]; c !== -1; c = this.nextSib[c]) {
        const L = this.letter[c];
        if (counts[L] === 0) continue;
        counts[L]--;
        const e = this.entry[c];
        if (e !== -1) cb(e);
        walk(c);
        counts[L]++;
      }
    };
    if (first < 0) {
      walk(0);
      return;
    }
    // Only the root child for `first`.
    for (let c = this.firstChild[0]; c !== -1; c = this.nextSib[c]) {
      if (this.letter[c] !== first) continue;
      if (counts[first] === 0) return;
      counts[first]--;
      const e = this.entry[c];
      if (e !== -1) cb(e);
      walk(c);
      counts[first]++;
      return;
    }
  }
}

export function countsOf(word: string, out = new Uint8Array(26)): Uint8Array {
  for (let i = 0; i < word.length; i++) out[word.charCodeAt(i) - A]++;
  return out;
}
