// Run with `npm test`.

import assert from "node:assert/strict";
import { test } from "node:test";
import { A, BAG, Dict, countsOf, rewardOf } from "./dict.js";
import { RackEval } from "./rackeval.js";
import { makeRng } from "./rng.js";

const dict = Dict.load();

test("wordlist loads with the expected size", () => {
  assert.ok(dict.allWords.size > 170000, `got ${dict.allWords.size}`);
  assert.ok(dict.isWord("QUIZ"));
});

test("rewards follow the length multipliers", () => {
  assert.equal(rewardOf("QI"), 11);
  assert.equal(rewardOf("ZAX"), 19); // 10+1+8, 3 letters
  assert.equal(rewardOf("JAZZ"), Math.floor((8 + 1 + 10 + 10) * 1.5)); // 43
  assert.equal(rewardOf("QUIZS"), (10 + 1 + 1 + 10 + 1) * 2);
  assert.equal(rewardOf("QUIZES"), Math.floor((10 + 1 + 1 + 10 + 1 + 1) * 2.5));
  assert.equal(rewardOf("QUIZZES"), (10 + 1 + 1 + 10 + 10 + 1 + 1) * 3);
});

// Independent oracle: scan the whole wordlist for words that fit, then search
// every set of disjoint words.
function bruteForce(rack: Uint8Array): number {
  const fits: { c: Uint8Array; r: number }[] = [];
  for (const w of dict.allWords) {
    if (w.length > 8) continue;
    const wc = countsOf(w);
    let ok = true;
    for (let i = 0; i < 26 && ok; i++) if (wc[i] > rack[i]) ok = false;
    if (ok) fits.push({ c: wc, r: rewardOf(w) });
  }
  const work = rack.slice();
  const go = (from: number): number => {
    let best = 0;
    for (let j = from; j < fits.length; j++) {
      const { c, r } = fits[j];
      let ok = true;
      for (let i = 0; i < 26 && ok; i++) if (c[i] > work[i]) ok = false;
      if (!ok) continue;
      for (let i = 0; i < 26; i++) work[i] -= c[i];
      const v = r + go(j);
      for (let i = 0; i < 26; i++) work[i] += c[i];
      if (v > best) best = v;
    }
    return best;
  };
  return go(0);
}

function randomRack(rng: () => number, size: number): Uint8Array {
  const pool: number[] = [];
  BAG.forEach((n, i) => {
    for (let k = 0; k < n; k++) pool.push(i);
  });
  const c = new Uint8Array(26);
  for (let k = 0; k < size; k++) {
    const j = Math.floor(rng() * pool.length);
    c[pool[j]]++;
    pool.splice(j, 1);
  }
  return c;
}

test("B(rack) matches brute force on 500 random racks of <= 8 tiles", () => {
  const rng = makeRng(12345);
  const ev = new RackEval(dict, 14);
  for (let t = 0; t < 500; t++) {
    const size = 2 + Math.floor(rng() * 7);
    const rack = randomRack(rng, size);
    const expect = bruteForce(rack);
    const before = rack.slice();
    const got = ev.best(rack);
    assert.deepEqual(rack, before, "best() must restore the rack");
    assert.equal(got, expect, `rack ${RackEval.keyOf(rack)}`);
    const plan = ev.plan(rack);
    assert.equal(plan.value, expect);
    // The plan's words must be disjoint, playable, and sum to the value.
    const left = rack.slice();
    let sum = 0;
    for (const w of plan.words) {
      assert.ok(dict.isWord(w), w);
      for (const ch of w) assert.ok(left[ch.charCodeAt(0) - A]-- > 0, w);
      sum += rewardOf(w);
    }
    assert.equal(sum, expect);
  }
});

test("budgeted evaluation falls back to greedy without corrupting the rack", () => {
  const ev = new RackEval(dict, 14);
  const rack = randomRack(makeRng(7), 13);
  const before = rack.slice();
  const r = ev.bestBudgeted(rack, 3);
  assert.deepEqual(rack, before);
  assert.equal(r.exact, false);
  assert.ok(ev.greedyCount >= 1);
  assert.ok(r.value >= 0);
});

test("greedy never beats exact", () => {
  const rng = makeRng(99);
  const ev = new RackEval(dict, 14);
  for (let t = 0; t < 50; t++) {
    const rack = randomRack(rng, 10);
    assert.ok(ev.greedy(rack).value <= ev.best(rack));
  }
});
