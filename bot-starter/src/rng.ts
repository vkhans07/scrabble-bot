// xoshiro128** seeded through splitmix32, so runs and replays are reproducible.

export type Rng = () => number; // uniform in [0, 1)

export function makeRng(seed: number): Rng {
  let s = seed >>> 0;
  const sm = () => {
    s = (s + 0x9e3779b9) >>> 0;
    let z = s;
    z = Math.imul(z ^ (z >>> 16), 0x85ebca6b) >>> 0;
    z = Math.imul(z ^ (z >>> 13), 0xc2b2ae35) >>> 0;
    return (z ^ (z >>> 16)) >>> 0;
  };
  let a = sm(),
    b = sm(),
    c = sm(),
    d = sm();
  const rotl = (x: number, k: number) => ((x << k) | (x >>> (32 - k))) >>> 0;
  return () => {
    const result = Math.imul(rotl(Math.imul(b, 5) >>> 0, 7), 9) >>> 0;
    const t = (b << 9) >>> 0;
    c ^= a;
    d ^= b;
    b ^= c;
    a ^= d;
    c ^= t;
    d = rotl(d >>> 0, 11);
    a >>>= 0;
    b >>>= 0;
    c >>>= 0;
    d >>>= 0;
    return result / 4294967296;
  };
}
