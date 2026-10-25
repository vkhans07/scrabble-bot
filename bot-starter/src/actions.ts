// Reservations live until the public WordPlay confirms consumption. Each
// occurrence has an ID: two copies of a word in a disjoint plan are legal.
export class WordReservations {
  private next = 0;
  pending = new Map<number, string>();
  uncertain = false;

  reserve(rack: string, words: string[]): { id: number; word: string }[] {
    if (this.uncertain) return [];
    const counts = new Map<string, number>();
    for (const c of rack) counts.set(c, (counts.get(c) ?? 0) + 1);
    const consume = (word: string) => {
      for (const c of word) counts.set(c, (counts.get(c) ?? 0) - 1);
    };
    for (const word of this.pending.values()) consume(word);
    const out = [];
    for (const word of words) {
      const need = new Map<string, number>();
      for (const c of word) need.set(c, (need.get(c) ?? 0) + 1);
      if ([...need].some(([c, n]) => (counts.get(c) ?? 0) < n)) continue;
      consume(word);
      const id = ++this.next;
      this.pending.set(id, word);
      out.push({ id, word });
    }
    return out;
  }

  confirm(word: string) {
    for (const [id, w] of this.pending) {
      if (w === word) { this.pending.delete(id); return; }
    }
  }
  reject(id: number) { this.pending.delete(id); }
  disconnect() { if (this.pending.size) this.uncertain = true; }
}

export function shouldReplaceBid(previous: number | undefined, amount: number, leftMs: number, guardMs: number): boolean {
  return leftMs > guardMs && amount !== previous && (amount > 0 || (previous ?? 0) > 0);
}

// Server timestamps are UTC. Do not estimate clock skew from notification
// receipt: that folds network delay into the deadline and grants extra time.
// A synchronized local clock is assumed; cap at the auction duration and
// freeze the deadline on a monotonic clock so later wall-clock changes cannot
// reopen an auction. The send guard below provides additional headroom.
export function auctionDeadlineMs(opensAtMs: number, closesAtMs: number, wallNow: number, monotonicNow: number): number {
  return monotonicNow + Math.min(closesAtMs - wallNow, Math.max(0, closesAtMs - opensAtMs));
}

export class BidTiming {
  private samples: { at: number; ms: number }[] = [];
  private offsetCeilingMs = 0;
  deadline(opensAtMs: number, closesAtMs: number, wallNow: number, monotonicNow: number): number {
    // A negative receive-minus-open proves that our clock trails server time.
    // Keep the smallest observation, but NEVER add a positive delivery delay
    // to a deadline. This can only shorten the raw UTC deadline.
    this.offsetCeilingMs = Math.min(this.offsetCeilingMs, wallNow - opensAtMs);
    return auctionDeadlineMs(opensAtMs + this.offsetCeilingMs,
      closesAtMs + this.offsetCeilingMs, wallNow, monotonicNow);
  }
  observe(roundTripMs: number, now: number) {
    if (!Number.isFinite(roundTripMs) || roundTripMs < 0) return;
    this.samples.push({ at: now, ms: roundTripMs });
    if (this.samples.length > 32) this.samples.shift();
  }
  guardMs(now: number): number {
    // Use the full observed round trip as a conservative proxy, not RTT/2:
    // uplink/downlink symmetry and server queueing are unknown. Cap the
    // heuristic at 750 ms so a one-second auction can still receive an early
    // bid after a latency spike. This is headroom, not a delivery guarantee.
    this.samples = this.samples.filter(s => now - s.at < 30_000);
    const values = this.samples.map(s => s.ms).sort((a, b) => a - b);
    const p90 = values.length ? values[Math.ceil(values.length * 0.9) - 1] : 250;
    return Math.min(750, Math.max(300, p90 + 50));
  }
}
