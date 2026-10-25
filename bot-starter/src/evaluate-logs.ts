// Read-only aggregation: npm run evaluate:logs -- logs/<run-directory>
import { readdirSync, readFileSync } from 'node:fs';
import { join } from 'node:path';

const dir = process.argv[2];
if (!dir) throw new Error('Pass a log directory');
const percentile = (values: number[], p: number) => {
  if (!values.length) return null;
  const sorted = [...values].sort((a, b) => a - b);
  return Math.round(sorted[Math.ceil(p * sorted.length) - 1] * 10) / 10;
};
for (const name of readdirSync(dir).filter(n => /^match-.*\.jsonl$/.test(n)).sort()) {
  const events = readFileSync(join(dir, name), 'utf8').trim().split('\n').filter(Boolean).flatMap(line => {
    try { return [JSON.parse(line)]; } catch { return []; } // tolerate an active file's partial last line
  });
  const of = (type: string) => events.filter(e => e.type === type);
  const timing = (type: string, field: string, phase?: string) => {
    const values = of(type).filter(e => !phase || e.phase === phase).map(e => e[field]).filter(Number.isFinite);
    return { n: values.length, median: percentile(values, 0.5), p95: percentile(values, 0.95), max: values.length ? Math.max(...values) : null };
  };
  const summary = of('match_summary').at(-1);
  console.log(JSON.stringify({
    file: name, complete: !!summary, summary,
    actions: {
      bids: of('bid').length, bidAcks: of('bid_ack').length,
      bidFailures: of('bid_failed').length, zeroReplacements: of('bid').filter(e => e.amount === 0).length,
      words: of('word').length, wordAcks: of('word_ack').length,
      wordFailures: of('word_failed').length,
      uncertainWordFailures: of('word_failed').filter(e => !e.confirmed).length,
      lateSkipped: of('late_action_skipped').length,
      reducerErrors: of('reducer_error').map(e => e.message),
    },
    timingsMs: { bidDecision: timing('bid', 'decisionMs'), bidRoundTrip: timing('bid_ack', 'roundTripMs'), wordRoundTrip: timing('word_ack', 'roundTripMs'), engine: timing('engine', 'ms'), engineFast: timing('engine', 'ms', 'fast'), engineFull: timing('engine', 'ms', 'full'), engineRefine: timing('engine', 'ms', 'refine'), engineSearch: timing('engine', 'ms', 'search') },
    engineRequests: of('engine_request').length,
    fullResults: of('engine').filter(e => e.phase === 'full').length,
    nativeResults: of('engine').filter(e => e.perf?.native).length,
    beliefMismatches: of('belief_mismatch').length,
  }));
}
