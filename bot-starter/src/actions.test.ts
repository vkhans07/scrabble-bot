import { test } from 'node:test';
import assert from 'node:assert/strict';
import { WordReservations, shouldReplaceBid } from './actions.js';

test('zero withdraws an active bid, including deadline guard', () => {
  assert.equal(shouldReplaceBid(10, 0, 300, 150), true);
  assert.equal(shouldReplaceBid(10, 0, 150, 150), false);
  assert.equal(shouldReplaceBid(undefined, 0, 300, 150), false);
  assert.equal(shouldReplaceBid(10, 10, 300, 150), false);
  assert.equal(shouldReplaceBid(undefined, 10, -1, 150), false);
});
test('repeated words reserve separate occurrences; overlapping plans cannot reuse tiles', () => {
  const q = new WordReservations();
  assert.equal(q.reserve('AATT', ['AT', 'AT']).length, 2);
  assert.equal(q.reserve('AATT', ['AT']).length, 0);
  q.confirm('AT');
  assert.equal(q.pending.size, 1);
  assert.equal(q.reserve('AT', ['AT']).length, 0);
  q.confirm('AT');
  assert.equal(q.reserve('', ['AT']).length, 0);
  assert.equal(q.reserve('AT', ['AT']).length, 1);
});
test('explicit rejection releases only the failed occurrence', () => {
  const q = new WordReservations();
  const [a] = q.reserve('AATT', ['AT', 'AT']);
  q.reject(a.id);
  assert.equal(q.reserve('AATT', ['AT', 'AT']).length, 1);
});
test('disconnect with unresolved submissions quarantines the match', () => {
  const q = new WordReservations();
  q.reserve('AT', ['AT']);
  q.disconnect();
  q.confirm('AT');
  assert.deepEqual(q.reserve('AATT', ['AT']), []);
  const clean = new WordReservations();
  clean.disconnect();
  assert.equal(clean.reserve('AT', ['AT']).length, 1);
});

test('delayed auction notification does not extend the server deadline', async () => {
  const { auctionDeadlineMs } = await import('./actions.js');
  // Live regression: 691 ms elapsed before we received the open event.
  // The old offset correction reported 742 ms remaining; only 308 remain.
  const deadline = auctionDeadlineMs(3_922, 4_922, 4_614, 100);
  assert.equal(deadline - 100, 308);
  assert.equal(deadline - 409, -1);
  assert.equal(auctionDeadlineMs(1000, 2000, 2100, 500), 400);
  // A behind local clock must never grant more than the auction duration.
  assert.equal(auctionDeadlineMs(1000, 2000, 0, 500), 1500);
});
test('latency guard tightens on slow responses and ages out stale samples', async () => {
  const { BidTiming } = await import('./actions.js');
  const timing = new BidTiming();
  assert.equal(timing.guardMs(0), 300);
  timing.observe(450, 10);
  assert.equal(timing.guardMs(10), 500);
  assert.equal(shouldReplaceBid(10, 15, 400, timing.guardMs(10)), false);
  assert.equal(shouldReplaceBid(undefined, 10, 900, timing.guardMs(10)), true);
  timing.observe(1200, 20);
  assert.equal(timing.guardMs(20), 750);
  assert.equal(timing.guardMs(30_020), 300);
});
test('evidence of a behind local clock only shortens future deadlines', async () => {
  const { BidTiming } = await import('./actions.js');
  const timing = new BidTiming();
  // Local clock trails the server. Preserve the negative observation.
  assert.equal(timing.deadline(1000, 2000, 750, 0), 1000);
  // A later event takes 400 ms longer to arrive: do not reset its window.
  assert.equal(timing.deadline(2000, 3000, 2150, 1000), 1600);
  // Positive delivery delays cannot grant back any time.
  assert.equal(timing.deadline(3000, 4000, 3400, 2000), 2350);
});
