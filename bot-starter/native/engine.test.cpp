#define main engine_main
#include "engine.cpp"
#undef main
#include <cassert>

int main(int argc, char** argv) {
  assert(argc == 2 && D.load(argv[1]));
  Eval ev(10);
  Snap s;
  s.rack[0] = 7;
  s.rack[25] = 1;
  s.usableLeft = 3;
  s.balance = 0;
  Table tb{};
  tb.lam = 1;
  tb.v[25] = 100;
  Scen scenario;
  scenario.T = 1;
  scenario.L = {25}; scenario.P = {1};
  scenario.tied = {1}; scenario.tieU = {0};
  Arm cash;
  cash.cashEntry = D.byKey.at("AAAAAAA");
  const double hold = runArm(s, scenario, tb, Arm{}, ev);
  const double play = runArm(s, scenario, tb, cash, ev);
  assert(hold == 21);
  assert(play == 41);
  assert(play - hold == 20); // Previously both arms auto-cashed: gain was zero.
  scenario.T = 0;
  assert(runArm(s, scenario, tb, cash, ev) == runArm(s, scenario, tb, Arm{}, ev));
  puts("native cash-in regression passed");

  {
    // Opponent kind 4 (fitted prior) parses and samples as specified; the old
    // kinds keep their format and values.
    string line = "1 m a 4 10 50 0 0";
    for (int i = 0; i < 52; i++) line += " 0";
    line += " 2 4 30 0.25";
    for (int L = 0; L < 26; L++) line += L == 25 ? " 40 0.05" : " 6.4 0.05";
    line += " 3 30 8 2.5 4";
    istringstream in(line);
    Snap ps;
    assert(parseSnap(in, ps));
    assert(ps.opps.size() == 2);
    const Opp& p = ps.opps[0];
    assert(p.kind == PRIOR && p.bal == 30 && abs(p.pi - 0.25) < 1e-12);
    assert(oppBid(p, 0, 0.1) == 0);    // u < pi: no bid
    assert(oppBid(p, 0, 0.625) == 6);  // median of the rest: round(6.4)
    assert(oppBid(p, 25, 0.625) == 30); // capped at balance
    const Opp& r = ps.opps[1];
    assert(r.kind == REAL && r.bal == 30 && r.bonus == 8 && r.slope == 2.5 && r.jit == 4);
    assert(oppBid(r, 0, 0.5) == 11);   // round(8 + 2.5) + jitter 0
    // A wide prior's empirical median and no-bid share.
    Opp w = p;
    for (int L = 0; L < 26; L++) w.sd[L] = 3;
    w.bal = 100;
    Rng rr(9);
    int zeros = 0, below = 0, n = 200000;
    for (int k = 0; k < n; k++) {
      int b = oppBid(w, 3, rr.u());
      zeros += b == 0;
      below += b > 0 && b <= 6;
    }
    // P(0) = pi + (1 - pi) F((0.5 - 6.4) / 3); P(1..6) = (1 - pi) (F(-0.1/3) - F(-5.9/3))
    auto F = [](double z) { return 1 / (1 + exp(-z)); };
    assert(abs(zeros / (double)n - (0.25 + 0.75 * F(-5.9 / 3))) < 0.005);
    assert(abs(below / (double)n - 0.75 * (F(0.1 / 3) - F(-5.9 / 3))) < 0.005);
    puts("native prior-opponent regression passed");
  }

  CFG.threads = 1;
  CFG.memoLog2 = 10;
  s.seq = 7;
  s.letter = 25;
  s.U[0] = 3;
  s.U[25] = 3;
  s.balance = 5;
  LATEST_SEQ.store(s.seq);
  vector<Arm> arms(3);
  arms[1].addCoins = 15;
  arms[2].addLetter = 25;
  const JobResult first = runJob(s, tb, arms, 7, 10000);
  const JobResult next = runJob(s, tb, arms, 13, 10000, 7);
  const JobResult merged = mergeJobs(first, next);
  const JobResult all = runJob(s, tb, arms, 20, 10000);
  assert(first.n == 7 && next.n == 13 && merged.n == 20 && all.n == 20);
  assert(abs(merged.baseMean - all.baseMean) < 1e-9);
  for (size_t i = 0; i < arms.size(); i++)
    assert(abs(merged.diff[i] - all.diff[i]) < 1e-9);
  assert(mergeJobs(JobResult{}, first).n == first.n);
  assert(mergeJobs(first, JobResult{}).n == first.n);

  // Exercise the real phase publisher/protocol with small deterministic jobs.
  CFG.nFast = 4;
  CFG.nRefine = 10;
  CFG.nMax = 6;
  CFG.fastMs = CFG.refineMs = CFG.fullMs = 10000;
  s.matchId = "test";
  s.auctionId = "auction";
  handleSnap(s);
  CFG.nFast = 0;
  s.seq = 8;
  LATEST_SEQ.store(s.seq);
  handleSnap(s);
  puts("native progressive regression passed");
}
