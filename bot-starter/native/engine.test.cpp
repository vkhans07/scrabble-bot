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
