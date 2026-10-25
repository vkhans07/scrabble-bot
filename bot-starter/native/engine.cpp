// Native ScrabbleBot engine: exact rack evaluation and full-game Monte Carlo
// rollouts on a thread pool. Talks to the Node worker over stdin/stdout in a
// line-based text protocol (see README section at the bottom of this file).
//
// A scenario is the rest of the match: every future usable letter drawn from
// the unseen pool, plus the best opposing bid for each. Each "arm" replays the
// same scenario from a slightly different start (an extra tile, extra coins,
// a word cashed in now) and bids / plays with a fixed policy to the end, so
// arm values are final scores and arm differences are first-principles values:
//   v[L]    = E[score | win tile L now] - E[score | don't]
//   lambda  = d E[score] / d coin       (from the extra-coins arm)
//   gain(w) = E[score | play w now] - E[score | hold]
// Common random numbers across arms keep those differences low-variance.

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

using namespace std;
typedef array<uint8_t, 26> Rack;
typedef chrono::steady_clock Clock;

static const int BAGC[26] = {9, 2, 2, 4, 12, 2, 3, 2, 9, 1, 1, 4, 2, 6, 8, 2, 1, 6, 4, 6, 4, 2, 2, 1, 2, 1};
static const int VAL[26] = {1, 3, 3, 2, 1, 4, 2, 4, 1, 8, 5, 1, 3, 1, 1, 3, 10, 1, 1, 1, 1, 4, 4, 8, 4, 10};

// ---------------------------------------------------------------- dictionary

struct Dict {
  vector<string> words, keys;
  vector<int> rewards, lens;
  vector<Rack> counts;
  vector<int> fc{-1}, ns{-1}, entry{-1};
  vector<signed char> let{-1};
  unordered_map<string, int> byKey;

  static int rewardOf(const string& w) {
    int b = 0;
    for (char c : w) b += VAL[c - 'A'];
    int n = (int)w.size();
    if (n <= 3) return b;
    if (n == 4) return b * 3 / 2;
    if (n == 5) return b * 2;
    if (n == 6) return b * 5 / 2;
    return b * 3;
  }

  bool load(const string& path) {
    ifstream f(path);
    if (!f) return false;
    string w;
    while (getline(f, w)) {
      while (!w.empty() && (w.back() == '\r' || w.back() == ' ')) w.pop_back();
      if (w.size() < 2) continue;
      bool ok = true;
      for (auto& ch : w) {
        ch = (char)toupper(ch);
        if (ch < 'A' || ch > 'Z') ok = false;
      }
      if (!ok) continue;
      string k = w;
      sort(k.begin(), k.end());
      auto it = byKey.find(k);
      if (it == byKey.end()) {
        byKey[k] = (int)words.size();
        words.push_back(w);
        keys.push_back(k);
        rewards.push_back(rewardOf(w));
        lens.push_back((int)w.size());
        Rack c{};
        for (char ch : w) c[ch - 'A']++;
        counts.push_back(c);
      } else if (w < words[it->second]) {
        words[it->second] = w;  // deterministic representative, like the JS side
      }
    }
    for (size_t i = 0; i < keys.size(); i++) insertTrie(keys[i], (int)i);
    return true;
  }

  void insertTrie(const string& key, int idx) {
    int node = 0;
    for (char ch : key) {
      int L = ch - 'A';
      int c = fc[node];
      while (c != -1 && let[c] != L) c = ns[c];
      if (c == -1) {
        c = (int)let.size();
        let.push_back((signed char)L);
        fc.push_back(-1);
        ns.push_back(fc[node]);
        entry.push_back(-1);
        fc[node] = c;
      }
      node = c;
    }
    entry[node] = idx;
  }
};

static Dict D;

template <class F>
static void walkAll(int node, uint8_t* c, F& cb) {
  for (int ch = D.fc[node]; ch != -1; ch = D.ns[ch]) {
    int L = D.let[ch];
    if (!c[L]) continue;
    c[L]--;
    int e = D.entry[ch];
    if (e != -1) cb(e);
    walkAll(ch, c, cb);
    c[L]++;
  }
}

// Keys whose lowest letter is `first`.
template <class F>
static void walkFirst(int first, uint8_t* c, F& cb) {
  for (int ch = D.fc[0]; ch != -1; ch = D.ns[ch]) {
    if (D.let[ch] != first) continue;
    if (!c[first]) return;
    c[first]--;
    int e = D.entry[ch];
    if (e != -1) cb(e);
    walkAll(ch, c, cb);
    c[first]++;
    return;
  }
}

// ---------------------------------------------------------------- evaluator

struct Memo {
  struct Slot {
    uint64_t a, b;
    int32_t v;
  };
  vector<Slot> t;
  size_t mask = 0, n = 0;
  void init(int log2) {
    t.assign((size_t)1 << log2, Slot{0, 0, -1});
    mask = ((size_t)1 << log2) - 1;
    n = 0;
  }
  inline size_t h(uint64_t a, uint64_t b) const {
    uint64_t x = a * 0x9E3779B97F4A7C15ULL ^ (b + 0x632BE59BD9B4E019ULL) * 0xC2B2AE3D27D4EB4FULL;
    x ^= x >> 31;
    return (size_t)x & mask;
  }
  inline int get(uint64_t a, uint64_t b) const {
    for (size_t i = h(a, b);; i = (i + 1) & mask) {
      const Slot& s = t[i];
      if (s.v < 0) return -1;
      if (s.a == a && s.b == b) return s.v;
    }
  }
  inline void put(uint64_t a, uint64_t b, int v) {
    if ((n + 1) * 2 > t.size()) {  // full: start over rather than grow
      for (auto& s : t) s.v = -1;
      n = 0;
    }
    size_t i = h(a, b);
    while (t[i].v >= 0) i = (i + 1) & mask;
    t[i] = Slot{a, b, v};
    n++;
  }
};

struct Eval {
  Memo memo;
  int exactCap = 16;
  long hits = 0, misses = 0, bigCalls = 0;

  explicit Eval(int log2 = 21) { memo.init(log2); }

  static int total(const uint8_t* c) {
    int n = 0;
    for (int i = 0; i < 26; i++) n += c[i];
    return n;
  }

  // Exact best set of disjoint words. `c` is restored on return.
  int exact(uint8_t* c) {
    int i = 0;
    while (i < 26 && !c[i]) i++;
    if (i == 26) return 0;
    uint64_t a = 0, b = 0;
    int tot = 0;
    for (int j = 0; j < 16; j++) {
      a |= (uint64_t)c[j] << (4 * j);
      tot += c[j];
    }
    for (int j = 16; j < 26; j++) {
      b |= (uint64_t)c[j] << (4 * (j - 16));
      tot += c[j];
    }
    if (tot < 2) return 0;
    int g = memo.get(a, b);
    if (g >= 0) {
      hits++;
      return g;
    }
    misses++;
    c[i]--;
    int best = exact(c);
    c[i]++;
    auto cb = [&](int e) {
      int v = D.rewards[e] + exact(c);
      if (v > best) best = v;
    };
    walkFirst(i, c, cb);
    memo.put(a, b, best);
    return best;
  }

  // Greedy completion: repeatedly take the highest-reward word that fits.
  int greedy(Rack r, vector<int>* out) {
    int value = 0;
    for (;;) {
      int be = -1, br = 0;
      auto cb = [&](int e) {
        if (D.rewards[e] > br) {
          br = D.rewards[e];
          be = e;
        }
      };
      walkAll(0, r.data(), cb);
      if (be < 0) break;
      value += br;
      if (out) out->push_back(be);
      for (int i = 0; i < 26; i++) r[i] -= D.counts[be][i];
    }
    return value;
  }

  // Above the exact cap: greedy with one step of lookahead over the top 10
  // first words (measured at 2-3% from optimal on 14-18 tile racks).
  int bigPlan(Rack r, vector<int>* out) {
    bigCalls++;
    vector<int> fit;
    auto cb = [&](int e) { fit.push_back(e); };
    walkAll(0, r.data(), cb);
    sort(fit.begin(), fit.end(), [&](int x, int y) { return D.rewards[x] != D.rewards[y] ? D.rewards[x] > D.rewards[y] : x < y; });
    vector<int> bestPlan;
    int best = greedy(r, &bestPlan);
    for (size_t k = 0; k < fit.size() && k < 10; k++) {
      Rack r2 = r;
      for (int i = 0; i < 26; i++) r2[i] -= D.counts[fit[k]][i];
      vector<int> p;
      int v = D.rewards[fit[k]] + greedy(r2, &p);
      if (v > best) {
        best = v;
        bestPlan.assign(1, fit[k]);
        bestPlan.insert(bestPlan.end(), p.begin(), p.end());
      }
    }
    if (out) *out = bestPlan;
    return best;
  }

  int best(const Rack& r) {
    int tot = total(r.data());
    if (tot < 2) return 0;
    if (tot > exactCap) return bigPlan(r, nullptr);
    Rack w = r;
    return exact(w.data());
  }

  // The words of a best plan, as dictionary entries.
  vector<int> plan(const Rack& r, int* value = nullptr) {
    vector<int> words;
    int tot = total(r.data());
    if (tot < 2) {
      if (value) *value = 0;
      return words;
    }
    if (tot > exactCap) {
      int v = bigPlan(r, &words);
      if (value) *value = v;
      return words;
    }
    Rack work = r;
    uint8_t* w = work.data();
    int v0 = exact(w);
    if (value) *value = v0;
    for (;;) {
      int i = 0;
      while (i < 26 && !w[i]) i++;
      if (i == 26) break;
      w[i]--;
      int best = exact(w);
      w[i]++;
      int pick = -1;
      auto cb = [&](int e) {
        int v = D.rewards[e] + exact(w);
        if (v > best) {
          best = v;
          pick = e;
        }
      };
      walkFirst(i, w, cb);
      if (pick < 0) {
        w[i]--;
      } else {
        words.push_back(pick);
        for (int j = 0; j < 26; j++) w[j] -= D.counts[pick][j];
      }
    }
    return words;
  }
};

// ---------------------------------------------------------------- RNG

struct Rng {
  uint64_t s;
  explicit Rng(uint64_t seed) : s(seed) {}
  inline uint64_t next() {
    uint64_t z = (s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
  }
  inline double u() { return (next() >> 11) * (1.0 / 9007199254740992.0); }
};

// ---------------------------------------------------------------- model

// PRIOR (4): a fitted per-opponent model (src/oppmodel.ts). With probability
// pi no bid; otherwise clamp(round(mu[L] + sd[L] * logistic noise), 0, bal).
enum Kind { CHEAP = 0, VALUE = 1, AGGR = 2, REAL = 3, PRIOR = 4 };
struct Opp {
  int kind = REAL;
  int bal = 100;
  double bonus = 8, slope = 2.5;
  int jit = 4;
  double pi = 0;
  float mu[26] = {};
  float sd[26] = {};
};

struct Cfg {
  int threads = 8;
  uint64_t seed = 1;
  int exactCap = 16;
  int memoLog2 = 21;
  int allinK = 5;
  double allinMinV = 1;
  int endgameK = 2;
  int minPlayLen = 7;
  int flushSize = 14;
  int nMax = 400;
  int nFast = 120;
  int nRefine = 3000;
  int refineMs = 75;
  int bidSearch = 0; // 0: coarse curve, 1: adaptive local search, 2: every price
  int fastMs = 150;
  int fullMs = 600;
  int cashK = 6;
  int coinArm = 15;
  double damp = 0.5;
  double lamDamp = 0.3;
  double lamMin = 0.05, lamMax = 1.0;
  double lamScale = 1.0;
};
static Cfg CFG;

static const bool VOWEL[26] = {1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0};

static inline int oppBid(const Opp& o, int L, double u) {
  if (o.bal <= 0) return 0;
  if (o.kind == PRIOR) {
    if (u < o.pi) return 0;
    double v = (u - o.pi) / (1 - o.pi);
    v = min(1 - 1e-9, max(1e-9, v));
    double y = o.mu[L] + o.sd[L] * log(v / (1 - v));
    return max(0, min((int)lround(y), o.bal));
  }
  int face = VAL[L];
  int base, jitter;
  switch (o.kind) {
    case CHEAP: base = face - 1; jitter = (int)floor(u * 5) - 2; break;
    case VALUE: base = face; jitter = (int)floor(u * 5) - 2; break;
    case AGGR: base = face + (VOWEL[L] ? 4 : 2); jitter = (int)floor(u * 5) - 2; break;
    default: {
      int j = max(0, o.jit);
      base = (int)floor(o.bonus + o.slope * face + 0.5);
      jitter = (int)floor(u * (2 * j + 1)) - j;
    }
  }
  return max(0, min(max(base + jitter, 1), o.bal));
}

struct Snap {
  string matchId, auctionId;
  long seq = 0;
  int letter = -1, usableLeft = 0, balance = 0, score = 0;
  bool needCash = false;
  Rack rack{}, U{};
  vector<Opp> opps;
};

struct Table {
  double v[26];
  double lam = 0.35;
};

static double expectedOppPrice(const Snap& s, int L) {
  Rng rng(CFG.seed ^ (0xA0761D6478BD642FULL + (uint64_t)L * 0xE7037ED1A0B428DBULL));
  double sum = 0;
  constexpr int N = 256;
  for (int k = 0; k < N; k++) {
    int top = 0;
    for (const Opp& o : s.opps) top = max(top, oppBid(o, L, rng.u()));
    sum += top;
  }
  return sum / N;
}

static double fairBidFromCurve(double freeGain, const vector<pair<int, double>>& paid) {
  if (!(freeGain > 0)) return 0;
  int prevPrice = 0;
  double prevGain = freeGain;
  for (const auto& [price, gain] : paid) {
    if (gain <= 0) {
      if (prevGain <= 0) return prevPrice;
      double crossing = prevPrice + (double)(price - prevPrice) * prevGain / (prevGain - gain);
      return max(0.0, floor(crossing + 1e-9));
    }
    prevPrice = price;
    prevGain = gain;
  }
  return prevPrice;
}

struct Scen {
  int T = 0;
  vector<int8_t> L;
  vector<int16_t> P;
  vector<int8_t> tied;
  vector<float> tieU;
};

static void makeScenario(const Snap& s, uint64_t idx, Scen& S) {
  Rng rng(CFG.seed * 0x9E3779B97F4A7C15ULL + idx * 0xD1B54A32D192ED03ULL + 12345);
  int pool[128], np = 0;
  for (int i = 0; i < 26; i++)
    for (int k = 0; k < s.U[i] && np < 128; k++) pool[np++] = i;
  S.T = max(0, min(s.usableLeft, np));
  S.L.resize(S.T);
  S.P.resize(S.T);
  S.tied.resize(S.T);
  S.tieU.resize(S.T);
  for (int t = 0; t < S.T; t++) {
    int j = t + (int)(rng.u() * (np - t));
    swap(pool[t], pool[j]);
    int L = pool[t];
    S.L[t] = (int8_t)L;
    int P = 0, tied = 0;
    for (const Opp& o : s.opps) {
      int b = oppBid(o, L, rng.u());
      if (b > P) {
        P = b;
        tied = 1;
      } else if (b == P && b > 0) {
        tied++;
      }
    }
    S.P[t] = (int16_t)P;
    S.tied[t] = (int8_t)tied;
    S.tieU[t] = (float)rng.u();
  }
}

// ---------------------------------------------------------------- policy

static inline int bidRule(double v, double lam, int bal, int ul) {
  if (bal < 1 || !(v > 0)) return 0;
  if (ul < CFG.allinK && v >= CFG.allinMinV) return bal;
  int amt = (int)min((double)bal, floor(v / lam));
  return amt < 1 ? 0 : amt;
}

// Play words from the best plan when the rule says so. Called whenever the
// rack changes. Long words are cashed at once (7+ letters all score x3, so
// there is nothing to wait for and the coins fund more bids); shorter words
// are held in case later tiles lengthen them, until the endgame or until the
// rack gets too big to evaluate.
static void applyPolicy(Rack& rack, int& bal, double& score, int ul, Eval& ev) {
  int tot = Eval::total(rack.data());
  if (tot < 2) return;
  vector<int> plan = ev.plan(rack);
  if (plan.empty()) return;
  stable_sort(plan.begin(), plan.end(), [&](int a, int b) { return D.rewards[a] > D.rewards[b]; });
  bool endgame = ul <= CFG.endgameK;
  vector<char> played(plan.size(), 0);
  for (size_t k = 0; k < plan.size(); k++) {
    int e = plan[k];
    if (endgame || D.lens[e] >= CFG.minPlayLen) {
      for (int i = 0; i < 26; i++) rack[i] -= D.counts[e][i];
      bal += D.rewards[e];
      score += D.rewards[e];
      tot -= D.lens[e];
      played[k] = 1;
    }
  }
  for (size_t k = 0; k < plan.size() && tot > CFG.flushSize; k++) {
    if (played[k]) continue;
    int e = plan[k];
    for (int i = 0; i < 26; i++) rack[i] -= D.counts[e][i];
    bal += D.rewards[e];
    score += D.rewards[e];
    tot -= D.lens[e];
  }
}

struct Arm {
  int addLetter = -1;
  int addCoins = 0;
  int cashEntry = -1;
};

static double runArm(const Snap& s, const Scen& S, const Table& tb, const Arm& arm, Eval& ev) {
  Rack rack = s.rack;
  int bal = s.balance;
  double score = 0;
  if (arm.addLetter >= 0) rack[arm.addLetter]++;
  bal += arm.addCoins;
  if (arm.cashEntry >= 0) {
    int e = arm.cashEntry;
    for (int i = 0; i < 26; i++) rack[i] -= D.counts[e][i];
    bal += D.rewards[e];
    score += D.rewards[e];
  }
  // The baseline holds through the next auction. Cash only the explicit
  // candidate now; otherwise the continuation policy erases its funding
  // advantage before the first bid (unlike the live gain-gated client).
  for (int t = 0; t < S.T; t++) {
    int L = S.L[t];
    int ul = s.usableLeft - (t + 1);
    int bid = bidRule(tb.v[L], tb.lam * CFG.lamScale, bal, ul);
    if (bid <= 0) continue;
    int P = S.P[t];
    bool win = bid > P || (bid == P && S.tieU[t] < 1.0f / (S.tied[t] + 1));
    if (!win) continue;
    int pay = min(bid, max(P, 1));
    if (pay > bal) continue;
    bal -= pay;
    rack[L]++;
    applyPolicy(rack, bal, score, ul, ev);
  }
  int v;
  ev.plan(rack, &v);  // final auction: dump everything
  return score + v;
}

// ---------------------------------------------------------------- jobs

static vector<unique_ptr<Eval>> EVALS;

struct JobResult {
  int n = 0;
  double ms = 0;
  vector<double> diff;  // per arm: mean (arm - base)
  double baseMean = 0;
};

static atomic<long> LATEST_SEQ{0};

static JobResult runJob(const Snap& s, const Table& tb, const vector<Arm>& arms, int nMax, int budgetMs, int scenarioOffset = 0) {
  int nThreads = max(1, CFG.threads);
  while ((int)EVALS.size() < nThreads) {
    EVALS.emplace_back(new Eval(CFG.memoLog2));
    EVALS.back()->exactCap = CFG.exactCap;
  }
  auto t0 = Clock::now();
  auto deadline = t0 + chrono::milliseconds(budgetMs);
  atomic<int> next{0};
  vector<vector<double>> sums(nThreads, vector<double>(arms.size(), 0.0));
  vector<double> baseSum(nThreads, 0.0);
  vector<int> counts(nThreads, 0);
  auto work = [&](int ti) {
    Eval& ev = *EVALS[ti];
    Scen S;
    vector<double> val(arms.size());
    for (;;) {
      if (LATEST_SEQ.load() != s.seq) break;  // a newer snapshot arrived
      if (Clock::now() > deadline) break;
      int idx = next.fetch_add(1);
      if (idx >= nMax) break;
      makeScenario(s, (uint64_t)(scenarioOffset + idx), S);
      for (size_t a = 0; a < arms.size(); a++) val[a] = runArm(s, S, tb, arms[a], ev);
      for (size_t a = 0; a < arms.size(); a++) sums[ti][a] += val[a] - val[0];
      baseSum[ti] += val[0];
      counts[ti]++;
    }
  };
  vector<thread> th;
  for (int t = 1; t < nThreads; t++) th.emplace_back(work, t);
  work(0);
  for (auto& x : th) x.join();
  JobResult r;
  r.diff.assign(arms.size(), 0.0);
  double bs = 0;
  for (int t = 0; t < nThreads; t++) {
    r.n += counts[t];
    bs += baseSum[t];
    for (size_t a = 0; a < arms.size(); a++) r.diff[a] += sums[t][a];
  }
  if (r.n > 0) {
    for (auto& d : r.diff) d /= r.n;
    r.baseMean = bs / r.n;
  }
  r.ms = chrono::duration<double, milli>(Clock::now() - t0).count();
  return r;
}

// Combine disjoint scenario batches evaluated against the SAME rollout policy.
// The second checkpoint must not count the first samples twice or recursively
// blend their already-damped estimate into itself.
static JobResult mergeJobs(const JobResult& a, const JobResult& b) {
  if (a.n == 0) return b;
  if (b.n == 0) return a;
  JobResult out;
  out.n = a.n + b.n;
  out.ms = a.ms + b.ms;
  out.baseMean = (a.baseMean * a.n + b.baseMean * b.n) / out.n;
  out.diff.resize(a.diff.size());
  for (size_t k = 0; k < a.diff.size(); k++)
    out.diff[k] = (a.diff[k] * a.n + b.diff[k] * b.n) / out.n;
  return out;
}

// ---------------------------------------------------------------- protocol

static mutex OUT_MU;
static void emit(const string& line) {
  lock_guard<mutex> g(OUT_MU);
  fputs(line.c_str(), stdout);
  fputc('\n', stdout);
  fflush(stdout);
}

static string fmt(const char* f, ...) __attribute__((format(printf, 1, 2)));
static string fmt(const char* f, ...) {
  char buf[512];
  va_list ap;
  va_start(ap, f);
  vsnprintf(buf, sizeof buf, f, ap);
  va_end(ap);
  return buf;
}

static void parseCfg(istringstream& in) {
  string k;
  while (in >> k) {
    double v;
    if (!(in >> v)) break;
    if (k == "threads") CFG.threads = (int)v;
    else if (k == "seed") CFG.seed = (uint64_t)v;
    else if (k == "exactCap") CFG.exactCap = (int)v;
    else if (k == "memoLog2") CFG.memoLog2 = (int)v;
    else if (k == "allinK") CFG.allinK = (int)v;
    else if (k == "allinMinV") CFG.allinMinV = v;
    else if (k == "endgameK") CFG.endgameK = (int)v;
    else if (k == "minPlayLen") CFG.minPlayLen = (int)v;
    else if (k == "flushSize") CFG.flushSize = (int)v;
    else if (k == "nMax") CFG.nMax = (int)v;
    else if (k == "nFast") CFG.nFast = (int)v;
    else if (k == "nRefine") CFG.nRefine = (int)v;
    else if (k == "refineMs") CFG.refineMs = (int)v;
    else if (k == "bidSearch") CFG.bidSearch = max(0, min(2, (int)v));
    else if (k == "fastMs") CFG.fastMs = (int)v;
    else if (k == "fullMs") CFG.fullMs = (int)v;
    else if (k == "cashK") CFG.cashK = (int)v;
    else if (k == "coinArm") CFG.coinArm = (int)v;
    else if (k == "damp") CFG.damp = v;
    else if (k == "lamDamp") CFG.lamDamp = v;
    else if (k == "lamScale") CFG.lamScale = v;
  }
  for (auto& e : EVALS) e->exactCap = CFG.exactCap;
}

// SNAP seq matchId auctionId letter usableLeft balance score needCash
//      rack*26 U*26 nOpps (kind bal bonus slope jit | 4 bal pi (mu sd)*26)*nOpps
static bool parseSnap(istringstream& in, Snap& s) {
  int nc, nOpp;
  if (!(in >> s.seq >> s.matchId >> s.auctionId >> s.letter >> s.usableLeft >> s.balance >> s.score >> nc)) return false;
  s.needCash = nc != 0;
  for (int i = 0; i < 26; i++) {
    int x;
    if (!(in >> x)) return false;
    s.rack[i] = (uint8_t)x;
  }
  for (int i = 0; i < 26; i++) {
    int x;
    if (!(in >> x)) return false;
    s.U[i] = (uint8_t)x;
  }
  if (!(in >> nOpp)) return false;
  s.opps.clear();
  for (int i = 0; i < nOpp; i++) {
    Opp o;
    if (!(in >> o.kind >> o.bal)) return false;
    if (o.kind == PRIOR) {
      // kind 4: bal pi (mu sd)*26
      if (!(in >> o.pi)) return false;
      o.pi = min(0.999, max(0.0, o.pi));
      for (int L = 0; L < 26; L++) {
        if (!(in >> o.mu[L] >> o.sd[L])) return false;
        o.sd[L] = max(0.05f, o.sd[L]);
      }
    } else if (!(in >> o.bonus >> o.slope >> o.jit)) {
      return false;
    }
    s.opps.push_back(o);
  }
  return true;
}

static Table TABLE;
static string TABLE_MATCH;

static void resetTable() {
  for (int i = 0; i < 26; i++) TABLE.v[i] = VAL[i] + 1;  // cold-start prior
  TABLE.lam = 0.35;
}

static void sendResult(const char* phase, const Snap& s, const JobResult& r, const vector<pair<string, array<double, 3>>>& cash, Eval& ev0, double fairBid = -1) {
  string line = fmt("RES %s %ld %s %s %d %.1f %.4f %.3f", phase, s.seq, s.matchId.c_str(), s.auctionId.c_str(), r.n, r.ms, TABLE.lam * CFG.lamScale, r.baseMean);
  for (int i = 0; i < 26; i++) line += fmt(" %.4f", TABLE.v[i]);
  double hit = 0;
  long big = 0;
  for (auto& e : EVALS) {
    hit += e->hits;
    big += e->bigCalls;
  }
  long tot = 0;
  for (auto& e : EVALS) tot += e->hits + e->misses;
  line += fmt(" %.4f %ld", tot ? hit / tot : 0.0, big);
  line += fmt(" %d", (int)cash.size());
  for (auto& c : cash) line += fmt(" %s %.0f %.4f %.4f", c.first.c_str(), c.second[0], c.second[1], c.second[2]);
  line += fmt(" %.3f", fairBid);
  (void)ev0;
  emit(line);
}

static void handleSnap(const Snap& s) {
  if (s.matchId != TABLE_MATCH) {
    TABLE_MATCH = s.matchId;
    resetTable();
  }
  while (EVALS.empty()) {
    EVALS.emplace_back(new Eval(CFG.memoLog2));
    EVALS.back()->exactCap = CFG.exactCap;
  }

  auto blend = [&](const vector<Arm>& arms, const JobResult& r, int lamArm, const vector<int>& letterArm, const Table& prior) {
    if (r.n <= 0) return;
    for (size_t k = 0; k < letterArm.size(); k++) {
      int L = letterArm[k];
      if (L < 0) continue;
      double est = r.diff[k];
      TABLE.v[L] = prior.v[L] + CFG.damp * (est - prior.v[L]);
    }
    if (lamArm >= 0 && CFG.coinArm > 0) {
      double est = r.diff[lamArm] / CFG.coinArm;
      est = max(CFG.lamMin, min(CFG.lamMax, est));
      TABLE.lam = prior.lam + CFG.lamDamp * (est - prior.lam);
      TABLE.lam = max(CFG.lamMin, min(CFG.lamMax, TABLE.lam));
    }
    (void)arms;
  };

  bool pricedCurrent = false;

  // --- fast pass: base, coin arm, and the open letter
  {
    vector<Arm> arms(1);
    vector<int> letterArm(1, -1);  // index in arms -> letter
    Arm coin;
    coin.addCoins = CFG.coinArm;
    arms.push_back(coin);
    letterArm.push_back(-1);
    int freeArm = -1;
    vector<pair<int, int>> paidArms; // payment price, arm index
    if (s.letter >= 0) {
      Arm a;
      a.addLetter = s.letter;
      freeArm = (int)arms.size();
      arms.push_back(a);
      letterArm.push_back(s.letter);
      const int meanPrice = min(s.balance, max(1, (int)round(expectedOppPrice(s, s.letter))));
      vector<int> prices;
      if (CFG.bidSearch == 2) {
        for (int price = 1; price <= s.balance; price++) prices.push_back(price);
      } else {
        prices = {max(1, meanPrice / 2), meanPrice, s.balance};
      }
      for (int price : prices) {
        if (price <= 0 || price > s.balance || (!paidArms.empty() && paidArms.back().first == price)) continue;
        Arm paid;
        paid.addLetter = s.letter;
        paid.addCoins = -price;
        paidArms.push_back({price, (int)arms.size()});
        arms.push_back(paid);
        letterArm.push_back(-1);
      }
    }
    const Table prior = TABLE;
    JobResult r = runJob(s, prior, arms, CFG.nFast, CFG.fastMs);
    if (LATEST_SEQ.load() != s.seq) return;
    blend(arms, r, 1, letterArm, prior);
    pricedCurrent = s.letter >= 0 && r.n > 0;
    vector<pair<int, double>> paidCurve;
    if (r.n > 0 && freeArm >= 0) {
      for (const auto& [price, armIndex] : paidArms) paidCurve.push_back({price, r.diff[armIndex]});
    }
    const double fastFairBid = paidCurve.empty() ? -1 : fairBidFromCurve(r.diff[freeArm], paidCurve);
    sendResult("fast", s, r, {}, *EVALS[0], fastFairBid);

    // Refine only this auction's letter and coin value before spending time
    // on the all-letter table and cash-in candidates. Continue with distinct
    // scenarios under the original policy and publish a cumulative estimate.
    const int firstRefineMs = CFG.bidSearch == 1 ? max(1, CFG.refineMs / 2) : CFG.refineMs;
    if (s.letter >= 0 && CFG.nRefine > r.n && firstRefineMs > 0) {
      JobResult extra = runJob(s, prior, arms, CFG.nRefine - r.n, firstRefineMs, r.n);
      if (LATEST_SEQ.load() != s.seq) return;
      if (extra.n > 0) {
        pricedCurrent = true;
        JobResult combined = mergeJobs(r, extra);
        blend(arms, combined, 1, letterArm, prior);
        paidCurve.clear();
        for (const auto& [price, armIndex] : paidArms) paidCurve.push_back({price, combined.diff[armIndex]});
        double refinedFairBid = fairBidFromCurve(combined.diff[freeArm], paidCurve);
        sendResult("refine", s, combined, {}, *EVALS[0], refinedFairBid);

        // Adaptive mode spends a second, paired batch near the estimated
        // zero crossing. Reuse identical future scenarios for base, free and
        // nearby paid arms so the local comparison shares its random draws.
        if (CFG.bidSearch == 1 && refinedFairBid > 0 && s.balance > 1 && CFG.refineMs > 0) {
          const int center = (int)round(refinedFairBid);
          vector<int> localPrices;
          for (int delta = -2; delta <= 2; delta++) {
            int price = max(1, min(s.balance, center + delta));
            if (localPrices.empty() || localPrices.back() != price) localPrices.push_back(price);
          }
          vector<Arm> localArms(1);
          int localFree = (int)localArms.size();
          Arm freeLetter; freeLetter.addLetter = s.letter;
          localArms.push_back(freeLetter);
          vector<pair<int, int>> localPaid;
          for (int price : localPrices) {
            Arm paid; paid.addLetter = s.letter; paid.addCoins = -price;
            localPaid.push_back({price, (int)localArms.size()});
            localArms.push_back(paid);
          }
          const int localBudgetMs = CFG.refineMs - firstRefineMs;
          JobResult local = localBudgetMs > 0
            ? runJob(s, prior, localArms, max(200, CFG.nRefine / 2), localBudgetMs, combined.n)
            : JobResult{};
          if (LATEST_SEQ.load() != s.seq) return;
          if (local.n > 0) {
            vector<pair<int, double>> localCurve;
            for (const auto& [price, armIndex] : localPaid) localCurve.push_back({price, local.diff[armIndex]});
            const double localFairBid = fairBidFromCurve(local.diff[localFree], localCurve);
            sendResult("search", s, local, {}, *EVALS[0], localFairBid);
          }
        }
      }
    }
  }

  // --- broad pass: other letters and cash-in candidates. Preserve the
  // focused current-letter/coin estimate instead of overwriting it with an
  // independent lower-sample run on a changed continuation policy.
  {
    vector<Arm> arms(1);
    vector<int> letterArm(1, -1);
    int lamArm = -1;
    if (!pricedCurrent) {
      Arm coin;
      coin.addCoins = CFG.coinArm;
      lamArm = (int)arms.size();
      arms.push_back(coin);
      letterArm.push_back(-1);
    }
    for (int L = 0; L < 26; L++) {
      if (pricedCurrent && L == s.letter) continue;
      if (s.U[L] > 0 || L == s.letter) {
        Arm a;
        a.addLetter = L;
        arms.push_back(a);
        letterArm.push_back(L);
      }
    }
    size_t firstCash = arms.size();
    vector<int> candEntries;
    if (s.needCash && Eval::total(s.rack.data()) >= 2) {
      Eval& ev = *EVALS[0];
      vector<int> plan = ev.plan(s.rack);
      for (int e : plan) candEntries.push_back(e);
      vector<int> fit;
      Rack w = s.rack;
      auto cb = [&](int e) { fit.push_back(e); };
      walkAll(0, w.data(), cb);
      sort(fit.begin(), fit.end(), [&](int x, int y) { return D.rewards[x] != D.rewards[y] ? D.rewards[x] > D.rewards[y] : x < y; });
      for (size_t k = 0; k < fit.size() && (int)k < CFG.cashK; k++) {
        if (find(candEntries.begin(), candEntries.end(), fit[k]) == candEntries.end()) candEntries.push_back(fit[k]);
      }
      for (int e : candEntries) {
        Arm a;
        a.cashEntry = e;
        arms.push_back(a);
        letterArm.push_back(-1);
      }
    }
    const Table prior = TABLE;
    JobResult r = runJob(s, prior, arms, CFG.nMax, CFG.fullMs);
    if (LATEST_SEQ.load() != s.seq) return;
    blend(arms, r, lamArm, letterArm, prior);
    vector<pair<string, array<double, 3>>> cash;
    if (r.n > 0) {
      for (size_t k = 0; k < candEntries.size(); k++) {
        int e = candEntries[k];
        double gain = r.diff[firstCash + k];
        cash.push_back({D.words[e], {(double)D.rewards[e], gain, gain}});
      }
      sort(cash.begin(), cash.end(), [](auto& a, auto& b) { return a.second[2] > b.second[2]; });
      if (cash.size() > 8) cash.resize(8);
    }
    sendResult("full", s, r, cash, *EVALS[0], -1);
  }
}

// ---------------------------------------------------------------- main

static mutex MU;
static condition_variable CV;
static bool HAVE_PENDING = false, EOF_SEEN = false;
static Snap PENDING;

static void readerLoop() {
  string line;
  while (getline(cin, line)) {
    istringstream in(line);
    string cmd;
    in >> cmd;
    if (cmd == "CFG") {
      lock_guard<mutex> g(MU);
      parseCfg(in);
    } else if (cmd == "SNAP") {
      Snap s;
      if (!parseSnap(in, s)) {
        emit("ERR bad snapshot");
        continue;
      }
      LATEST_SEQ.store(s.seq);
      {
        lock_guard<mutex> g(MU);
        PENDING = s;
        HAVE_PENDING = true;
      }
      CV.notify_one();
    } else if (cmd == "CANCEL") {
      LATEST_SEQ.store(-1);
    } else if (cmd == "QUIT") {
      break;
    }
  }
  {
    lock_guard<mutex> g(MU);
    EOF_SEEN = true;
  }
  CV.notify_one();
}

// `engine --eval`: read racks (26 counts per line), print exact B and the plan.
static int evalMode() {
  Eval ev(22);
  ev.exactCap = 99;
  string line;
  while (getline(cin, line)) {
    istringstream in(line);
    Rack r{};
    bool ok = true;
    for (int i = 0; i < 26; i++) {
      int x;
      if (!(in >> x)) {
        ok = false;
        break;
      }
      r[i] = (uint8_t)x;
    }
    if (!ok) continue;
    int v;
    vector<int> p = ev.plan(r, &v);
    string out = fmt("%d", v);
    for (int e : p) out += " " + D.words[e];
    puts(out.c_str());
  }
  return 0;
}

int main(int argc, char** argv) {
  string wordlist = "../spacetimedb/wordlist.txt";
  bool eval = false;
  for (int i = 1; i < argc; i++) {
    string a = argv[i];
    if (a == "--eval") eval = true;
    else if (a == "--wordlist" && i + 1 < argc) wordlist = argv[++i];
  }
  ios::sync_with_stdio(false);
  if (!D.load(wordlist)) {
    fprintf(stderr, "cannot read wordlist %s\n", wordlist.c_str());
    return 2;
  }
  if (eval) return evalMode();
  resetTable();
  emit(fmt("READY %zu %zu", D.words.size(), D.let.size()));
  thread reader(readerLoop);
  for (;;) {
    Snap s;
    {
      unique_lock<mutex> g(MU);
      CV.wait(g, [] { return HAVE_PENDING || EOF_SEEN; });
      if (!HAVE_PENDING && EOF_SEEN) break;
      s = PENDING;
      HAVE_PENDING = false;
    }
    if (LATEST_SEQ.load() != s.seq) continue;  // superseded while queued
    try {
      handleSnap(s);
    } catch (const exception& e) {
      emit(string("ERR ") + e.what());
    }
  }
  reader.join();
  return 0;
}
