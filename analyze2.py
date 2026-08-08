#!/usr/bin/env python3
"""Sequence-aware pass: mistakes as Korey experiences them.

Fixes two problems with analyze.py:
  1. `tool_err` only caught mechanical failures. Semantic mistakes (wrong part
     number, wrong diagnosis, ignored constraint) never set is_error. Here a
     mistake is inferred from an admission that FOLLOWS user pushback -- i.e.
     a mistake Korey had to catch.
  2. Rates were per assistant message. Korey experiences cost per *request*,
     so everything is normalised per human turn and per session too.

Also measures repeated instructions: re-raising the same specific content
after already having said it.
"""
import json, glob, os, re, math, collections, datetime

ROOT = os.path.expanduser("~/.claude/projects")
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "analyze.py")).read()
exec(src.split("# ---------------------------------------------------------------- counters")[0])

LOOKBACK_ASST = 3     # how many assistant msgs back an admission may reach for its trigger

# An admission that explicitly credits Korey with catching it. This is the
# cleanest available evidence that a mistake required human correction --
# Claude cannot say "you're right" unless Korey just contradicted it.
USER_CREDIT = re.compile(r"""(?ix)
      \byou'?re\ (?:absolutely\ |completely\ |totally\ |quite\ )?(?:right|correct)\b
    | \byou\ are\ (?:absolutely\ |completely\ )?(?:right|correct)\b
    | \bgood\ catch\b
    | \bas\ you\ (?:said|pointed\ out|noted)\b
    | \byou'?re\ right\ to\ (?:push|question|call)\b
    | \bfair\ point\b
    | \byou\ caught\b
    | \bthanks\ for\ (?:catching|the\ correction)\b
""")

# Sensitivity sweep for "re-raised an instruction I'd already given".
RERAISE_LEVELS = [("loose", 3, 0.30), ("mid", 4, 0.45), ("strict", 5, 0.60)]


def wilson_pair(k, n):
    return wilson(k, n)


def ztest(k1, n1, k2, n2):
    if not n1 or not n2:
        return 1.0
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return 1.0
    z = (k2 / n2 - k1 / n1) / se
    return math.erfc(abs(z) / math.sqrt(2))


def rate_ratio(k1, n1, k2, n2):
    r1 = k1 / n1 if n1 else 0
    r2 = k2 / n2 if n2 else 0
    return (100 * r1, 100 * r2, (r2 / r1) if r1 else float("inf"))


# "Specific" tokens: identifiers, part numbers, paths, long words. Repetition of
# these is meaningful; repetition of "the thing" is not.
def specific_tokens(s):
    out = set()
    for w in re.findall(r"[A-Za-z0-9_./-]{3,}", s.lower()):
        if w in STOPWORDS:
            continue
        if any(c.isdigit() for c in w) or "/" in w or "." in w or "_" in w or len(w) >= 6:
            out.add(w)
    return out


class S:
    def __init__(self):
        self.d = collections.Counter()
        self.sessions = set()


by_model = collections.defaultdict(S)
by_week = collections.defaultdict(S)
ex = collections.defaultdict(list)


def bump(model, week, key, n=1):
    by_model[model].d[key] += n
    by_week[week].d[key] += n


for path in sorted(glob.glob(os.path.join(ROOT, "**", "*.jsonl"), recursive=True)):
    rows = []
    for line in open(path, errors="replace"):
        try:
            rows.append(json.loads(line))
        except Exception:
            pass

    cur_model = next((m for r in rows if r.get("type") == "assistant"
                      for m in [(r.get("message") or {}).get("model")]
                      if m and m != "<synthetic>"), "unknown")

    seq = []          # ordered ('h'|'a', ts, model, text, pushback?)
    for d in rows:
        ts = d.get("timestamp")
        if not ts:
            continue
        msg = d.get("message") or {}
        if d.get("type") == "assistant":
            m = msg.get("model")
            if not m or m == "<synthetic>":
                continue
            cur_model = m
            seq.append(("a", ts, m, text_of(msg)))
        elif d.get("type") == "user":
            if d.get("isSidechain") or d.get("isMeta") or d.get("isCompactSummary"):
                continue
            c = msg.get("content")
            if isinstance(c, list) and any(
                isinstance(b, dict) and b.get("type") == "tool_result" for b in c):
                continue
            t = text_of(msg).strip()
            if not t or t.startswith("[Request interrupted"):
                continue
            if SYNTHETIC_PREFIX.match(t) or INJECTED.match(t) or is_paste(t):
                continue
            seq.append(("h", ts, cur_model, t))

    if not seq:
        return_early = False
    sid = os.path.basename(path)
    prior_specific = []   # (tokenset, text) of earlier human turns this session

    for i, (kind, ts, model, text) in enumerate(seq):
        wk = week_of(ts)
        by_model[model].sessions.add(sid)
        by_week[wk].sessions.add(sid)

        if kind == "h":
            bump(model, wk, "human_turns")
            pushed = bool(CORRECT.search(text))
            if pushed:
                bump(model, wk, "pushback")

            toks = specific_tokens(text)
            if len(toks) >= 3:
                for lvl, need, frac in RERAISE_LEVELS:
                    for ptoks, ptext in prior_specific:
                        shared = toks & ptoks
                        if len(shared) >= need and \
                           len(shared) / min(len(toks), len(ptoks)) >= frac:
                            bump(model, wk, "reraise_" + lvl)
                            if lvl == "loose":
                                ex["reraise"].append((ts, model, sorted(shared)[:6],
                                                      text[:150].replace("\n", " "),
                                                      ptext[:150].replace("\n", " ")))
                            break
                prior_specific.append((toks, text))

        else:  # assistant
            bump(model, wk, "asst")
            if not ADMIT.search(text):
                continue
            bump(model, wk, "admit")
            if USER_CREDIT.search(text):
                bump(model, wk, "admit_credits_user")
                ex["credit"].append((ts, model, text[:150].replace("\n", " ")))
            else:
                bump(model, wk, "admit_self")
            # Was this admission triggered by Korey pushing back?
            trigger = None
            seen_asst = 0
            for j in range(i - 1, -1, -1):
                if seq[j][0] == "a":
                    seen_asst += 1
                    if seen_asst > LOOKBACK_ASST:
                        break
                    continue
                trigger = seq[j]
                break
            if trigger and CORRECT.search(trigger[3]):
                bump(model, wk, "admit_corrected")
                ex["corrected"].append((ts, model, trigger[3][:130].replace("\n", " "),
                                        text[:130].replace("\n", " ")))
            else:
                bump(model, wk, "admit_unprompted")


def report(title, table, order):
    print("=" * 96)
    print(title)
    print("=" * 96)
    hdr = (f"{'bucket':<16}{'sess':>5}{'human':>7}{'asst':>7}{'a/h':>6}"
           f"{'admits':>8}{'/100h':>7}"
           f"{'credit-u':>9}{'/100h':>7}{'self':>6}"
           f"{'reraise':>9}{'/100h':>7}")
    print(hdr)
    print("-" * len(hdr))
    for k in order:
        b = table[k]
        d, ns = b.d, len(b.sessions)
        h = d["human_turns"]
        pc = lambda x: (100 * x / h) if h else 0
        print(f"{k:<16}{ns:>5}{h:>7}{d['asst']:>7}"
              f"{(d['asst']/h if h else 0):>6.1f}"
              f"{d['admit']:>8}{pc(d['admit']):>7.1f}"
              f"{d['admit_credits_user']:>9}{pc(d['admit_credits_user']):>7.1f}"
              f"{d['admit_self']:>6}"
              f"{d['reraise_loose']:>9}{pc(d['reraise_loose']):>7.1f}")
    print()


models = [m for m in sorted(by_model, key=lambda k: -by_model[k].d["asst"])
          if by_model[m].d["asst"] > 300]
report("PER-REQUEST VIEW  (normalised by human turn and by session)", by_model, models)
report("BY WEEK", by_week, sorted(by_week))

a, b = by_model["claude-opus-4-8"].d, by_model["claude-opus-5"].d
na, nb = a["human_turns"], b["human_turns"]
print("=" * 96)
print("opus-4-8  ->  opus-5     (denominator = human turns, not assistant messages)")
print("=" * 96)
for name, key in [("admissions (any)", "admit"),
                  ("admissions crediting Korey ('you're right')", "admit_credits_user"),
                  ("admissions not crediting Korey", "admit_self"),
                  ("admissions after explicit pushback", "admit_corrected"),
                  ("pushback turns", "pushback"),
                  ("re-raise (loose)", "reraise_loose"),
                  ("re-raise (mid)", "reraise_mid"),
                  ("re-raise (strict)", "reraise_strict")]:
    r1, r2, ratio = rate_ratio(a[key], na, b[key], nb)
    p = ztest(a[key], na, b[key], nb)
    flag = "SIGNIFICANT" if p < 0.05 else "ns"
    print(f"  {name:<34} {r1:6.2f}% -> {r2:6.2f}% per human turn "
          f"({a[key]}/{na} -> {b[key]}/{nb})  x{ratio:.2f}  p={p:.4f}  {flag}")

print()
print("share of admissions that explicitly credit Korey with the catch:")
for m in models:
    d = by_model[m].d
    tot = d["admit"]
    if tot:
        print(f"  {m:<20} {100*d['admit_credits_user']/tot:5.1f}%  "
              f"({d['admit_credits_user']}/{tot})")

json.dump({m: dict(by_model[m].d, sessions=len(by_model[m].sessions)) for m in by_model},
          open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "metrics2.json"), "w"),
          indent=2)
print("\n--- sample re-raised instructions ---")
for ts, m, shared, now, pre in ex["reraise"][:10]:
    print(f"[{ts[:10]} {m[7:]}] shared={shared}\n   NOW: {now}\n   PRE: {pre}")
