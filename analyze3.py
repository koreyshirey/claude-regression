#!/usr/bin/env python3
"""Three claims, measured per request.

  1. correction burden -- output tokens spent on turns where Claude ends up
     conceding, vs turns where it doesn't.
  2. context Korey must supply -- characters he types per request, and in the
     opening brief of each session.
  3. reasoning quality -- thinking tokens, tool churn, redundant work
     (duplicate commands, re-reads), failed edits, mid-task self-reversals.

An "episode" = one human prose turn plus every assistant message until the
next human turn. That is the unit Korey actually experiences.
"""
import json, glob, os, re, math, statistics, collections

ROOT = os.path.expanduser("~/.claude/projects")
here = os.path.dirname(os.path.abspath(__file__))
exec(open(os.path.join(here, "analyze.py")).read()
     .split("# ---------------------------------------------------------------- counters")[0])
exec(open(os.path.join(here, "analyze2.py")).read()
     .split("class S:")[0].split("LOOKBACK_ASST = 3")[1])

# Claude reversing itself mid-task.
REVERSAL = re.compile(r"""(?ix)
      \bactually,?\ (?:no|wait|the|it|that|I)\b
    | \bwait[,.—-]
    | \bon\ second\ thought\b
    | \blet\ me\ reconsider\b
    | \bhold\ on\b
    | \bthat'?s\ not\ right\b
    | \bI\ was\ about\ to\b
    | \bscratch\ that\b
    | \bcorrection[:,]
""")


def toklen(s):
    return len(s) / 4.0     # crude but consistent char->token proxy


class Ep:
    """One request episode."""
    __slots__ = ("model", "week", "human_chars", "out_tok", "think_chars",
                 "tools", "admitted", "pushed", "reversals", "edit_fail",
                 "dup_cmd", "reread", "is_first")

    def __init__(self, model, week):
        self.model, self.week = model, week
        self.human_chars = self.out_tok = self.think_chars = self.tools = 0
        self.reversals = self.edit_fail = self.dup_cmd = self.reread = 0
        self.admitted = self.pushed = self.is_first = False


episodes = []

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

    ep = None
    seen_cmds, seen_files = collections.Counter(), collections.Counter()
    first_done = False

    for d in rows:
        ts = d.get("timestamp")
        if not ts:
            continue
        msg = d.get("message") or {}

        if d.get("type") == "user":
            if d.get("isSidechain") or d.get("isMeta") or d.get("isCompactSummary"):
                continue
            c = msg.get("content")
            if isinstance(c, list) and any(
                    isinstance(b, dict) and b.get("type") == "tool_result" for b in c):
                for b in c:
                    if isinstance(b, dict) and b.get("type") == "tool_result" and b.get("is_error"):
                        body = b.get("content")
                        body = body if isinstance(body, str) else json.dumps(body)
                        if ep and re.search(r"(?i)string to replace|not found in file|no changes", body):
                            ep.edit_fail += 1
                continue
            t = text_of(msg).strip()
            if not t or t.startswith("[Request interrupted"):
                continue
            if SYNTHETIC_PREFIX.match(t) or INJECTED.match(t) or is_paste(t):
                continue
            if ep:
                episodes.append(ep)
            ep = Ep(cur_model, week_of(ts))
            ep.human_chars = len(t)
            ep.pushed = bool(CORRECT.search(t))
            ep.is_first = not first_done
            first_done = True

        elif d.get("type") == "assistant":
            m = msg.get("model")
            if not m or m == "<synthetic>":
                continue
            cur_model = m
            if ep is None:
                continue
            ep.model = m
            u = msg.get("usage") or {}
            ep.out_tok += u.get("output_tokens", 0) or 0
            c = msg.get("content")
            if not isinstance(c, list):
                continue
            for b in c:
                if not isinstance(b, dict):
                    continue
                if b.get("type") == "thinking":
                    ep.think_chars += len(b.get("thinking", "") or "")
                elif b.get("type") == "text":
                    txt = b.get("text", "") or ""
                    if ADMIT.search(txt):
                        ep.admitted = True
                    ep.reversals += len(REVERSAL.findall(txt))
                elif b.get("type") == "tool_use":
                    ep.tools += 1
                    name, inp = b.get("name"), b.get("input") or {}
                    if name == "Bash":
                        cmd = (inp.get("command") or "").strip()
                        if cmd:
                            seen_cmds[cmd] += 1
                            if seen_cmds[cmd] > 1:
                                ep.dup_cmd += 1
                    elif name == "Read":
                        fp = inp.get("file_path") or ""
                        if fp:
                            seen_files[fp] += 1
                            if seen_files[fp] > 1:
                                ep.reread += 1
    if ep:
        episodes.append(ep)

# ---------------------------------------------------------------- aggregate

MODELS = ["claude-opus-4-8", "claude-opus-5", "claude-sonnet-4-6", "claude-sonnet-5"]


def med(xs):
    return statistics.median(xs) if xs else 0


def mean(xs):
    return statistics.fmean(xs) if xs else 0


def mwu(a, b):
    """Mann-Whitney U -> normal approx two-sided p. Robust to skew."""
    if not a or not b:
        return 1.0
    comb = sorted([(v, 0) for v in a] + [(v, 1) for v in b])
    ranks, i = {}, 0
    rs = [0.0] * len(comb)
    while i < len(comb):
        j = i
        while j + 1 < len(comb) and comb[j + 1][0] == comb[i][0]:
            j += 1
        r = (i + j) / 2 + 1
        for k in range(i, j + 1):
            rs[k] = r
        i = j + 1
    r1 = sum(rs[k] for k in range(len(comb)) if comb[k][1] == 0)
    n1, n2 = len(a), len(b)
    u1 = r1 - n1 * (n1 + 1) / 2
    mu = n1 * n2 / 2
    sd = math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    if sd == 0:
        return 1.0
    return math.erfc(abs(u1 - mu) / sd / math.sqrt(2))


eps_by = {m: [e for e in episodes if e.model == m] for m in MODELS}

print("=" * 100)
print("PER-EPISODE MEDIANS  (one episode = one thing Korey asked for)")
print("=" * 100)
hdr = (f"{'model':<20}{'eps':>5}{'humChar':>9}{'outTok':>8}{'thinkTok':>9}"
       f"{'tools':>7}{'rev':>6}{'editFail':>9}{'dupCmd':>8}{'reRead':>8}")
print(hdr); print("-" * len(hdr))
for m in MODELS:
    E = eps_by[m]
    if not E:
        continue
    print(f"{m:<20}{len(E):>5}"
          f"{med([e.human_chars for e in E]):>9.0f}"
          f"{med([e.out_tok for e in E]):>8.0f}"
          f"{med([toklen('x'*e.think_chars) for e in E]):>9.0f}"
          f"{med([e.tools for e in E]):>7.0f}"
          f"{mean([e.reversals for e in E]):>6.2f}"
          f"{mean([e.edit_fail for e in E]):>9.2f}"
          f"{mean([e.dup_cmd for e in E]):>8.2f}"
          f"{mean([e.reread for e in E]):>8.2f}")

print()
print("=" * 100)
print("CLAIM 1 -- CORRECTION BURDEN")
print("=" * 100)
for m in MODELS:
    E = eps_by[m]
    if not E:
        continue
    adm = [e for e in E if e.admitted]
    tot_out = sum(e.out_tok for e in E) or 1
    adm_out = sum(e.out_tok for e in adm)
    print(f"  {m:<20} episodes ending in a concession: {len(adm):>3}/{len(E):<4} "
          f"({100*len(adm)/len(E):5.1f}%)   "
          f"output tokens burned in them: {100*adm_out/tot_out:5.1f}%")
a, b = eps_by["claude-opus-4-8"], eps_by["claude-opus-5"]
print(f"\n  median output tokens, concession episodes vs clean:")
for lbl, E in (("opus-4-8", a), ("opus-5", b)):
    print(f"    {lbl:<10} clean={med([e.out_tok for e in E if not e.admitted]):>6.0f}  "
          f"concession={med([e.out_tok for e in E if e.admitted]):>6.0f}")

print()
print("=" * 100)
print("CLAIM 2 -- CONTEXT KOREY MUST SUPPLY")
print("=" * 100)
for m in MODELS:
    E = eps_by[m]
    if not E:
        continue
    op = [e.human_chars for e in E if e.is_first]
    print(f"  {m:<20} median chars/request={med([e.human_chars for e in E]):>6.0f}  "
          f"mean={mean([e.human_chars for e in E]):>7.0f}  "
          f"opening brief median={med(op):>6.0f} (n={len(op)})")
print(f"\n  chars/request opus-4-8 vs opus-5   Mann-Whitney p="
      f"{mwu([e.human_chars for e in a], [e.human_chars for e in b]):.4f}")

print()
print("=" * 100)
print("CLAIM 3 -- REASONING / CHURN  (opus-4-8 -> opus-5, Mann-Whitney)")
print("=" * 100)
for name, get in [("output tokens per request", lambda e: e.out_tok),
                  ("thinking tokens per request", lambda e: e.think_chars / 4),
                  ("tool calls per request", lambda e: e.tools),
                  ("self-reversals per request", lambda e: e.reversals),
                  ("failed edits per request", lambda e: e.edit_fail),
                  ("duplicate commands per request", lambda e: e.dup_cmd),
                  ("repeat file reads per request", lambda e: e.reread)]:
    xa, xb = [get(e) for e in a], [get(e) for e in b]
    p = mwu(xa, xb)
    print(f"  {name:<32} median {med(xa):>7.1f} -> {med(xb):>7.1f}   "
          f"mean {mean(xa):>7.1f} -> {mean(xb):>7.1f}   p={p:.4f}"
          f"{'  SIGNIFICANT' if p < 0.05 else ''}")

wk = collections.defaultdict(list)
for e in episodes:
    wk[e.week].append(e)
print()
print("BY WEEK (medians)")
print(f"{'week':<12}{'eps':>5}{'humChar':>9}{'outTok':>8}{'thinkTok':>9}{'tools':>7}{'concede%':>10}")
for w in sorted(wk):
    E = wk[w]
    print(f"{w:<12}{len(E):>5}{med([e.human_chars for e in E]):>9.0f}"
          f"{med([e.out_tok for e in E]):>8.0f}"
          f"{med([e.think_chars/4 for e in E]):>9.0f}"
          f"{med([e.tools for e in E]):>7.0f}"
          f"{100*sum(1 for e in E if e.admitted)/len(E):>10.1f}")

json.dump([{s: getattr(e, s) for s in Ep.__slots__} for e in episodes],
          open(os.path.join(here, "episodes.json"), "w"))
print(f"\nwrote episodes.json ({len(episodes)} episodes)")
