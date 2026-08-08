#!/usr/bin/env python3
"""Measure Claude behaviour drift across local Claude Code transcripts.

Reads ~/.claude/projects/**/*.jsonl and emits per-week and per-model rates for:
  - self-admitted mistakes (apology / "you're right" language)
  - tool errors and user-rejected tool calls
  - user interruptions
  - user repetition (having to restate the same instruction)
  - user correction/frustration markers

Every rate is reported with a Wilson 95% interval so small buckets are
visibly small rather than silently noisy.
"""
import json, glob, os, re, math, collections, datetime, sys

ROOT = os.path.expanduser("~/.claude/projects")

# ---------------------------------------------------------------- patterns

# Claude conceding it got something wrong.
ADMIT = re.compile(r"""(?ix)
      \byou'?re\ (?:absolutely\ |completely\ |totally\ |quite\ )?right\b
    | \byou\ are\ (?:absolutely\ |completely\ )?right\b
    | \bmy\ (?:mistake|error|bad|apologies)\b
    | \bi\ apologi[sz]e\b
    | \bsorry[,.\ ]
    | \bi\ was\ wrong\b
    | \bi\ (?:incorrectly|wrongly|mistakenly|erroneously)\ \w+
    | \bthat\ was\ (?:wrong|incorrect|my\ error)\b
    | \bi\ (?:missed|misread|misunderstood|overlooked|conflated)\b
    | \bgood\ catch\b
    | \bi\ should\ have\b
    | \bi\ (?:made\ (?:a|an)\ (?:mistake|error))\b
    | \blet\ me\ (?:correct|fix)\ that\b
""")

# STRONG pushback: unambiguous "you got this wrong / I already told you".
CORRECT = re.compile(r"""(?ix)
      ^\s*no[,.!\ ]
    | \bthat'?s\ (?:not|wrong|incorrect)\b
    | \bthat\ is\ not\ (?:what|one\ of)\b
    | \bi\ (?:said|told\ you|already\ (?:said|told|asked|explained))\b
    | \bi\ asked\ you\ to\b
    | \byou\ (?:didn'?t|did\ not|never)\ \w+
    | \byou\ (?:keep|still)\ \w+
    | \byou\ are\ (?:really\ )?(?:bad|wrong)\b
    | \bstop\ (?:assuming|doing|trying|making|changing|adding)\b
    | \b(?:is|thats|that'?s|it'?s)\ wrong\b
    | \bwhy\ (?:did|are|would)\ you\b
    | \blisten\ to\ me\b
    | \bread\ (?:what\ i|the\ file|it\ again)\b
    | \bagain\?\ *$
""")

# WEAK: ordinary directives. Tracked separately -- these are NOT corrections.
DIRECTIVE = re.compile(r"(?ix)\bdon'?t\ \w+ | \bjust\ do\b | \bi\ don'?t\ want\b")

# IDE/system injections masquerading as user turns.
INJECTED = re.compile(r"^\s*<(?:ide_opened_file|ide_selection|ide_diagnostics)")

# Pasted terminal/log output rather than the user talking.
def is_paste(t):
    lines = t.splitlines()
    if re.search(r"(?m)^\s*\S+@\S+\s+\S*\s*\$\s", t):      # shell prompt
        return True
    if re.search(r"(?i)\b(?:ERC|DRC) report \(", t):        # pasted tool report
        return True
    logish = sum(bool(re.match(r"\s*(?:\[[\d:.]+\]|\[[DWIEV]\]|\{|<|/\w+/|\w+\s+\|)", l))
                 for l in lines)
    return len(lines) >= 3 and logish >= max(3, len(lines) * 0.4)

PROFANITY = re.compile(r"(?i)\b(fuck\w*|shit|damn(?:it)?|wtf|bullshit|christ|jesus)\b")

# Non-human user records that must never count as a human turn.
SYNTHETIC_PREFIX = re.compile(
    r"^\s*(\[Request interrupted|<task-notification>|<command-name>|<command-message>"
    r"|<local-command|<bash-input>|<bash-stdout>|Caveat:|<system-reminder>"
    r"|<user-prompt-submit-hook>|<session-start-hook>|API Error|\[Tool"
    r"|This session is being continued from)"
)

# "I already told you this" -- the user explicitly flagging a repeat.
RESTATE_MARK = re.compile(
    r"(?ix)\b(?:as|like)\ i\ (?:said|told|mentioned|explained)\b"
    r"| \bi\ already\ (?:said|told|asked|explained|mentioned)\b"
    r"| \bi\ (?:said|told\ you)\ (?:this|that|before|already)\b"
    r"| \bfor\ the\ (?:second|third|last)\ time\b"
    r"| \bagain[,:]\ "
)

STOPWORDS = set("""a an the and or but if then so to of in on at for with from by as is are was were
be been being it its this that these those i you he she they we me my your our their do does did
done doing have has had can could should would will shall may might must not no yes ok okay just
please can't cant dont don't im i'm ive i've thats that's what when where which who how why all any
now here there also too very really much more most some one two get got make made use used using
need needs want wants like about into out up down over under again still yet than them he's she's""".split())

REJECT_MARK = "user doesn't want to proceed"


def wilson(k, n, z=1.96):
    """95% Wilson interval for a proportion, returned as percentages."""
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * p, 100 * max(0.0, c - h), 100 * min(1.0, c + h))


def text_of(msg):
    c = msg.get("content")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "\n".join(
            b.get("text", "") for b in c
            if isinstance(b, dict) and b.get("type") == "text"
        )
    return ""


def content_words(s):
    ws = re.findall(r"[a-z0-9_/.-]{3,}", s.lower())
    return {w for w in ws if w not in STOPWORDS}


def jaccard(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def week_of(ts):
    d = datetime.datetime.fromisoformat(ts.replace("Z", "+00:00"))
    monday = d.date() - datetime.timedelta(days=d.weekday())
    return monday.isoformat()


# ---------------------------------------------------------------- counters

class Bucket:
    __slots__ = ("asst_msgs", "asst_admit", "tool_calls", "tool_err", "tool_reject",
                 "human_turns", "prose_turns", "human_correct", "human_directive",
                 "human_profane", "human_repeat", "human_restate_mark",
                 "interrupts", "sessions")

    def __init__(self):
        for s in self.__slots__:
            setattr(self, s, set() if s == "sessions" else 0)

    def add(self, o):
        for s in self.__slots__:
            if s == "sessions":
                self.sessions |= o.sessions
            else:
                setattr(self, s, getattr(self, s) + getattr(o, s))


by_week = collections.defaultdict(Bucket)
by_model = collections.defaultdict(Bucket)
examples = collections.defaultdict(list)


def bump(week, model, field, n=1, sid=None):
    for b in (by_week[week], by_model[model]):
        if field == "sessions":
            b.sessions.add(sid)
        else:
            setattr(b, field, getattr(b, field) + n)


# ---------------------------------------------------------------- main pass

files = sorted(glob.glob(os.path.join(ROOT, "**", "*.jsonl"), recursive=True))
for path in files:
    rows = []
    for line in open(path, errors="replace"):
        try:
            rows.append(json.loads(line))
        except Exception:
            pass

    # Seed with the session's first model so opening human turns aren't "unknown".
    cur_model = next(
        (m for r in rows if r.get("type") == "assistant"
         for m in [(r.get("message") or {}).get("model")]
         if m and m != "<synthetic>"),
        "unknown",
    )
    prior_human = []          # (wordset, week, model) seen earlier this session
    sid = os.path.basename(path)

    for d in rows:
        ts = d.get("timestamp")
        if not ts:
            continue
        wk = week_of(ts)
        typ = d.get("type")
        msg = d.get("message") or {}

        if typ == "assistant":
            m = msg.get("model")
            if m and m != "<synthetic>":
                cur_model = m
            if m == "<synthetic>" or not m:
                continue
            bump(wk, cur_model, "sessions", sid=sid)
            bump(wk, cur_model, "asst_msgs")
            t = text_of(msg)
            if t.strip() and ADMIT.search(t):
                bump(wk, cur_model, "asst_admit")
                snip = ADMIT.search(t)
                examples["admit"].append((ts, cur_model, t[max(0, snip.start() - 60):snip.end() + 90].replace("\n", " ")))

        elif typ == "user":
            if d.get("isSidechain") or d.get("isMeta") or d.get("isCompactSummary"):
                continue
            c = msg.get("content")

            # tool results ride in on user records
            if isinstance(c, list) and any(
                isinstance(b, dict) and b.get("type") == "tool_result" for b in c
            ):
                for b in c:
                    if not (isinstance(b, dict) and b.get("type") == "tool_result"):
                        continue
                    bump(wk, cur_model, "tool_calls")
                    if b.get("is_error"):
                        body = b.get("content")
                        body = body if isinstance(body, str) else json.dumps(body)
                        if REJECT_MARK in body.lower():
                            bump(wk, cur_model, "tool_reject")
                        else:
                            bump(wk, cur_model, "tool_err")
                continue

            t = text_of(msg).strip()
            if not t:
                continue
            if t.startswith("[Request interrupted"):
                bump(wk, cur_model, "interrupts")
                continue
            if SYNTHETIC_PREFIX.match(t) or INJECTED.match(t):
                continue

            # a genuine human turn
            bump(wk, cur_model, "human_turns")
            if PROFANITY.search(t):
                bump(wk, cur_model, "human_profane")

            # Pasted logs are real turns but are not the user *talking*, so they
            # are excluded from language-based and repetition metrics.
            if is_paste(t):
                continue
            bump(wk, cur_model, "prose_turns")

            if CORRECT.search(t):
                bump(wk, cur_model, "human_correct")
                examples["correct"].append((ts, cur_model, t[:220].replace("\n", " ")))
            elif DIRECTIVE.search(t):
                bump(wk, cur_model, "human_directive")

            if RESTATE_MARK.search(t):
                bump(wk, cur_model, "human_restate_mark")
                examples["restate_mark"].append((ts, cur_model, t[:220].replace("\n", " ")))

            # Lexical near-duplicate of something said in the last 8 prose turns.
            ws = content_words(t)
            if len(ws) >= 5:
                if any(jaccard(ws, prev) >= 0.25 for prev in prior_human[-8:]):
                    bump(wk, cur_model, "human_repeat")
                    examples["repeat"].append((ts, cur_model, t[:220].replace("\n", " ")))
                prior_human.append(ws)

# ---------------------------------------------------------------- reporting

def row(label, b):
    return {
        "label": label,
        "sessions": len(b.sessions),
        "assistant_msgs": b.asst_msgs,
        "human_turns": b.human_turns,
        "prose_turns": b.prose_turns,
        "tool_calls": b.tool_calls,
        "directive_n": b.human_directive,
        "admit_rate": wilson(b.asst_admit, b.asst_msgs),
        "admit_n": b.asst_admit,
        "tool_err_rate": wilson(b.tool_err, b.tool_calls),
        "tool_err_n": b.tool_err,
        "reject_rate": wilson(b.tool_reject, b.tool_calls),
        "reject_n": b.tool_reject,
        "correct_rate": wilson(b.human_correct, b.prose_turns),
        "correct_n": b.human_correct,
        "repeat_rate": wilson(b.human_repeat, b.prose_turns),
        "repeat_n": b.human_repeat,
        "restate_mark_rate": wilson(b.human_restate_mark, b.prose_turns),
        "restate_mark_n": b.human_restate_mark,
        "profane_n": b.human_profane,
        "interrupt_per_100_asst": (100 * b.interrupts / b.asst_msgs) if b.asst_msgs else 0,
        "interrupt_n": b.interrupts,
    }


def fmt(r):
    def pc(t, n):
        return f"{t[0]:5.1f}% [{t[1]:4.1f}–{t[2]:4.1f}]  n={n:<4}"
    return (f"{r['label']:<14} sess={r['sessions']:<3} asst={r['assistant_msgs']:<5} "
            f"human={r['human_turns']:<4} prose={r['prose_turns']:<4} tools={r['tool_calls']:<5}\n"
            f"    admits      {pc(r['admit_rate'], r['admit_n'])} of assistant msgs\n"
            f"    tool errors {pc(r['tool_err_rate'], r['tool_err_n'])} of tool calls\n"
            f"    rejected    {pc(r['reject_rate'], r['reject_n'])} of tool calls\n"
            f"    pushback    {pc(r['correct_rate'], r['correct_n'])} of prose turns\n"
            f"    near-dup    {pc(r['repeat_rate'], r['repeat_n'])} of prose turns\n"
            f"    'i said..' {pc(r['restate_mark_rate'], r['restate_mark_n'])} of prose turns\n"
            f"    interrupts  {r['interrupt_n']:<4} ({r['interrupt_per_100_asst']:.2f}/100 asst msgs)"
            f"   profanity n={r['profane_n']}")


weeks = [row(w, by_week[w]) for w in sorted(by_week)]
models = [row(m, by_model[m]) for m in
          sorted(by_model, key=lambda k: -by_model[k].asst_msgs)]

print("=" * 78)
print("BY WEEK")
print("=" * 78)
for r in weeks:
    print(fmt(r)); print()

print("=" * 78)
print("BY MODEL")
print("=" * 78)
for r in models:
    print(fmt(r)); print()

def ztest(k1, n1, k2, n2):
    """Two-proportion z-test; returns (z, two-sided p)."""
    if not n1 or not n2:
        return (0.0, 1.0)
    p1, p2 = k1 / n1, k2 / n2
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return (0.0, 1.0)
    z = (p2 - p1) / se
    pval = math.erfc(abs(z) / math.sqrt(2))
    return (z, pval)


print("=" * 78)
print("SIGNIFICANCE  (claude-opus-4-8  ->  claude-opus-5)")
print("=" * 78)
a, b = by_model["claude-opus-4-8"], by_model["claude-opus-5"]
tests = [
    ("self-admitted mistakes / asst msg", a.asst_admit, a.asst_msgs, b.asst_admit, b.asst_msgs),
    ("tool errors / tool call",           a.tool_err,   a.tool_calls, b.tool_err,  b.tool_calls),
    ("rejected tool calls / tool call",   a.tool_reject, a.tool_calls, b.tool_reject, b.tool_calls),
    ("user pushback / prose turn",        a.human_correct, a.prose_turns, b.human_correct, b.prose_turns),
    ("user near-dup / prose turn",        a.human_repeat, a.prose_turns, b.human_repeat, b.prose_turns),
    ("'i already said' / prose turn",     a.human_restate_mark, a.prose_turns, b.human_restate_mark, b.prose_turns),
]
for name, k1, n1, k2, n2 in tests:
    z, p = ztest(k1, n1, k2, n2)
    r1 = 100 * k1 / n1 if n1 else 0
    r2 = 100 * k2 / n2 if n2 else 0
    sig = "SIGNIFICANT" if p < 0.05 else "not significant"
    print(f"  {name:<36} {r1:5.2f}% -> {r2:5.2f}%   p={p:.4f}  {sig}")
print()

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "metrics.json")
with open(out, "w") as fh:
    json.dump({"weeks": weeks, "models": models,
               "examples": {k: v[:40] for k, v in examples.items()}}, fh, indent=2)
print(f"wrote {out}")
