#!/usr/bin/env python3
"""Build a BLIND labelling sample.

Writes two files:
  blind_sample.txt -- items with a random id, no model, no date. This is the
                      only file the labeller reads.
  sample_key.json  -- id -> model/date/session. Not opened until labels exist.

Unit: a human prose turn that is not the first of its session, shown with the
preceding human turns and a snippet of what Claude did in between.
"""
import json, glob, os, re, random, collections

ROOT = os.path.expanduser("~/.claude/projects")
here = os.path.dirname(os.path.abspath(__file__))
exec(open(os.path.join(here, "analyze.py")).read()
     .split("# ---------------------------------------------------------------- counters")[0])

PER_MODEL = 50
TARGETS = ["claude-opus-4-8", "claude-opus-5"]
random.seed(20260808)

items = []
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

    prior_h, asst_buf = [], []
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
            t = text_of(msg).strip()
            if t:
                asst_buf.append(t)
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
            if prior_h and cur_model in TARGETS:
                items.append({
                    "model": cur_model,
                    "date": ts[:10],
                    "session": os.path.basename(path)[:8],
                    "prior": [p[:150] for p in prior_h[-2:]],
                    "claude_did": (" ".join(asst_buf))[-260:] if asst_buf else "(no reply text)",
                    "turn": t[:380],
                })
            prior_h.append(t)
            asst_buf = []

by_model = collections.defaultdict(list)
for it in items:
    by_model[it["model"]].append(it)

sample = []
for m in TARGETS:
    pool = by_model[m]
    random.shuffle(pool)
    sample.extend(pool[:PER_MODEL])
random.shuffle(sample)

key = {}
lines = []
for i, it in enumerate(sample, 1):
    iid = f"T{i:03d}"
    key[iid] = {"model": it["model"], "date": it["date"], "session": it["session"]}
    lines.append(f"### {iid}")
    for p in it["prior"]:
        lines.append(f"[KOREY EARLIER] {p}")
    lines.append(f"[CLAUDE DID] {it['claude_did']}")
    lines.append(f"[KOREY NOW] {it['turn']}")
    lines.append("")

open(os.path.join(here, "blind_sample.txt"), "w").write("\n".join(lines))
json.dump(key, open(os.path.join(here, "sample_key.json"), "w"), indent=2)
print(f"pool sizes: " + ", ".join(f"{m}={len(by_model[m])}" for m in TARGETS))
print(f"wrote blind_sample.txt with {len(sample)} items, key held back")
