#!/usr/bin/env python3
"""Idempotent snapshot: archive raw transcripts, then append weekly metrics.

Claude Code deletes transcripts after cleanupPeriodDays (default 30), so the
raw evidence is on a rolling window. This archives it before it rotates, then
folds each week into a durable metrics_history.jsonl.

Safe to run as often as you like -- archiving skips unchanged files and the
history upserts by (week, model).

  python3 snapshot.py            # archive + recompute + show trend
  python3 snapshot.py --trend    # just show the trend, no work
"""
import json, os, sys, gzip, shutil, hashlib, subprocess, collections, datetime

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.expanduser("~/.claude/projects")
ARCHIVE = os.path.join(HERE, "archive")
HISTORY = os.path.join(HERE, "metrics_history.jsonl")
MANIFEST = os.path.join(ARCHIVE, "manifest.json")

# Metrics we track over time. Keep this list append-only so old rows stay
# comparable to new ones.
def aggregate(episodes):
    buckets = collections.defaultdict(list)
    for e in episodes:
        buckets[(e["week"], e["model"])].append(e)
    rows = []
    for (week, model), E in sorted(buckets.items()):
        n = len(E)
        med = lambda f: sorted(f(x) for x in E)[n // 2]
        rows.append({
            "week": week,
            "model": model,
            "episodes": n,
            "sessions_first_turns": sum(1 for e in E if e["is_first"]),
            "conceded_pct": round(100 * sum(1 for e in E if e["admitted"]) / n, 2),
            "credited_pct": round(100 * sum(1 for e in E if e.get("credited")) / n, 2),
            "pushback_pct": round(100 * sum(1 for e in E if e["pushed"]) / n, 2),
            "out_tok_median": med(lambda e: e["out_tok"]),
            "out_tok_total": sum(e["out_tok"] for e in E),
            "out_tok_in_concessions": sum(e["out_tok"] for e in E if e["admitted"]),
            "human_chars_median": med(lambda e: e["human_chars"]),
            "human_chars_total": sum(e["human_chars"] for e in E),
            "tools_median": med(lambda e: e["tools"]),
            "tools_total": sum(e["tools"] for e in E),
            "reversals_total": sum(e["reversals"] for e in E),
            "edit_fail_total": sum(e["edit_fail"] for e in E),
            "dup_cmd_total": sum(e["dup_cmd"] for e in E),
            "reread_total": sum(e["reread"] for e in E),
        })
    return rows


def archive_raw():
    """Copy any new/changed transcript into archive/ as .gz. Never deletes."""
    os.makedirs(ARCHIVE, exist_ok=True)
    man = json.load(open(MANIFEST)) if os.path.exists(MANIFEST) else {}
    new = changed = 0
    for root, _, files in os.walk(SRC):
        for fn in files:
            if not fn.endswith(".jsonl"):
                continue
            path = os.path.join(root, fn)
            rel = os.path.relpath(path, SRC)
            st = os.stat(path)
            sig = f"{st.st_size}:{int(st.st_mtime)}"
            if man.get(rel) == sig:
                continue
            dest = os.path.join(ARCHIVE, rel + ".gz")
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with open(path, "rb") as fi, gzip.open(dest, "wb") as fo:
                shutil.copyfileobj(fi, fo)
            if rel in man:
                changed += 1
            else:
                new += 1
            man[rel] = sig
    json.dump(man, open(MANIFEST, "w"), indent=0)
    total = sum(os.path.getsize(os.path.join(r, f))
                for r, _, fs in os.walk(ARCHIVE) for f in fs)
    print(f"archive: {new} new, {changed} updated, {len(man)} sessions held, "
          f"{total/1e6:.1f} MB on disk")
    return len(man)


def upsert(rows):
    """Replace rows for any (week, model) we just recomputed; keep the rest."""
    old = []
    if os.path.exists(HISTORY):
        for line in open(HISTORY):
            line = line.strip()
            if line:
                old.append(json.loads(line))
    fresh = {(r["week"], r["model"]) for r in rows}
    kept = [r for r in old if (r["week"], r["model"]) not in fresh]
    stamp = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    for r in rows:
        r["snapshot_at"] = stamp
    allrows = sorted(kept + rows, key=lambda r: (r["week"], r["model"]))
    with open(HISTORY, "w") as fh:
        for r in allrows:
            fh.write(json.dumps(r) + "\n")
    print(f"history: {len(rows)} week/model rows refreshed, "
          f"{len(kept)} preserved from archived-away data, {len(allrows)} total")
    return allrows


def trend(rows, weeks=10):
    """Weighted roll-up per week across models, plus the per-model split."""
    byweek = collections.defaultdict(list)
    for r in rows:
        byweek[r["week"]].append(r)
    print()
    print(f"{'week':<12}{'eps':>5}{'conceded%':>11}{'outTok med':>12}"
          f"{'concession$':>13}{'yourChars':>11}  models")
    print("-" * 84)
    for w in sorted(byweek)[-weeks:]:
        R = byweek[w]
        n = sum(r["episodes"] for r in R) or 1
        conceded = sum(r["conceded_pct"] * r["episodes"] for r in R) / n
        tot = sum(r["out_tok_total"] for r in R) or 1
        burn = 100 * sum(r["out_tok_in_concessions"] for r in R) / tot
        otm = sum(r["out_tok_median"] * r["episodes"] for r in R) / n
        hc = sum(r["human_chars_median"] * r["episodes"] for r in R) / n
        tags = ",".join(sorted({r["model"].replace("claude-", "") for r in R}))
        print(f"{w:<12}{n:>5}{conceded:>10.1f}%{otm:>12.0f}"
              f"{burn:>12.1f}%{hc:>11.0f}  {tags}")
    print("\nconceded% = share of your requests that ended with Claude admitting "
          "it was wrong\nconcession$ = share of output tokens burned in those requests")


def main():
    if "--trend" in sys.argv:
        rows = [json.loads(l) for l in open(HISTORY) if l.strip()]
        trend(rows)
        return
    archive_raw()
    r = subprocess.run([sys.executable, os.path.join(HERE, "analyze3.py")],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print("analyze3 failed:\n" + r.stderr[-2000:])
        sys.exit(1)
    episodes = json.load(open(os.path.join(HERE, "episodes.json")))
    rows = upsert(aggregate(episodes))
    trend(rows)
    # Rebuild index.html from the history we just wrote. The report used to be
    # hand-maintained, which meant it drifted silently every time this ran.
    r = subprocess.run([sys.executable, os.path.join(HERE, "report.py")],
                       capture_output=True, text=True)
    print("\n" + (r.stdout.strip() if r.returncode == 0
                  else "report.py failed:\n" + r.stderr[-2000:]))


if __name__ == "__main__":
    main()
