# Claude regression tracking

Measures whether Claude is getting worse, from local Claude Code transcripts.

## Why this exists at all

`cleanupPeriodDays` defaults to **30**, so `~/.claude/projects` is a rolling
30-day window. Without archiving, the baseline you compare against deletes
itself and the comparison can never be reproduced.

## Running it

```
python3 snapshot.py           # archive + recompute + print trend   (~2s)
python3 snapshot.py --trend   # just print the trend, no work
```

`snapshot.py` is idempotent. It archives only new/changed transcripts and
upserts history by (week, model), so rows for weeks whose raw data has since
been deleted are preserved.

## The two numbers that matter

| column | meaning |
|---|---|
| `conceded%` | share of your requests that ended with Claude admitting it was wrong |
| `concession$` | share of output tokens burned inside those requests |

Both roughly doubled from opus-4-8 to opus-5 (15.8% -> 32.9% per request,
Fisher p=0.000075; token burn 19.5% -> 30.9%).

## Files

| file | role |
|---|---|
| `snapshot.py` | the thing to run on a schedule |
| `analyze.py` | regex/detector definitions, message-level rates |
| `analyze2.py` | sequence-aware: admissions that credit you vs self-initiated |
| `analyze3.py` | per-episode token, context and churn metrics |
| `sample.py` | builds a **blind** labelling sample (model identity withheld) |
| `labels.json` | hand labels; `C`=corrected an error, `D`=did something not asked, `B`=restated |
| `metrics_history.jsonl` | durable record, safe to commit |
| `archive/` | gzipped raw transcripts — **contains secrets, never push** |

## Quarterly: re-run the blind labelling

Automated regexes undercount semantic mistakes. Every quarter, or whenever a
new model lands:

1. Edit `TARGETS` in `sample.py` to the two models you want to compare.
2. `python3 sample.py` -> writes `blind_sample.txt` and holds back the key.
3. Label every item in `blind_sample.txt` **before** opening `sample_key.json`.
4. Join and test.

Labelling before joining is the whole point — it is the only guard against
grading in favour of a conclusion you already hold.

## Known blind spots

- **Reasoning depth is unmeasurable.** Thinking text is stored as an empty
  string plus an encrypted signature for every model except sonnet-4-6. Any
  "thinking got shorter" claim would be a storage artifact.
- **Cross-session repetition is invisible.** Labelling sees two prior turns
  within one session, so re-teaching the same lesson in a *new* session does
  not register.
- **Model and date are confounded.** Model assignment was never randomised;
  opus-4-8 dominates July, opus-5 dominates August.
- **Task mix shifts.** Harder debugging sessions produce more errors
  regardless of model.
