#!/usr/bin/env python3
"""Regenerate index.html from the measured data.

The report used to be hand-written with its numbers typed into the markup,
which meant every snapshot silently widened the gap between what the page
claimed and what the data said. Nothing here is hand-entered.

  python3 report.py                       # rebuild index.html
  python3 report.py --baseline claude-opus-4-8 --current claude-opus-5
  python3 report.py --check               # exit 1 if index.html is stale

Data sources, in order of durability:

  metrics_history.jsonl  committed, counts only, keeps weeks whose raw
                         transcripts have since rotated out. Drives the
                         weekly chart and every rate.
  episodes.json          rebuilt by snapshot.py from the rolling window in
                         ~/.claude/projects. Drives medians and the
                         distribution tests, which need per-episode values.
  labels.json + sample_key.json
                         the blind hand-labelling. The key is gitignored, so
                         the joined *counts* are cached to blind_result.json
                         (no message content) and reused when it is absent.

Verdicts are derived from the p-values, not asserted. If an effect stops
being significant the page says so on the next run.
"""
import json, os, sys, math, html, argparse, datetime, statistics, collections

HERE = os.path.dirname(os.path.abspath(__file__))
HISTORY = os.path.join(HERE, "metrics_history.jsonl")
EPISODES = os.path.join(HERE, "episodes.json")
LABELS = os.path.join(HERE, "labels.json")
KEY = os.path.join(HERE, "sample_key.json")
BLIND = os.path.join(HERE, "blind_result.json")
OUT = os.path.join(HERE, "index.html")

SMALL_N = 20          # weeks below this are drawn faded; too small to read


# ---------------------------------------------------------------- statistics

def _lhyp(a, b, c, d):
    lg = math.lgamma
    n = a + b + c + d
    return (lg(a + b + 1) + lg(c + d + 1) + lg(a + c + 1) + lg(b + d + 1)
            - lg(n + 1) - lg(a + 1) - lg(b + 1) - lg(c + 1) - lg(d + 1))


def fisher(a, b, c, d):
    """Two-sided Fisher exact on a 2x2. Used instead of the z-test because on
    counts this small the normal approximation lies -- it called a 0/292 vs
    2/149 split significant at p=0.047 where Fisher gives p=0.11."""
    if min(a + b, c + d, a + c, b + d) < 0:
        return 1.0
    n, r1, c1 = a + b + c + d, a + b, a + c
    if n == 0 or r1 in (0, n) or c1 in (0, n):
        return 1.0
    p0, tot = _lhyp(a, b, c, d), 0.0
    for x in range(max(0, c1 - (n - r1)), min(r1, c1) + 1):
        p = _lhyp(x, r1 - x, c1 - x, n - r1 - c1 + x)
        if p <= p0 + 1e-9:
            tot += math.exp(p)
    return min(tot, 1.0)


def mwu(a, b):
    """Mann-Whitney U, normal approximation with tie correction. Two-sided.
    Medians of token/tool counts are heavily skewed; a t-test would not do."""
    if not a or not b:
        return 1.0
    comb = sorted([(v, 0) for v in a] + [(v, 1) for v in b])
    rs, i, ties = [0.0] * len(comb), 0, 0.0
    while i < len(comb):
        j = i
        while j + 1 < len(comb) and comb[j + 1][0] == comb[i][0]:
            j += 1
        r, t = (i + j) / 2 + 1, j - i + 1
        for k in range(i, j + 1):
            rs[k] = r
        ties += t ** 3 - t
        i = j + 1
    n1, n2 = len(a), len(b)
    r1 = sum(rs[k] for k in range(len(comb)) if comb[k][1] == 0)
    u1 = r1 - n1 * (n1 + 1) / 2
    n = n1 + n2
    mu = n1 * n2 / 2
    var = n1 * n2 * (n ** 3 - n - ties) / (12.0 * n * (n - 1)) if n > 1 else 0
    if var <= 0:
        return 1.0
    z = (abs(u1 - mu) - 0.5) / math.sqrt(var)
    return max(0.0, min(1.0, 2 * (1 - 0.5 * (1 + math.erf(z / math.sqrt(2))))))


def pfmt(p):
    if p < 0.0001:
        return "<0.0001"
    if p < 0.001:
        return f"{p:.5f}".rstrip("0")
    return f"{p:.3f}" if p < 0.1 else f"{p:.2f}"


# ---------------------------------------------------------------- data load

def load_history():
    if not os.path.exists(HISTORY):
        sys.exit("no metrics_history.jsonl -- run: python3 snapshot.py")
    return [json.loads(l) for l in open(HISTORY) if l.strip()]


def load_episodes():
    if not os.path.exists(EPISODES):
        return []
    try:
        return json.load(open(EPISODES))
    except Exception:
        return []


def counts(rows, model, field="conceded_pct"):
    """Recover (hits, total) from the weekly aggregates. Percentages are
    stored to 2dp against n<1000, so the integer is exactly recoverable."""
    hit = tot = 0
    for r in rows:
        if r["model"] != model or field not in r:
            continue
        tot += r["episodes"]
        hit += round(r["episodes"] * r[field] / 100.0)
    return hit, tot


def pick_models(rows, args):
    """Baseline and current default to the two highest-volume models, ordered
    by when they first appear -- which is what a regression comparison wants."""
    vol = collections.Counter()
    first = {}
    for r in rows:
        if r["model"] == "unknown":
            continue
        vol[r["model"]] += r["episodes"]
        first[r["model"]] = min(first.get(r["model"], r["week"]), r["week"])
    top = sorted((m for m, _ in vol.most_common(2)), key=lambda m: first[m])
    base = args.baseline or (top[0] if len(top) > 1 else None)
    cur = args.current or (top[1] if len(top) > 1 else None)
    if not base or not cur:
        sys.exit("need two models to compare; pass --baseline and --current")
    return base, cur


def blind_result(base, cur):
    """Join hand labels to the withheld key. Cache counts so the report still
    builds from a fresh clone, where sample_key.json is gitignored."""
    if os.path.exists(LABELS) and os.path.exists(KEY):
        labels, key = json.load(open(LABELS)), json.load(open(KEY))
        agg = collections.defaultdict(lambda: collections.Counter())
        for tid, lab in labels.items():
            m = (key.get(tid) or {}).get("model")
            if not m:
                continue
            a = agg[m]
            a["n"] += 1
            if lab:
                a["friction"] += 1
            if "B" in lab:
                a["restate"] += 1
        res = {"models": {m: dict(c) for m, c in agg.items()},
               "labelled": len(labels),
               "sampled_at": datetime.date.today().isoformat()}
        if os.path.exists(BLIND):                 # keep the original date
            try:
                res["sampled_at"] = json.load(open(BLIND)).get(
                    "sampled_at", res["sampled_at"])
            except Exception:
                pass
        json.dump(res, open(BLIND, "w"), indent=2)
    elif os.path.exists(BLIND):
        res = json.load(open(BLIND))
    else:
        return None
    m = res.get("models", {})
    if base not in m or cur not in m:
        return None
    return {"base": m[base], "cur": m[cur], "labelled": res.get("labelled", 0),
            "sampled_at": res.get("sampled_at", "")}


# ---------------------------------------------------------------- measures

def build_fx(rows, eps, base, cur, blind):
    """The effect-size table. Each row: label, baseline, current, x, p, sig."""
    fx = []

    def rate_row(label, field):
        a, na = counts(rows, base, field)
        c, nc = counts(rows, cur, field)
        if not na or not nc or not a:
            return
        r0, r1 = 100 * a / na, 100 * c / nc
        fx.append([label, f"{r0:.1f}%", f"{r1:.1f}%", r1 / r0,
                   fisher(a, na - a, c, nc - c)])

    if blind:
        b, c = blind["base"], blind["cur"]
        for label, k in (("Blind-labelled friction turns", "friction"),
                         ("Me restating myself", "restate")):
            nb, nc = b.get("n", 0), c.get("n", 0)
            hb, hc = b.get(k, 0), c.get(k, 0)
            if nb and nc and hb:
                fx.append([label, f"{100*hb/nb:.0f}%", f"{100*hc/nc:.0f}%",
                           (hc / nc) / (hb / nb),
                           fisher(hb, nb - hb, hc, nc - hc)])

    rate_row("Requests ending in a concession", "conceded_pct")
    rate_row("Admissions crediting me", "credited_pct")

    # Medians need per-episode values, so these come from episodes.json and
    # are silently dropped when the rolling window no longer covers a model.
    E = collections.defaultdict(list)
    for e in eps:
        E[e["model"]].append(e)
    for label, f, fmt in (("Tool calls per request", "tools", "{:,.0f}"),
                          ("Characters I type per request", "human_chars", "{:,.0f}"),
                          ("Output tokens per request", "out_tok", "{:,.0f}")):
        a = [x[f] for x in E.get(base, [])]
        b = [x[f] for x in E.get(cur, [])]
        if len(a) < 5 or len(b) < 5:
            continue
        m0, m1 = statistics.median(a), statistics.median(b)
        if not m0:
            continue
        fx.append([label, fmt.format(m0), fmt.format(m1), m1 / m0, mwu(a, b)])

    fx.sort(key=lambda r: -r[3])
    return [r + [r[4] < 0.05] for r in fx]


def build_weeks(rows):
    """One bar per week, coloured by the dominant model of that week."""
    by = collections.defaultdict(list)
    for r in rows:
        by[r["week"]].append(r)
    out = []
    for wk in sorted(by):
        R = by[wk]
        n = sum(r["episodes"] for r in R)
        if not n:
            continue
        conceded = 100 * sum(round(r["episodes"] * r["conceded_pct"] / 100.0)
                             for r in R) / n
        tot = sum(r["out_tok_total"] for r in R)
        burn = 100 * sum(r["out_tok_in_concessions"] for r in R) / tot if tot else 0.0
        top = max(R, key=lambda r: r["episodes"])
        mixed = top["episodes"] < 0.8 * n
        name = "mixed" if mixed else top["model"].replace("claude-", "")
        out.append([wk, n, round(conceded, 1), round(burn, 1), name,
                    "mx" if mixed else MODEL_COLOR.get(name, "mx")])
    return out


MODEL_COLOR = {"sonnet-4-6": "s3", "opus-4-8": "s1", "opus-5": "s2",
               "sonnet-5": "s4", "fable-5": "s4"}


def totals(rows, models):
    n = sum(r["episodes"] for r in rows)
    weeks = sorted({r["week"] for r in rows})
    return {"episodes": n, "first": weeks[0] if weeks else "",
            "last": weeks[-1] if weeks else "", "models": models}


# ---------------------------------------------------------------- rendering

def esc(s):
    return html.escape(str(s), quote=False)


def verdicts(fx, blind, base, cur, rows):
    """Chips are derived, not asserted -- a dead effect reports as dead."""
    d = {r[0]: r for r in fx}
    V = []

    def chip(row):
        if row is None:
            return "unk", "Unmeasured"
        return ("sup", "Supported") if row[5] else ("nul", "Not supported")

    conc = d.get("Requests ending in a concession")
    bl = d.get("Blind-labelled friction turns")
    if conc:
        cls, lab = chip(conc)
        agree = bl and bl[5] and conc[5]
        body = (f"<p>Automated admission-counting gives <strong>&times;{conc[3]:.2f}</strong> "
                f"(p = {pfmt(conc[4])}), taking the share of requests that end with Claude "
                f"conceding from <strong>{conc[1]}</strong> to <strong>{conc[2]}</strong>.")
        if bl:
            body += (f" Hand-labelling {blind['labelled']} randomly sampled exchanges &mdash; with the "
                     f"model identity withheld from the labeller until every label was fixed &mdash; "
                     f"gives <strong>&times;{bl[3]:.2f}</strong> (p = {pfmt(bl[4])}).")
            body += (" Two methods that share no assumptions agree.</p>" if agree else
                     " The two methods do not agree, which is the thing to resolve before citing either.</p>")
        else:
            body += " The blind labelling has not been re-run against this pair.</p>"
        V.append((cls, "Mistakes I have to catch roughly doubled"
                  if conc[3] >= 1.8 else
                  f"Requests ending in a concession moved &times;{conc[3]:.2f}", body))

    tok = d.get("Output tokens per request")
    tools = d.get("Tool calls per request")
    chars = d.get("Characters I type per request")
    if tok or tools or chars:
        parts = []
        for r, name in ((tok, "median output tokens per request"),
                        (tools, "tool calls per request")):
            if r:
                parts.append(f"{name} {'rose' if r[3] >= 1 else 'fell'} "
                             f"<strong>{abs(r[3]-1)*100:.0f}%</strong> (p = {pfmt(r[4])})")
        b0, b1 = burn_pair(rows, base, cur)
        body = "<p>" + (" and ".join(parts).capitalize() + ". " if parts else "")
        body += (f"The share of all generated tokens spent inside requests that end in a "
                 f"concession went from <strong>{b0:.1f}% to {b1:.1f}%</strong>.</p>")
        if chars:
            body += (f"<p>Meanwhile the characters <em>I</em> type per request "
                     f"{'rose' if chars[3] >= 1 else 'fell'} <strong>{abs(chars[3]-1)*100:.0f}%</strong> "
                     f"(p = {pfmt(chars[4])}). Front-loading more context is not preventing the "
                     f"friction; both moved together.</p>")
        sig = [r for r in (tok, tools, chars) if r and r[5]]
        V.append(("sup" if sig else "nul",
                  "It costs more, and I supply more of the context", body))

    rest = d.get("Me restating myself")
    if rest:
        cls, _ = chip(rest)
        V.append((cls, "I am not repeating myself more" if not rest[5]
                  else "I am repeating myself more",
                  f"<p>Hand-labelled restatement went from <strong>{rest[1]} to {rest[2]}</strong> "
                  f"of requests &mdash; <strong>p = {pfmt(rest[4])}</strong>.</p>"
                  "<p>One real blind spot: the labeller saw only two prior turns within a single "
                  "session, so re-teaching the same lesson in a <em>new</em> session is invisible "
                  "to this method. That is a different measurement, and it has not been built yet.</p>"))

    V.append(("unk", "Reasoning quality cannot be assessed at all",
              "<p>Thinking content is stored as an empty string plus an encrypted signature for "
              "every model except sonnet-4-6. There is no reasoning text to measure. Any claim that "
              "&ldquo;thinking got shorter&rdquo; would be an artifact of a storage policy change, "
              "not a finding.</p>" + churn_para(rows, base, cur)))
    return V


def burn_pair(rows, base, cur):
    def burn(m):
        R = [r for r in rows if r["model"] == m]
        t = sum(r["out_tok_total"] for r in R)
        return 100 * sum(r["out_tok_in_concessions"] for r in R) / t if t else 0.0
    return burn(base), burn(cur)


def churn_para(rows, base, cur):
    """Behavioural proxies for flailing. Counts per 100 requests, Fisher on
    the raw totals -- these are the measures that came back null."""
    out = []
    for field, name in (("reversals_total", "self-reversals"),
                        ("edit_fail_total", "failed edits"),
                        ("dup_cmd_total", "duplicate commands"),
                        ("reread_total", "repeated file reads")):
        def agg(m):
            R = [r for r in rows if r["model"] == m]
            return sum(r.get(field, 0) for r in R), sum(r["episodes"] for r in R)
        a, na = agg(base)
        c, nc = agg(cur)
        if not na or not nc:
            continue
        p = fisher(a, max(na - a, 0), c, max(nc - c, 0))
        r0, r1 = 100 * a / na, 100 * c / nc
        out.append(f"{name} {r0:.0f}&rarr;{r1:.0f} per 100 requests, p = {pfmt(p)}")
    if not out:
        return ""
    return ("<p>Behavioural proxies for flailing: " + "; ".join(out) +
            ". More tokens and more tool calls per request are equally consistent with "
            "thoroughness and with churn; nothing here separates them.</p>")


def render(rows, fx, weeks, V, tot, base, cur, blind, stale_eps):
    css = open(os.path.join(HERE, "report.css")).read()
    js = open(os.path.join(HERE, "report.js")).read()
    bshort, cshort = base.replace("claude-", ""), cur.replace("claude-", "")

    hero = []
    for label, key, sub in (
            ("Requests ending in a concession", "Requests ending in a concession", None),
            ("Blind-labelled friction", "Blind-labelled friction turns", None),
    ):
        r = next((x for x in fx if x[0] == key), None)
        if r:
            hero.append((label, f"{r[3]:.2f}&times;",
                         f"{r[1]} &rarr; {r[2]} per request<br>Fisher exact p = {pfmt(r[4])}"))
    b0, b1 = burn_pair(rows, base, cur)
    hero.append(("Output tokens spent being wrong", f"{b1:.1f}%",
                 f"up from {b0:.1f}%<br>share of all generated tokens"
                 if b1 >= b0 else f"down from {b0:.1f}%<br>share of all generated tokens"))

    stats = "\n".join(
        f'    <div class="stat"><p class="k">{esc(k)}</p><p class="v">{v}</p>'
        f'<p class="sub">{s}</p></div>' for k, v, s in hero)

    vhtml = "\n".join(
        f'    <div class="verdict"><div class="vhead">'
        f'<span class="chip {c}">{esc(lab_of(c))}</span><h3>{t}</h3></div>{b}</div>'
        for c, t, b in V)

    sig_n = sum(1 for r in fx if r[5])
    legend_models = sorted({w[4] for w in weeks if w[4] != "mixed"})
    legend = "\n".join(
        f'        <span><i class="swatch" style="background:var(--{MODEL_COLOR.get(m, "rule-2")})"></i>{esc(m)}</span>'
        for m in legend_models)

    note = ""
    if stale_eps:
        note = ('<p class="figsub">Medians are computed from the transcripts still on disk; '
                'weeks that have rotated out contribute rates but not distributions.</p>')

    return TEMPLATE.format(
        css=css, js=js, stats=stats, verdicts=vhtml, legend=legend, note=note,
        fx=json.dumps([r[:5] + [bool(r[5])] for r in fx]),
        wk=json.dumps(weeks),
        base=esc(bshort), cur=esc(cshort),
        episodes=f"{tot['episodes']:,}", first=esc(tot["first"]), last=esc(tot["last"]),
        models=esc(", ".join(m.replace("claude-", "") for m in tot["models"])),
        sig_n=sig_n, fx_n=len(fx),
        blind_n=blind["labelled"] if blind else 0,
        small_n=SMALL_N,
        generated=datetime.date.today().isoformat(),
    )


def lab_of(c):
    return {"sup": "Supported", "nul": "Not supported", "unk": "Unmeasurable"}[c]


TEMPLATE = """<title>Is Claude Getting Worse? A Measurement</title>

<style>
{css}
</style>

<div class="rp">
<div class="wrap">

  <header class="head">
    <p class="eyebrow">Empirical audit &middot; Claude Code transcripts</p>
    <h1>Is Claude getting worse?</h1>
    <p class="standfirst">
      I measured it against my own logs instead of arguing about it. Every
      figure on this page is regenerated from the transcripts; none is typed in
      by hand.
    </p>
    <p class="byline">
      <span>{episodes} requests</span><span>{first} &rarr; {last}</span>
      <span>{base} vs {cur}</span><span>generated {generated}</span>
    </p>
  </header>

  <div class="stats">
{stats}
  </div>

  <section>
    <h2>What this measures <span class="n">&sect;1</span></h2>
    <p>
      Claude Code writes every session to disk as JSONL, timestamped, with the
      model recorded per message. That makes a personal, longitudinal dataset:
      not benchmark scores, but what actually happened across months of real
      work &mdash; homelab automation, PCB design, firmware bring-up, media
      infrastructure.
    </p>
    <p>
      The comparison is <strong>{base}</strong> against <strong>{cur}</strong>.
      Everything is normalised <strong>per request</strong> &mdash; one thing I
      asked for &mdash; rather than per message, because a chattier model emits
      more messages for the same work. Dividing by messages hides the effect
      behind the model's own verbosity.
    </p>
    <p>
      The core proxy for &ldquo;it made a mistake&rdquo; is <strong>an admission
      that credits the human</strong> &mdash; Claude saying &ldquo;you're
      right&rdquo; or &ldquo;good catch&rdquo;. It cannot say that unless it was
      just contradicted, so admissions track caught errors rather than
      spontaneous self-doubt.
    </p>
  </section>

  <section>
    <h2>Effect sizes <span class="n">&sect;2</span></h2>
    <figure>
      <p class="figtitle">Change from {base} to {cur}, per request</p>
      <p class="figsub">multiplier &middot; baseline 1.0 = no change</p>
      <div id="fx"></div>
      <div class="legend">
        <span><i class="swatch" style="background:var(--s1)"></i>statistically significant (p &lt; 0.05)</span>
        <span><i class="swatch" style="background:var(--rule-2)"></i>not significant</span>
      </div>
      <details>
        <summary>Show as table</summary>
        <div class="scroll">
          <table>
            <thead><tr><th>Measure</th><th class="num">{base}</th><th class="num">{cur}</th><th class="num">&times;</th><th class="num">p</th></tr></thead>
            <tbody id="fxtable"></tbody>
          </table>
        </div>
      </details>
      <figcaption>
        {sig_n} of {fx_n} measures moved significantly. Direction is what
        matters here: read the colour, then the p-value, then the sample size.
      </figcaption>
    </figure>
  </section>

  <section>
    <h2>Over time <span class="n">&sect;3</span></h2>
    <figure>
      <p class="figtitle">Share of requests that ended with Claude admitting it was wrong</p>
      <p class="figsub">by week &middot; bar colour = dominant model that week</p>
      {note}
      <div id="wk"></div>
      <div class="legend">
{legend}
        <span><i class="swatch" style="background:var(--rule-2)"></i>mixed / low sample</span>
      </div>
      <details>
        <summary>Show as table</summary>
        <div class="scroll">
          <table>
            <thead><tr><th>Week</th><th class="num">Requests</th><th class="num">Conceded</th><th class="num">Token burn</th><th>Model</th></tr></thead>
            <tbody id="wktable"></tbody>
          </table>
        </div>
      </details>
      <figcaption>
        Faded bars are weeks with fewer than {small_n} requests &mdash; too small to
        read anything into. Model regimes changed over this window, so this is
        not a single continuous trend; it is several regimes side by side.
      </figcaption>
    </figure>
  </section>

  <section>
    <h2>Verdicts <span class="n">&sect;4</span></h2>
{verdicts}
  </section>

  <section>
    <h2>Method <span class="n">&sect;5</span></h2>
    <p>
      Everything is computed from local transcripts by scripts that re-read them
      live. Three guards matter:
    </p>
    <ul>
      <li>
        <strong>Blind labelling.</strong> The sample file carries no model name
        and no date. Labels were committed before the key was opened. Grading
        with the answer visible would invalidate the entire result.
      </li>
      <li>
        <strong>Fisher exact, not the z-test.</strong> On small counts the normal
        approximation lies. It called a 0/292 vs 2/149 split significant at
        p = 0.047; Fisher gave p = 0.11. Two headline numbers were nearly
        published wrong this way.
      </li>
      <li>
        <strong>Per-request denominators.</strong> Rates per assistant message
        flatter a verbose model. The unit is the thing a human asked for.
      </li>
    </ul>

    <div class="caveat">
      <p class="lab">Limitations &mdash; read before citing this</p>
      <p>
        <strong>Model and date are the same variable.</strong> Model assignment
        was never randomised, so anything that changed over calendar time is
        indistinguishable from anything that changed with the model.
      </p>
      <p>
        <strong>Task mix shifted.</strong> The {cur} sessions skew toward harder
        debugging, where errors are likelier regardless of which model is
        running.
      </p>
      <p>
        <strong>The labeller was Claude.</strong> Blinding removes the model
        identity, but writing style may still leak the answer. An independent
        human labeller on the same sample would settle it.
      </p>
      <p>
        <strong>The headline metric measures admission, not error.</strong> A
        future model that concedes <em>less</em> readily without actually
        improving will look like progress on this chart. Periodic blind
        labelling &mdash; last run on {blind_n} exchanges &mdash; is the only
        thing that catches that.
      </p>
    </div>
  </section>

  <section>
    <h2>Run it on your own logs <span class="n">&sect;6</span></h2>
    <p>
      One machine's transcripts cannot separate &ldquo;this model got worse&rdquo;
      from &ldquo;this user's work got harder.&rdquo; More datasets can. The
      tooling is a few Python files with no dependencies beyond the standard
      library, and <strong>it never transmits anything</strong> &mdash; it reads
      <code>~/.claude/projects</code> and writes locally.
    </p>
    <p>
      <code>python3 snapshot.py</code> archives your transcripts, appends weekly
      per-model metrics, and rebuilds this page. It takes about two seconds and
      is safe to re-run. What is worth sharing back is
      <code>metrics_history.jsonl</code>: counts only, no message content.
    </p>
    <div class="caveat">
      <p class="lab">Before you share anything</p>
      <p>
        Raw transcripts contain plaintext secrets. Mine held a wifi password,
        personal email addresses, and vault variable names &mdash; none of which
        I put there deliberately. The archive directory is gitignored for that
        reason. Never push it, and check what you are sending before you send it.
      </p>
    </div>
    <p>
      One setting is worth changing immediately whether or not you run any of
      this: <code>cleanupPeriodDays</code> defaults to <strong>30</strong>, so
      your transcripts are on a rolling window that quietly deletes its own
      baseline. Mine was about three weeks from being unrecoverable when I
      checked.
    </p>
  </section>

  <p class="foot">
    Generated {generated} from {episodes} requests &middot; {first} to {last}
    &middot; models: {models} &middot; rebuild with
    <code>python3 report.py</code>
  </p>

</div>
</div>

<div class="tip" id="tip" role="status" aria-live="polite"></div>

<script>
var FX = {fx};
var WK = {wk};
var SMALL_N = {small_n};
{js}
</script>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline")
    ap.add_argument("--current")
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if index.html differs from what the data says")
    args = ap.parse_args()

    rows = load_history()
    eps = load_episodes()
    base, cur = pick_models(rows, args)
    blind = blind_result(base, cur)
    fx = build_fx(rows, eps, base, cur, blind)
    weeks = build_weeks(rows)
    V = verdicts(fx, blind, base, cur, rows)

    models = sorted({r["model"] for r in rows if r["model"] != "unknown"})
    # True when history covers weeks the rolling transcript window no longer
    # does -- those weeks contribute rates but cannot contribute medians.
    stale = bool({r["week"] for r in rows} - {e["week"] for e in eps})
    page = render(rows, fx, weeks, V, totals(rows, models), base, cur, blind, stale)

    if args.check:
        old = open(OUT).read() if os.path.exists(OUT) else ""
        if old.strip() != page.strip():
            print("index.html is stale -- run: python3 report.py")
            sys.exit(1)
        print("index.html is current")
        return

    open(OUT, "w").write(page)
    print(f"wrote index.html  {base} vs {cur}  "
          f"{sum(1 for r in fx if r[5])}/{len(fx)} measures significant")
    for r in fx:
        print(f"  {'*' if r[5] else ' '} {r[0]:<34} {r[1]:>9} -> {r[2]:>9}"
              f"  x{r[3]:.2f}  p={pfmt(r[4])}")


if __name__ == "__main__":
    main()
