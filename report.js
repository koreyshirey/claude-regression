/* Chart rendering. FX, WK and SMALL_N are injected above by report.py --
   nothing in this file is edited to publish new numbers. */
(function () {
  var NS = "http://www.w3.org/2000/svg";
  var tip = document.getElementById("tip");

  function el(n, a) {
    var e = document.createElementNS(NS, n);
    for (var k in a) e.setAttribute(k, a[k]);
    return e;
  }
  function show(e, txt) {
    tip.textContent = txt;
    tip.style.opacity = "1";
    tip.style.left = e.clientX + "px";
    tip.style.top = e.clientY + "px";
  }
  function hide() { tip.style.opacity = "0"; }

  /* Smallest round step that covers `v` in `n` divisions. Rounding the whole
     span to a power of ten instead would leave half the plot empty -- a 25%
     max on a 0-50 axis reads as a much smaller number than it is. */
  function niceStep(v, n) {
    if (!(v > 0)) return 1;
    var raw = v / n;
    var p = Math.pow(10, Math.floor(Math.log(raw) / Math.LN10));
    var s = raw / p;
    return (s <= 1 ? 1 : s <= 2 ? 2 : s <= 2.5 ? 2.5 : s <= 5 ? 5 : 10) * p;
  }
  function ticks(step, n) {
    var out = [];
    for (var i = 0; i <= n; i++) out.push(step * i);
    return out;
  }

  /* ---------------- effect sizes ---------------- */
  (function drawFx() {
    if (!FX.length) return;
    var W = 720, rowH = 40, top = 34, H = top + FX.length * rowH + 26;
    var x0 = 258, x1 = 690, lo = 1, nTick = 4;
    var span = Math.max(0.4, Math.max.apply(null, FX.map(function (d) {
      return d[3];
    })) * 1.06 - lo);
    var step = niceStep(span, nTick), hi = lo + step * nTick;
    var sx = function (v) { return x0 + (v - lo) / (hi - lo) * (x1 - x0); };
    var svg = el("svg", { viewBox: "0 0 " + W + " " + H, role: "img",
      "aria-label": "Multiplier change across " + FX.length + " measures" });

    ticks(step, nTick).map(function (t) { return lo + t; }).forEach(function (g) {
      svg.appendChild(el("line", { x1: sx(g), x2: sx(g), y1: top - 16, y2: H - 26,
        stroke: g === 1 ? "var(--rule-2)" : "var(--rule)",
        "stroke-width": g === 1 ? 1.5 : 1 }));
      var t = el("text", { x: sx(g), y: H - 10, "text-anchor": "middle",
        fill: "var(--muted)", "font-size": "11", "font-family": "var(--f-mono)" });
      t.textContent = (step < 0.2 ? g.toFixed(2) : g.toFixed(1)) + "×";
      svg.appendChild(t);
    });

    FX.forEach(function (d, i) {
      var y = top + i * rowH, col = d[5] ? "var(--s1)" : "var(--rule-2)";
      var lab = el("text", { x: 0, y: y + 4, fill: "var(--ink)", "font-size": "13",
        "font-family": "var(--f-mono)" });
      lab.textContent = d[0];
      svg.appendChild(lab);

      /* a multiplier below 1 draws leftward from the baseline, and is clamped
         into view rather than silently running off the axis */
      var v = Math.max(lo, Math.min(hi, d[3]));
      svg.appendChild(el("line", { x1: sx(1), x2: sx(v), y1: y, y2: y,
        stroke: col, "stroke-width": 2, "stroke-linecap": "round" }));
      svg.appendChild(el("circle", { cx: sx(v), cy: y, r: 5.5, fill: col,
        stroke: "var(--surface)", "stroke-width": 2 }));

      var vt = el("text", { x: sx(v) + 13, y: y + 4, fill: "var(--ink-2)",
        "font-size": "12", "font-family": "var(--f-mono)" });
      vt.textContent = d[3].toFixed(2) + "×";
      svg.appendChild(vt);

      var hit = el("rect", { x: 0, y: y - rowH / 2, width: W, height: rowH,
        fill: "transparent", style: "cursor:crosshair" });
      hit.addEventListener("mousemove", function (e) {
        show(e, d[0] + "\n" + d[1] + "  →  " + d[2] + "   " + d[3].toFixed(2) +
          "×\np = " + d[4] + (d[5] ? "" : "  (not significant)"));
      });
      hit.addEventListener("mouseleave", hide);
      svg.appendChild(hit);
    });

    document.getElementById("fx").appendChild(svg);

    document.getElementById("fxtable").innerHTML = FX.map(function (d) {
      return "<tr><td>" + d[0] + "</td><td class='num'>" + d[1] + "</td><td class='num'>" +
        d[2] + "</td><td class='num'>" + d[3].toFixed(2) + "×</td><td class='num " +
        (d[5] ? "sig" : "ns") + "'>" + d[4] + "</td></tr>";
    }).join("");
  })();

  /* ---------------- weekly ---------------- */
  var COL = { s1: "var(--s1)", s2: "var(--s2)", s3: "var(--s3)",
              s4: "var(--s4)", mx: "var(--rule-2)" };

  (function drawWk() {
    if (!WK.length) return;
    var W = 720, H = 300, padL = 44, padR = 8, top = 16, base = 214;
    var n = WK.length, slot = (W - padL - padR) / n, bw = Math.min(42, slot - 16);
    var nTick = 3;
    var wstep = niceStep(Math.max.apply(null, WK.map(function (d) {
      return d[2];
    })) * 1.1, nTick);
    var max = wstep * nTick;
    var sy = function (v) { return base - (v / max) * (base - top); };
    var svg = el("svg", { viewBox: "0 0 " + W + " " + H, role: "img",
      "aria-label": "Weekly share of requests ending in a concession, by dominant model" });

    ticks(wstep, nTick).forEach(function (g) {
      svg.appendChild(el("line", { x1: padL, x2: W - padR, y1: sy(g), y2: sy(g),
        stroke: g === 0 ? "var(--rule-2)" : "var(--rule)" }));
      var t = el("text", { x: padL - 10, y: sy(g) + 4, "text-anchor": "end",
        fill: "var(--muted)", "font-size": "11", "font-family": "var(--f-mono)" });
      t.textContent = Math.round(g) + "%";
      svg.appendChild(t);
    });

    WK.forEach(function (d, i) {
      var cx = padL + slot * i + slot / 2, small = d[1] < SMALL_N;
      var h = Math.max(sy(0) - sy(d[2]), d[2] > 0 ? 3 : 0);
      if (h > 0) {
        var r = Math.min(4, h), x = cx - bw / 2, y = base - h;
        svg.appendChild(el("path", {
          d: "M" + x + " " + base + " L" + x + " " + (y + r) +
             " Q" + x + " " + y + " " + (x + r) + " " + y +
             " L" + (x + bw - r) + " " + y +
             " Q" + (x + bw) + " " + y + " " + (x + bw) + " " + (y + r) +
             " L" + (x + bw) + " " + base + " Z",
          fill: COL[d[5]] || COL.mx, opacity: small ? 0.32 : 1
        }));
      }
      var lv = el("text", { x: cx, y: base - h - 8, "text-anchor": "middle",
        fill: "var(--ink-2)", "font-size": "11.5", "font-family": "var(--f-mono)",
        opacity: small ? 0.5 : 1 });
      lv.textContent = d[2].toFixed(1);
      svg.appendChild(lv);

      var dt = el("text", { x: cx, y: base + 18, "text-anchor": "middle",
        fill: "var(--muted)", "font-size": "10", "font-family": "var(--f-mono)" });
      dt.textContent = d[0].slice(5);
      svg.appendChild(dt);

      var nn = el("text", { x: cx, y: base + 33, "text-anchor": "middle",
        fill: "var(--muted)", "font-size": "9.5", "font-family": "var(--f-mono)",
        opacity: .75 });
      nn.textContent = "n=" + d[1];
      svg.appendChild(nn);

      var hit = el("rect", { x: cx - slot / 2, y: top, width: slot,
        height: base - top + 40, fill: "transparent", style: "cursor:crosshair" });
      hit.addEventListener("mousemove", function (e) {
        show(e, "week of " + d[0] + "\n" + d[1] + " requests · " + d[4] +
          "\nconceded   " + d[2].toFixed(1) + "%\ntoken burn " + d[3].toFixed(1) + "%" +
          (small ? "\n⚠ sample too small to read" : ""));
      });
      hit.addEventListener("mouseleave", hide);
      svg.appendChild(hit);
    });

    document.getElementById("wk").appendChild(svg);

    document.getElementById("wktable").innerHTML = WK.map(function (d) {
      return "<tr><td>" + d[0] + "</td><td class='num'>" + d[1] + "</td><td class='num'>" +
        d[2].toFixed(1) + "%</td><td class='num'>" + d[3].toFixed(1) + "%</td><td>" +
        d[4] + (d[1] < SMALL_N ? " (low n)" : "") + "</td></tr>";
    }).join("");
  })();
})();
