#!/usr/bin/env python3
"""Yearly Camarilla S3 + monthly bullish reversal scanner (NSE).

Run on the 1st of every month (after month end). Scans completed monthly candles,
backtests the same rule over history, and writes a mobile-friendly HTML dashboard.

Usage:  python scanner.py [symbols.txt] [docs/index.html]
symbols.txt: one NSE symbol per line (e.g. RELIANCE). ".NS" is added automatically.
"""
import json, sys, datetime as dt
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf

TOL = 0.015          # zone width around S3 / below S4
RISK = 500           # fixed Rs risk per trade
RR = 3               # target = entry + RR x risk
HORIZONS = (3, 6, 12)  # months for backtest
STRICT = False       # PGINVIT-style filter: downtrend + oversold RSI + S3 reclaim
SMA_N = 20           # months, for the downtrend check
RSI_N, RSI_MAX = 14, 40
PATTERNS = ["Bullish Engulfing", "Bullish Harami", "Morning Star"]
DEFAULT = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "SBIN", "LT", "ITC",
           "BHARTIARTL", "AXISBANK", "MARUTI", "TATASTEEL", "SUNPHARMA", "NTPC", "ONGC"]


def load_symbols(path):
    p = Path(path)
    syms = p.read_text().split() if p.exists() else DEFAULT
    return [s if s.endswith(".NS") else s + ".NS" for s in syms]


def yearly_camarilla(m):
    y = m.groupby(m.index.year).agg(High=("High", "max"), Low=("Low", "min"), Close=("Close", "last"))
    prev = y.shift(1).reindex(m.index.year)
    prev.index = m.index
    rng = prev.High - prev.Low
    m["S3"] = prev.Close - rng * 1.1 / 4
    m["S4"] = prev.Close - rng * 1.1 / 2
    return m


def rsi(c, n=14):
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


def add_patterns(m):
    o, l, c = m.Open, m.Low, m.Close
    m["RSI"] = rsi(c, RSI_N)
    sma = c.rolling(SMA_N).mean()
    # strict: fell below 20M average before, RSI got oversold in last 3 months,
    # and the signal candle closed back ABOVE S3 (S3 reclaimed)
    strict = (c > m.S3) & (m.RSI.rolling(3).min() < RSI_MAX) & (c.shift(2) < sma.shift(2))
    o1, c1, o2, c2 = o.shift(1), c.shift(1), o.shift(2), c.shift(2)
    body = (c - o).abs()
    red1, green = c1 < o1, c > o
    flags = {
        "Bullish Engulfing": red1 & green & (o <= c1) & (c >= o1),
        "Bullish Harami": red1 & green & (o > c1) & (c < o1) & (body < body.shift(1) * 0.6),
        "Morning Star": (c2 < o2) & ((c1 - o1).abs() < body.shift(2) * 0.4)
                        & (pd.concat([o1, c1], axis=1).max(axis=1) < c2) & green & (c > (o2 + c2) / 2),
    }
    low2 = pd.concat([l, l.shift(1)], axis=1).min(axis=1)
    low3 = pd.concat([l, l.shift(1), l.shift(2)], axis=1).min(axis=1)
    for name, f in flags.items():
        plow = low3 if name == "Morning Star" else low2
        zone = (plow <= m.S3 * (1 + TOL)) & (plow >= m.S4 * (1 - TOL))
        sig = f & zone
        if STRICT:
            sig = sig & strict
        m[name] = sig.fillna(False)
        m["low_" + name] = plow
    return m


def process(sym, cur_period):
    m = yf.download(sym, period="max", interval="1mo", auto_adjust=True, progress=False)
    if m is None or m.empty:
        return None
    if isinstance(m.columns, pd.MultiIndex):
        m.columns = m.columns.get_level_values(0)
    m = m[["Open", "High", "Low", "Close"]].dropna()
    m = m[m.index.to_period("M") < cur_period]          # completed months only
    if len(m) < 30:
        return None
    m = add_patterns(yearly_camarilla(m))
    for h in HORIZONS:
        m[f"f{h}"] = m.Close.shift(-h) / m.Close - 1
    return m


def main():
    sym_file = sys.argv[1] if len(sys.argv) > 1 else "symbols.txt"
    out = Path(sys.argv[2] if len(sys.argv) > 2 else "docs/index.html")
    today = pd.Timestamp.today()
    cur = today.to_period("M")
    target = cur - 1
    symbols = load_symbols(sym_file)

    hist, base, latest, scanned = [], [], [], 0
    for s in symbols:
        try:
            m = process(s, cur)
        except Exception as e:
            print("skip", s, e)
            continue
        if m is None:
            continue
        scanned += 1
        base.append(m[[f"f{h}" for h in HORIZONS]])
        for p in PATTERNS:
            rows = m[m[p]]
            for d, r in rows.iterrows():
                rec = {"symbol": s[:-3], "pattern": p, "date": d.strftime("%Y-%m")}
                rec.update({f"f{h}": (None if pd.isna(r[f"f{h}"]) else float(r[f"f{h}"])) for h in HORIZONS})
                hist.append(rec)
                if d.to_period("M") == target:
                    entry, stop = float(r.High), float(r["low_" + p])
                    risk = entry - stop
                    if risk <= 0:
                        continue
                    latest.append({
                        "symbol": s[:-3], "pattern": p, "month": d.strftime("%b %Y"),
                        "close": round(float(r.Close), 2), "S3": round(float(r.S3), 2),
                        "S4": round(float(r.S4), 2), "entry": round(entry, 2),
                        "stop": round(stop, 2), "target": round(entry + RR * risk, 2),
                        "risk_pct": round(risk / entry * 100, 1), "qty": int(RISK // risk),
                        "rsi": round(float(r.RSI), 1)})
    h = pd.DataFrame(hist)
    b = pd.concat(base) if base else pd.DataFrame(columns=[f"f{x}" for x in HORIZONS])

    def stats(df):
        row = {"n": int(len(df))}
        for k in HORIZONS:
            v = df[f"f{k}"].dropna() if len(df) else pd.Series(dtype=float)
            row[f"w{k}"] = round(float((v > 0).mean() * 100), 1) if len(v) else None
            row[f"a{k}"] = round(float(v.mean() * 100), 1) if len(v) else None
            row[f"n{k}"] = int(len(v))
        return row

    bt = [{"name": p, **stats(h[h.pattern == p] if len(h) else h)} for p in PATTERNS]
    bt.append({"name": "All patterns", **stats(h)})
    bt.append({"name": "Any month (baseline)", **{**stats(b), "n": int(len(b))}})

    data = {"generated": today.strftime("%d %b %Y"), "signal_month": target.strftime("%b %Y"),
            "scanned": scanned, "risk": RISK, "rr": RR, "tol": TOL * 100, "strict": STRICT,
            "horizons": list(HORIZONS), "signals": latest, "backtest": bt}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(HTML.replace("__DATA__", json.dumps(data)), encoding="utf-8")
    print(f"{scanned} scanned, {len(latest)} signals for {data['signal_month']} -> {out}")


HTML = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Camarilla S3 Monthly Scanner</title>
<style>
:root{--bg:#0f1419;--card:#1a2129;--tx:#e6edf3;--mu:#8b98a5;--ac:#3fb950;--rd:#f85149;--bd:#2b3640}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--tx);font:15px/1.4 system-ui,sans-serif;padding:12px;max-width:760px;margin:auto}
h1{font-size:19px;margin:4px 0}h2{font-size:16px;margin:18px 0 8px}.mu{color:var(--mu);font-size:13px}
.card{background:var(--card);border:1px solid var(--bd);border-radius:10px;padding:12px;margin:10px 0}
.row{display:flex;justify-content:space-between;gap:8px;flex-wrap:wrap}.g{display:grid;grid-template-columns:repeat(3,1fr);gap:6px;margin-top:8px}
.g div{background:#121820;border-radius:6px;padding:6px;text-align:center}.g b{display:block;font-size:14px}.g span{font-size:11px;color:var(--mu)}
.t{color:var(--ac)}.s{color:var(--rd)}.badge{background:#1f3a2a;color:var(--ac);border-radius:6px;padding:2px 8px;font-size:12px}
table{width:100%;border-collapse:collapse;font-size:13px}th,td{padding:6px 4px;text-align:right;border-bottom:1px solid var(--bd)}
th:first-child,td:first-child{text-align:left}.sc{overflow-x:auto}
input[type=text]{width:100%;background:#121820;border:1px solid var(--bd);color:var(--tx);border-radius:6px;padding:6px;margin-top:8px}
label{display:flex;gap:8px;align-items:center;margin-top:8px}
</style></head><body>
<h1>Camarilla S3 Monthly Scanner</h1>
<div class="mu" id="meta"></div>
<div class="mu" id="rule"></div>
<h2>Signals for <span id="sm"></span></h2><div id="sig"></div>
<h2>Backtest (monthly candles, yearly S3 zone)</h2>
<div class="card sc"><table id="bt"></table></div>
<div class="mu">Win % / average return after N months from the signal close. Compare each row with the baseline to see if S3 adds an edge. Past results do not guarantee future returns.</div>
<script id="data" type="application/json">__DATA__</script>
<script>
const D=JSON.parse(document.getElementById('data').textContent);
const f=n=>Number(n).toLocaleString('en-IN',{maximumFractionDigits:2});
let st={};try{st=JSON.parse(localStorage.getItem('cs3')||'{}')}catch(e){}
const save=()=>{try{localStorage.setItem('cs3',JSON.stringify(st))}catch(e){}};
document.getElementById('meta').textContent=`Updated ${D.generated} | ${D.scanned} stocks scanned | zone S3 \u00b1${D.tol}% | risk \u20b9${D.risk}/trade | target ${D.rr}R`;
document.getElementById('rule').textContent=D.strict?'Rule: dip into yearly S3-S4 zone, below 20M avg before, monthly RSI under 40, bullish pattern closing back above S3':'Rule: bullish pattern near yearly S3';
document.getElementById('sm').textContent=D.signal_month;
const box=document.getElementById('sig');
if(!D.signals.length)box.innerHTML='<div class="card mu">No signals this month.</div>';
D.signals.forEach(x=>{
 const k=x.symbol+x.month,s=st[k]||{};
 const c=document.createElement('div');c.className='card';
 c.innerHTML=`<div class="row"><b>${x.symbol}</b><span class="badge">${x.pattern}</span></div>
 <div class="mu">${x.month} close \u20b9${f(x.close)} | S3 \u20b9${f(x.S3)} | S4 \u20b9${f(x.S4)} | RSI ${x.rsi}</div>
 <div class="g"><div><b>\u20b9${f(x.entry)}</b><span>Entry (high)</span></div><div><b class="s">\u20b9${f(x.stop)}</b><span>Stop (${x.risk_pct}%)</span></div><div><b class="t">\u20b9${f(x.target)}</b><span>Target</span></div></div>
 <div class="row" style="margin-top:8px"><span>Qty for \u20b9${D.risk} risk: <b>${x.qty}</b></span></div>
 <label><input type="checkbox" ${s.taken?'checked':''}> Trade taken</label>
 <input type="text" placeholder="Notes / skip reason" value="${(s.note||'').replace(/"/g,'&quot;')}">`;
 const [cb,tx]=c.querySelectorAll('input');
 cb.onchange=()=>{st[k]={...st[k],taken:cb.checked};save()};
 tx.oninput=()=>{st[k]={...st[k],note:tx.value};save()};
 box.appendChild(c);
});
let h='<tr><th>Pattern</th><th>N</th>'+D.horizons.map(m=>`<th>${m}M win/avg</th>`).join('')+'</tr>';
D.backtest.forEach(r=>{h+=`<tr><td>${r.name}</td><td>${r.n}</td>`+D.horizons.map(m=>r['w'+m]==null?'<td>-</td>':`<td>${r['w'+m]}% / ${r['a'+m]}%</td>`).join('')+'</tr>'});
document.getElementById('bt').innerHTML=h;
</script></body></html>
"""

if __name__ == "__main__":
    main()
