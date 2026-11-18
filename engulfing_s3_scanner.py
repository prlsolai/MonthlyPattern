#!/usr/bin/env python3
"""Monthly Bullish Engulfing near yearly Camarilla S3 (NSE).

Rule (monthly candles, yearly Camarilla levels from the previous calendar year):
  - Bullish engulfing: green candle whose body engulfs the previous red candle's body.
  - Lowest low of the two candles sits near S3: between S4 (-TOL) and S3 (+TOL).
  - Entry = signal candle high, stop = lowest low of the two candles, target = 1:2.

Writes docs/index.html (signals + backtest), docs/journal.html (trade journal with live
P/L) and docs/signals.json (every signal ever shown, used by fetch_prices.py).
Run on the 1st of each month.  Usage: python engulfing_s3_scanner.py [symbols.txt] [docs/index.html]
"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf

TOL = 0.015            # zone width around S3 / below S4
RISK = 500             # fixed Rs risk per trade
RR = 2                 # target = entry + 2 x risk  (1:2)
MAXHOLD = 12           # months, backtest simulation limit
HORIZONS = (3, 6, 12)
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


def process(sym, cur):
    m = yf.download(sym, period="max", interval="1mo", auto_adjust=True, progress=False)
    if m is None or m.empty:
        return None
    if isinstance(m.columns, pd.MultiIndex):
        m.columns = m.columns.get_level_values(0)
    m = m[["Open", "High", "Low", "Close"]].dropna()
    m = m[m.index.to_period("M") < cur]            # completed months only
    if len(m) < 30:
        return None
    m = yearly_camarilla(m)
    o, l, c = m.Open, m.Low, m.Close
    o1, c1 = o.shift(1), c.shift(1)
    engulf = (c1 < o1) & (c > o) & (o <= c1) & (c >= o1)
    m["plow"] = pd.concat([l, l.shift(1)], axis=1).min(axis=1)
    zone = (m.plow <= m.S3 * (1 + TOL)) & (m.plow >= m.S4 * (1 - TOL))
    m["sig"] = (engulf & zone).fillna(False)
    for h in HORIZONS:
        m[f"f{h}"] = c.shift(-h) / c - 1
    return m


def simulate(m):
    """1:2 trade simulation on monthly bars. Entry triggers if price trades above the
    signal high within 2 months; stop is checked before target (conservative)."""
    H, L, C, PL = m.High.values, m.Low.values, m.Close.values, m.plow.values
    n, out = len(m), []
    for i in np.where(m.sig.values)[0]:
        entry, stop = H[i], PL[i]
        risk = entry - stop
        if not risk > 0:
            continue
        target = entry + RR * risk
        start = next((j for j in range(i + 1, min(i + 3, n)) if H[j] >= entry), None)
        if start is None:
            out.append(("nt", 0.0))
            continue
        end = min(start + MAXHOLD, n) - 1
        r = None
        for k in range(start, end + 1):
            if L[k] <= stop:
                r = -1.0
                break
            if H[k] >= target:
                r = float(RR)
                break
        if r is None:
            if end - start + 1 < MAXHOLD:
                out.append(("open", 0.0))
                continue
            r = (C[end] - entry) / risk
        out.append(("done", float(r)))
    return out


def main():
    sym_file = sys.argv[1] if len(sys.argv) > 1 else "symbols.txt"
    out = Path(sys.argv[2] if len(sys.argv) > 2 else "docs/index.html")
    today = pd.Timestamp.today()
    cur = today.to_period("M")
    target = cur - 1
    hist, base, latest, sims, scanned = [], [], [], [], 0
    for s in load_symbols(sym_file):
        try:
            m = process(s, cur)
        except Exception as e:
            print("skip", s, e)
            continue
        if m is None:
            continue
        scanned += 1
        cols = [f"f{h}" for h in HORIZONS]
        base.append(m[cols])
        sig = m[m.sig]
        hist.append(sig[cols])
        sims += simulate(m)
        for d, r in sig[sig.index.to_period("M") == target].iterrows():
            entry, stop = float(r.High), float(r.plow)
            risk = entry - stop
            if risk <= 0:
                continue
            latest.append({
                "id": f"{s[:-3]}-{d.strftime('%Y-%m')}", "symbol": s[:-3], "month": d.strftime("%b %Y"),
                "close": round(float(r.Close), 2), "S3": round(float(r.S3), 2), "S4": round(float(r.S4), 2),
                "entry": round(entry, 2), "stop": round(stop, 2), "target": round(entry + RR * risk, 2),
                "risk_pct": round(risk / entry * 100, 1), "qty": int(RISK // risk)})

    def stats(df):
        row = {"n": int(len(df))}
        for k in HORIZONS:
            v = df[f"f{k}"].dropna() if len(df) else pd.Series(dtype=float)
            row[f"w{k}"] = round(float((v > 0).mean() * 100), 1) if len(v) else None
            row[f"a{k}"] = round(float(v.mean() * 100), 1) if len(v) else None
        return row

    h = pd.concat(hist) if hist else pd.DataFrame(columns=[f"f{x}" for x in HORIZONS])
    b = pd.concat(base) if base else pd.DataFrame(columns=[f"f{x}" for x in HORIZONS])
    bt = [{"name": "Bullish engulfing at S3", **stats(h)}, {"name": "Any month (baseline)", **stats(b)}]
    done = [r for k, r in sims if k == "done"]
    sim = {"signals": len(sims), "triggered": len(done) + sum(1 for k, _ in sims if k == "open"),
           "closed": len(done), "win": round(float(np.mean([r > 0 for r in done]) * 100), 1) if done else None,
           "target": round(float(np.mean([r == RR for r in done]) * 100), 1) if done else None,
           "avgR": round(float(np.mean(done)), 2) if done else None}

    # accumulate every signal ever shown (journal + price updates use this)
    out.parent.mkdir(parents=True, exist_ok=True)
    sj = out.parent / "signals.json"
    try:
        allsig = json.loads(sj.read_text()) if sj.exists() else []
    except Exception:
        allsig = []
    have = {x["id"] for x in allsig}
    allsig += [x for x in latest if x["id"] not in have]
    sj.write_text(json.dumps(allsig), encoding="utf-8")

    data = {"generated": today.strftime("%d %b %Y"), "signal_month": target.strftime("%b %Y"),
            "scanned": scanned, "risk": RISK, "rr": RR, "tol": TOL * 100, "horizons": list(HORIZONS),
            "signals": latest, "backtest": bt, "sim": sim}
    out.write_text(INDEX.replace("__DATA__", json.dumps(data)), encoding="utf-8")
    (out.parent / "journal.html").write_text(JOURNAL, encoding="utf-8")
    print(f"{scanned} scanned, {len(latest)} signals for {data['signal_month']} -> {out}")


CSS = r"""
:root{--bg:#0f1419;--card:#1a2129;--tx:#e6edf3;--mu:#8b98a5;--ac:#3fb950;--rd:#f85149;--bd:#2b3640}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--tx);font:15px/1.4 system-ui,sans-serif;padding:12px;max-width:760px;margin:auto}
h1{font-size:19px;margin:4px 0}h2{font-size:16px;margin:18px 0 8px}.mu{color:var(--mu);font-size:13px}a{color:#58a6ff}
.card{background:var(--card);border:1px solid var(--bd);border-radius:10px;padding:12px;margin:10px 0}
.row{display:flex;justify-content:space-between;gap:8px;flex-wrap:wrap;align-items:center}
.g{display:grid;grid-template-columns:repeat(3,1fr);gap:6px;margin-top:8px}
.g div{background:#121820;border-radius:6px;padding:6px;text-align:center}.g b{display:block;font-size:14px}.g span{font-size:11px;color:var(--mu)}
.t{color:var(--ac)}.s{color:var(--rd)}.badge{background:#1f3a2a;color:var(--ac);border-radius:6px;padding:2px 8px;font-size:12px}
.badge.r{background:#3d1f1f;color:var(--rd)}.badge.n{background:#2b3640;color:var(--mu)}
table{width:100%;border-collapse:collapse;font-size:13px}th,td{padding:6px 4px;text-align:right;border-bottom:1px solid var(--bd)}
th:first-child,td:first-child{text-align:left}.sc{overflow-x:auto}
button{background:#238636;color:#fff;border:0;border-radius:6px;padding:8px 12px;font-size:14px}button.g2{background:#2b3640}button.d{background:#8b2a2a}
input[type=number]{width:90px;background:#121820;border:1px solid var(--bd);color:var(--tx);border-radius:6px;padding:5px}
"""

INDEX = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Monthly Bullish Engulfing at S3</title><style>__CSS__</style></head><body>
<h1>Monthly Bullish Engulfing at Camarilla S3</h1>
<div class="mu" id="meta"></div>
<div class="mu"><a href="journal.html">Open trade journal</a></div>
<h2>Signals for <span id="sm"></span></h2><div id="sig"></div>
<h2>Backtest</h2>
<div class="card" id="sim"></div>
<div class="card sc"><table id="bt"></table></div>
<div class="mu">Win % / average return after N months from the signal close, versus any month. The 1:2 test enters above the signal high within 2 months, checks the stop before the target on each monthly bar, and holds at most 12 months. Past results do not guarantee future returns.</div>
<script id="data" type="application/json">__DATA__</script>
<script>
const D=JSON.parse(document.getElementById('data').textContent);
const f=n=>Number(n).toLocaleString('en-IN',{maximumFractionDigits:2});
const KEY='cs3_journal_v1';
const load=()=>{try{return JSON.parse(localStorage.getItem(KEY)||'[]')}catch(e){return[]}};
const put=a=>{try{localStorage.setItem(KEY,JSON.stringify(a))}catch(e){}};
document.getElementById('meta').textContent=`Updated ${D.generated} | ${D.scanned} stocks scanned | zone \u00b1${D.tol}% | risk \u20b9${D.risk}/trade | target 1:${D.rr}`;
document.getElementById('sm').textContent=D.signal_month;
const box=document.getElementById('sig');
if(!D.signals.length)box.innerHTML='<div class="card mu">No signals this month.</div>';
D.signals.forEach(x=>{
 const c=document.createElement('div');c.className='card';
 c.innerHTML=`<div class="row"><b>${x.symbol}</b><span class="badge">Bullish engulfing</span></div>
 <div class="mu">${x.month} close \u20b9${f(x.close)} | S3 \u20b9${f(x.S3)} | S4 \u20b9${f(x.S4)}</div>
 <div class="g"><div><b>\u20b9${f(x.entry)}</b><span>Entry (high)</span></div><div><b class="s">\u20b9${f(x.stop)}</b><span>Stop (${x.risk_pct}%)</span></div><div><b class="t">\u20b9${f(x.target)}</b><span>Target 1:${D.rr}</span></div></div>
 <div class="row" style="margin-top:8px"><span>Qty for \u20b9${D.risk} risk: <b>${x.qty}</b></span><button></button></div>`;
 const bt=c.querySelector('button');
 const draw=()=>{const has=load().some(t=>t.id===x.id);bt.textContent=has?'In journal \u2713':'Take trade';bt.className=has?'g2':'';bt.disabled=has};
 bt.onclick=()=>{const a=load();if(a.some(t=>t.id===x.id))return;
  a.push({id:x.id,symbol:x.symbol,month:x.month,entry:x.entry,stop:x.stop,target:x.target,qty:x.qty,taken:new Date().toISOString().slice(0,10),status:'open'});put(a);draw()};
 draw();box.appendChild(c);
});
const S=D.sim;
document.getElementById('sim').innerHTML=`<b>1:${D.rr} trade test</b><div class="mu">${S.signals} signals, ${S.triggered} triggered, ${S.closed} finished</div>`+(S.closed?`<div class="g"><div><b>${S.win}%</b><span>Win rate</span></div><div><b>${S.target}%</b><span>Hit target</span></div><div><b>${S.avgR}R</b><span>Average per trade</span></div></div>`:'<div class="mu">Not enough finished trades yet.</div>');
let h='<tr><th>Rule</th><th>N</th>'+D.horizons.map(m=>`<th>${m}M win/avg</th>`).join('')+'</tr>';
D.backtest.forEach(r=>{h+=`<tr><td>${r.name}</td><td>${r.n}</td>`+D.horizons.map(m=>r['w'+m]==null?'<td>-</td>':`<td>${r['w'+m]}% / ${r['a'+m]}%</td>`).join('')+'</tr>'});
document.getElementById('bt').innerHTML=h;
</script></body></html>
""".replace("__CSS__", CSS)

JOURNAL = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Trade Journal</title><style>__CSS__</style></head><body>
<h1>Trade Journal</h1>
<div class="mu" id="stamp">Loading prices...</div>
<div class="mu"><a href="index.html">Back to signals</a> | <a href="#" id="rf">Refresh now</a></div>
<div class="card" id="sum"></div>
<h2>Open trades</h2><div id="open"></div>
<h2>Closed trades</h2><div id="closed"></div>
<script>
const KEY='cs3_journal_v1';
const f=n=>Number(n).toLocaleString('en-IN',{maximumFractionDigits:2});
const load=()=>{try{return JSON.parse(localStorage.getItem(KEY)||'[]')}catch(e){return[]}};
let T=load(),P={};
const save=()=>{try{localStorage.setItem(KEY,JSON.stringify(T))}catch(e){}};
const cls=v=>v>=0?'t':'s', sg=v=>(v>=0?'+':'-')+'\u20b9'+f(Math.abs(v));
async function prices(){
 try{const r=await fetch('prices.json?'+Date.now(),{cache:'no-store'});P=await r.json()}catch(e){P={}}
 render();
}
function age(){
 const el=document.getElementById('stamp');
 if(!P.ts){el.textContent='No price data yet. Prices appear after the first price update runs.';return}
 const m=Math.round((Date.now()/1000-P.ts)/60);
 el.textContent=`Prices updated ${P.updated} (${m<1?'just now':m<90?m+' min ago':Math.round(m/60)+' h ago'}). Page re-checks every 2 minutes.`;
}
function render(){
 age();T=load();
 let openPL=0,real=0,invested=0;
 const o=document.getElementById('open');o.innerHTML='';
 const op=T.filter(t=>t.status!=='closed');
 if(!op.length)o.innerHTML='<div class="card mu">No open trades. Tap "Take trade" on a signal card.</div>';
 op.forEach(t=>{
  const p=P.prices&&P.prices[t.symbol],cmp=p?p.price:null;
  const pl=cmp==null?null:(cmp-t.entry)*t.qty,pct=cmp==null?null:(cmp/t.entry-1)*100;
  if(pl!=null){openPL+=pl;invested+=t.entry*t.qty}
  let badge='';
  if(cmp!=null&&cmp<=t.stop)badge='<span class="badge r">Below stop</span>';
  else if(cmp!=null&&cmp>=t.target)badge='<span class="badge">Target reached</span>';
  else if(cmp!=null)badge='<span class="badge n">Running</span>';
  const c=document.createElement('div');c.className='card';
  c.innerHTML=`<div class="row"><b>${t.symbol}</b>${badge}</div>
  <div class="mu">Signal ${t.month} | taken ${t.taken}</div>
  <div class="g"><div><b>${cmp==null?'-':'\u20b9'+f(cmp)}</b><span>Current price${p?' ('+p.time+')':''}</span></div>
  <div><b class="${pl==null?'':cls(pl)}">${pl==null?'-':sg(pl)}</b><span>Profit / loss</span></div>
  <div><b class="${pct==null?'':cls(pct)}">${pct==null?'-':(pct>=0?'+':'')+pct.toFixed(2)+'%'}</b><span>Return</span></div></div>
  <div class="g"><div><b class="s">\u20b9${f(t.stop)}</b><span>Stop${cmp==null?'':' ('+((cmp/t.stop-1)*100).toFixed(1)+'% away)'}</span></div>
  <div><b class="t">\u20b9${f(t.target)}</b><span>Target${cmp==null?'':' ('+((t.target/cmp-1)*100).toFixed(1)+'% away)'}</span></div>
  <div><b>1:2</b><span>Risk : reward</span></div></div>
  <div class="row" style="margin-top:8px"><span>Entry <input type="number" step="0.05" value="${t.entry}" data-k="entry"> Qty <input type="number" value="${t.qty}" data-k="qty"></span></div>
  <div class="row" style="margin-top:8px"><button class="g2" data-a="close">Close trade</button><button class="d" data-a="del">Delete</button></div>`;
  c.querySelectorAll('input').forEach(i=>i.onchange=()=>{const v=parseFloat(i.value);if(v>0){t[i.dataset.k]=v;save();render()}});
  c.querySelector('[data-a=close]').onclick=()=>{const v=parseFloat(prompt('Exit price for '+t.symbol,cmp==null?t.entry:cmp));if(v>0){t.status='closed';t.exit=v;t.closed=new Date().toISOString().slice(0,10);save();render()}};
  c.querySelector('[data-a=del]').onclick=()=>{if(confirm('Delete '+t.symbol+' from journal?')){T=T.filter(x=>x!==t&&x.id!==t.id);save();render()}};
  o.appendChild(c);
 });
 const cl=document.getElementById('closed');cl.innerHTML='';
 const cs=T.filter(t=>t.status==='closed');
 if(!cs.length)cl.innerHTML='<div class="card mu">No closed trades yet.</div>';
 cs.forEach(t=>{const pl=(t.exit-t.entry)*t.qty;real+=pl;
  const c=document.createElement('div');c.className='card';
  c.innerHTML=`<div class="row"><b>${t.symbol}</b><b class="${cls(pl)}">${sg(pl)}</b></div><div class="mu">Entry \u20b9${f(t.entry)} \u2192 exit \u20b9${f(t.exit)} | qty ${t.qty} | closed ${t.closed}</div>`;
  cl.appendChild(c)});
 document.getElementById('sum').innerHTML=`<div class="g"><div><b class="${cls(openPL)}">${sg(openPL)}</b><span>Open P/L</span></div><div><b class="${cls(real)}">${sg(real)}</b><span>Realised P/L</span></div><div><b class="${cls(openPL+real)}">${sg(openPL+real)}</b><span>Total</span></div></div>`;
}
document.getElementById('rf').onclick=e=>{e.preventDefault();prices()};
prices();setInterval(prices,120000);
</script></body></html>
""".replace("__CSS__", CSS)

if __name__ == "__main__":
    main()
