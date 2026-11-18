#!/usr/bin/env python3
"""Fetch the latest price (15-minute bars) for every stock that ever appeared as a signal
and write docs/prices.json for the journal page. Usage: python fetch_prices.py"""
import json, time
from datetime import datetime, timezone, timedelta
from pathlib import Path
import pandas as pd
import yfinance as yf

IST = timezone(timedelta(hours=5, minutes=30))
SIGNALS = Path("docs/signals.json")
OUT = Path("docs/prices.json")


def last_close(sub):
    c = sub["Close"].dropna()
    if c.empty:
        return None
    ts = c.index[-1]
    try:
        ts = ts.tz_convert(IST) if ts.tzinfo else ts
    except Exception:
        pass
    return {"price": round(float(c.iloc[-1]), 2), "time": ts.strftime("%d %b %H:%M")}


def main():
    if not SIGNALS.exists():
        print("No signals.json yet; nothing to do")
        return
    syms = sorted({s["symbol"] for s in json.loads(SIGNALS.read_text())})
    if not syms:
        print("No signals; nothing to do")
        return
    tick = [s + ".NS" for s in syms]
    prices = {}
    try:
        df = yf.download(tick, period="5d", interval="15m", group_by="ticker",
                         auto_adjust=True, progress=False, threads=True)
        multi = isinstance(df.columns, pd.MultiIndex)
        for s, t in zip(syms, tick):
            sub = df[t] if multi and t in df.columns.get_level_values(0) else (None if multi else df)
            if sub is not None:
                r = last_close(sub)
                if r:
                    prices[s] = r
    except Exception as e:
        print("batch download failed:", e)
    for s, t in zip(syms, tick):          # fallback for anything missing
        if s in prices:
            continue
        try:
            h = yf.Ticker(t).history(period="5d", interval="1d")
            r = last_close(h)
            if r:
                prices[s] = r
        except Exception as e:
            print("no price", s, e)
        time.sleep(0.3)
    if not prices:
        print("No prices fetched; keeping old prices.json")
        return
    now = datetime.now(IST)
    OUT.write_text(json.dumps({"ts": int(now.timestamp()), "updated": now.strftime("%d %b %Y %H:%M IST"),
                               "prices": prices}), encoding="utf-8")
    print(f"Wrote {len(prices)} prices to {OUT}")


if __name__ == "__main__":
    main()
