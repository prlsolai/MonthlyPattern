#!/usr/bin/env python3
"""Download the current NIFTY 500 constituents from NSE and write symbols.txt.
If the download fails, the existing symbols.txt is left untouched.
Usage: python get_nifty500.py [symbols.txt]
"""
import io, sys, urllib.request
import pandas as pd

URLS = [
    "https://archives.nseindia.com/content/indices/ind_nifty500list.csv",
    "https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv",
]
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124 Safari/537.36",
           "Accept": "text/csv,*/*"}


def fetch():
    for url in URLS:
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            raw = urllib.request.urlopen(req, timeout=30).read().decode("utf-8-sig")
            df = pd.read_csv(io.StringIO(raw))
            df.columns = [c.strip() for c in df.columns]
            syms = df["Symbol"].astype(str).str.strip().str.upper().dropna().unique().tolist()
            if len(syms) >= 400:
                print(f"Got {len(syms)} symbols from {url}")
                return syms
            print(f"Only {len(syms)} symbols from {url}, ignoring")
        except Exception as e:
            print("Failed", url, e)
    return None


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "symbols.txt"
    syms = fetch()
    if syms:
        open(out, "w").write("\n".join(syms) + "\n")
        print("Wrote", out)
    else:
        print("Download failed; keeping existing", out)
        sys.exit(1)
