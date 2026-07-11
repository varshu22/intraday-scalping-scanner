"""
INTRADAY / SCALPING SCANNER — Nifty 500
Timeframes: Daily, 4H (resampled from 1h, aligned to 09:15), 30-min

Ships raw features to data.json. Strategy logic (doji breakout,
rectangle breakout, range-width buckets) lives in the dashboard JS.

Designed to run every 30 min during market hours via GitHub Actions
(trigger with cron-job.org -> workflow_dispatch for reliability).
"""

import json
import time
import logging
import warnings
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

logging.getLogger("yfinance").setLevel(logging.CRITICAL)
warnings.filterwarnings("ignore")

IST = timezone(timedelta(hours=5, minutes=30))
CHUNK = 50
OUT_FILE = "data.json"

# ===============================
# UNIVERSE: NIFTY 500 + SECTOR
# ===============================
n500 = pd.read_csv("https://archives.nseindia.com/content/indices/ind_nifty500list.csv")
sector_map = dict(zip(n500["Symbol"], n500["Industry"]))
base_symbols = n500["Symbol"].astype(str).str.strip().tolist()
symbols = [s + ".NS" for s in base_symbols]
print(f"Universe: {len(symbols)} symbols")


# ===============================
# HELPERS
# ===============================
def r2(v):
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if pd.isna(f) or np.isinf(f):
        return None
    return round(f, 2)


def candle_pattern(o, h, l, c):
    body = abs(c - o)
    rng = h - l
    if rng == 0 or pd.isna(rng):
        return "Flat"
    upper = h - max(o, c)
    lower = min(o, c) - l
    b, u, lo = body / rng, upper / rng, lower / rng
    if b < 0.10:
        return "Doji"
    if b > 0.80:
        return "Bullish Marubozu" if c > o else "Bearish Marubozu"
    if lo > 0.50 and b < 0.30 and u < 0.20:
        return "Hammer"
    if u > 0.50 and b < 0.30 and lo < 0.20:
        return "Shooting Star"
    return "Bullish" if c > o else "Bearish"


def wilder_rsi(close, period=14):
    close = pd.Series(close).dropna()
    if len(close) < period + 1:
        return None
    d = close.diff()
    gain = d.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-d.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rsi = 100 - (100 / (1 + gain / loss))
    return r2(rsi.iloc[-1])


def ema_last(close, span):
    close = pd.Series(close).dropna()
    if len(close) < span:
        return None
    return r2(close.ewm(span=span, adjust=False).mean().iloc[-1])


def calc_vwap(df, window=20):
    d = df.tail(window).dropna(subset=["High", "Low", "Close", "Volume"])
    if d.empty or d["Volume"].sum() == 0:
        return None
    tp = (d["High"] + d["Low"] + d["Close"]) / 3
    return r2((tp * d["Volume"]).sum() / d["Volume"].sum())


def candle_block(bar):
    o, h, l, c = float(bar["Open"]), float(bar["High"]), float(bar["Low"]), float(bar["Close"])
    return {
        "o": r2(o), "h": r2(h), "l": r2(l), "c": r2(c),
        "pat": candle_pattern(o, h, l, c),
        "f50": r2(l + 0.5 * (h - l)),
        "f618": r2(l + 0.618 * (h - l)),
    }


def rect(closes_hist, min_bars=4):
    cs = pd.Series(closes_hist).dropna()
    if len(cs) < min_bars:
        return None, None, None
    mx, mn = float(cs.max()), float(cs.min())
    if mn <= 0:
        return None, None, None
    return r2(mx), r2(mn), r2((mx - mn) / mn * 100)


def split_live(df, bar_minutes):
    """Return (completed_df, live_bar_or_None) for an intraday frame."""
    if df.empty:
        return df, None
    now = datetime.now(IST)
    last_start = df.index[-1]
    if last_start.tzinfo is None:
        last_start = last_start.tz_localize(IST)
    else:
        last_start = last_start.tz_convert(IST)
    if last_start + timedelta(minutes=bar_minutes) > now:
        return df.iloc[:-1], df.iloc[-1]
    return df, None


AGG = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}


def process_symbol(base, d1, h1, m30):
    d1 = d1.dropna(subset=["Close"])
    if len(d1) < 30:
        return None

    now = datetime.now(IST)
    daily_is_live = d1.index[-1].date() == now.date() and now.hour < 16
    ltp = float(d1["Close"].iloc[-1])

    # ---- 4H resample aligned to 09:15 session open ----
    h4 = pd.DataFrame()
    if h1 is not None and not h1.empty and "Close" in h1.columns:
        h1 = h1.dropna(subset=["Close"])
        try:
            h4 = h1.resample("4h", offset="9h15min").agg(AGG).dropna(subset=["Close"])
        except Exception:
            h4 = h1.resample("4h").agg(AGG).dropna(subset=["Close"])

    if m30 is not None and not m30.empty and "Close" in m30.columns:
        m30 = m30.dropna(subset=["Close"])
    else:
        m30 = pd.DataFrame()

    # ---- completed histories for rectangles ----
    d_hist = (d1["Close"].iloc[:-1] if daily_is_live else d1["Close"]).tail(21)

    h4_comp, _ = split_live(h4, 240) if not h4.empty else (h4, None)
    m30_comp, _ = split_live(m30, 30) if not m30.empty else (m30, None)

    h4_hist = h4_comp["Close"].tail(13) if not h4_comp.empty else pd.Series(dtype=float)
    m30_hist = m30_comp["Close"].tail(8) if not m30_comp.empty else pd.Series(dtype=float)

    d_max, d_min, d_w = rect(d_hist, 10)
    h4_max, h4_min, h4_w = rect(h4_hist, 6)
    m30_max, m30_min, m30_w = rect(m30_hist, 4)

    # ---- previous completed candles (doji strategies) ----
    pd_bar = d1.iloc[-2] if (daily_is_live and len(d1) >= 2) else d1.iloc[-1]
    p4_bar = h4_comp.iloc[-1] if not h4_comp.empty else None
    p30_bar = m30_comp.iloc[-1] if not m30_comp.empty else None

    row = {
        "s": base,
        "sec": sector_map.get(base),
        "ltp": r2(ltp),
        "dMax": d_max, "dMin": d_min, "dW": d_w,
        "h4Max": h4_max, "h4Min": h4_min, "h4W": h4_w,
        "m30Max": m30_max, "m30Min": m30_min, "m30W": m30_w,
        "pd": candle_block(pd_bar) if pd_bar is not None else None,
        "p4": candle_block(p4_bar) if p4_bar is not None else None,
        "p30": candle_block(p30_bar) if p30_bar is not None else None,
        "rsiD": wilder_rsi(d1["Close"]),
        "rsi4": wilder_rsi(h4_comp["Close"]) if not h4_comp.empty else None,
        "rsi30": wilder_rsi(m30_comp["Close"]) if not m30_comp.empty else None,
        "e9": ema_last(d1["Close"], 9),
        "e21": ema_last(d1["Close"], 21),
        "vwap": calc_vwap(d1),
    }

    vol = d1["Volume"].dropna()
    comp_vol = vol.iloc[:-1] if daily_is_live else vol
    row["v7"] = r2(comp_vol.tail(7).mean()) if len(comp_vol) else None
    row["vLive"] = r2(vol.iloc[-1]) if daily_is_live and len(vol) else None
    return row


# ===============================
# BATCH DOWNLOAD + PROCESS
# ===============================
def batch(chunk, **kw):
    try:
        return yf.download(chunk, group_by="ticker", threads=True,
                           progress=False, auto_adjust=False, **kw)
    except Exception:
        return pd.DataFrame()


def pick(data, sym):
    try:
        df = data[sym] if isinstance(data.columns, pd.MultiIndex) else data
        return df.dropna(how="all")
    except Exception:
        return pd.DataFrame()


rows = []
failed = 0

for i in range(0, len(symbols), CHUNK):
    chunk = symbols[i:i + CHUNK]
    d1_all = batch(chunk, period="6mo", interval="1d")
    h1_all = batch(chunk, period="60d", interval="1h")
    m30_all = batch(chunk, period="5d", interval="30m")

    for sym in chunk:
        base = sym.replace(".NS", "")
        try:
            row = process_symbol(base, pick(d1_all, sym), pick(h1_all, sym), pick(m30_all, sym))
            if row:
                rows.append(row)
            else:
                failed += 1
        except Exception:
            failed += 1

    done = min(i + CHUNK, len(symbols))
    print(f"{done}/{len(symbols)} processed | ok={len(rows)} fail={failed}")
    time.sleep(1.0)

# ===============================
# SAVE
# ===============================
payload = {
    "updated": datetime.now(IST).strftime("%Y-%m-%d %H:%M IST"),
    "updatedUtc": datetime.now(timezone.utc).isoformat(),
    "count": len(rows),
    "rows": rows,
}
with open(OUT_FILE, "w") as f:
    json.dump(payload, f, separators=(",", ":"))

print(f"Saved {OUT_FILE}: {len(rows)} rows, {failed} failed")
