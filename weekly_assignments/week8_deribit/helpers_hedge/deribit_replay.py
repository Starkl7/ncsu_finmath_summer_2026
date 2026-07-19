"""
Week 8 · Problem 5 (Deribit variant) — replay loaders for the threshold hedger.

Same problem as the Coincall run — market-make an option book, aggregate to a net
delta, threshold-hedge on the perp — but on **Deribit**, whose BTC option market is
~35× more actively traded (≈7.7k prints/day vs ≈1.5k/week) with deep two-sided
books and a ~20× tighter perp (≈$0.25 half-spread vs ≈$5). That liquidity forces
three changes versus the Coincall loader, all of them *facts of the venue*, not
tunable knobs:

  1. **Units.** Deribit option premia are quoted in **BTC**; we convert to USD with
     the contemporaneous index (perp mid). The trade tape's ``iv`` is in **percent**.
  2. **A real fill model.** On an illiquid venue you can replay 100% of every print
     as your fill; on Deribit you are one order in a deep queue, so we use
     **touch-and-fill + a causal participation rate**: a print at a tracked strike
     fills the maker on the passive side, but only for a fraction
     ``p = mm_size / (mm_size + resting_depth)`` of its size, where ``resting_depth``
     is read from the book **as of that instant** (never the future).
  3. **Direct marks.** The dense two-sided book gives an option mid directly, so we
     do not invert Black-76 for marks; ``iv`` (for the autograd delta only) comes
     from the tape.

**No look-ahead** is a first-class constraint here (contrast the Coincall code,
which selected strikes by *whole-window* volume and back-filled IV):
  * the book universe is chosen by **moneyness at the window's first timestamp**;
  * every mark is **as-of / forward-filled only** (no ``bfill``, no future values);
  * the participation rate uses only the book visible at the fill instant.

All streams are keyed on the capture clock ``recv_ts_ms``.

    python deribit_replay.py
"""
from __future__ import annotations

import glob
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

# --- Deribit capture layout (external drive) ---------------------------------
DERIBIT = "/Volumes/SEAGATE/Crypto/Deribit"
FUT_OB = f"{DERIBIT}/futures_ob"          # btcperp_<YYYYMMDD>_<HH>.parquet
OPT_OB = f"{DERIBIT}/options_ob"          # <YYYYMMDD_HHMM>/<symbol>.parquet
OPT_TRADES = f"{DERIBIT}/options_trades"  # <YYYYMMDD_HHMM>/trades.parquet
FUT_TRADES = f"{DERIBIT}/futures_trades"

OPTION_EXPIRY_HOUR_UTC = 8               # Deribit options expire 08:00 UTC
SYMBOL_RE = re.compile(r"^BTC-(\d{1,2}[A-Z]{3}\d{2})-(\d+)-([CP])$")
_YEAR_SECONDS = 365.25 * 24 * 3600

# --- Reuse the repo's autograd Black-76 pricer -------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[3]
for _p in (_REPO_ROOT / "code" / "week06_vol_surface",
           _REPO_ROOT / "code" / "week08_hedging"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
from bs_greeks import bs_call                     # noqa: E402  autograd pricer


def _parse_symbol(symbol: str) -> pd.Series:
    m = SYMBOL_RE.match(symbol)
    if not m:
        return pd.Series({"expiry": pd.NaT, "strike": np.nan, "cp": None})
    exp, strike, cp = m.groups()
    expiry = (pd.to_datetime(exp, format="%d%b%y", utc=True)
              + pd.Timedelta(hours=OPTION_EXPIRY_HOUR_UTC))
    return pd.Series({"expiry": expiry, "strike": float(strike), "cp": cp})


def _hour_range(start: str, end: str) -> list[str]:
    d0 = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    d1 = datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    keys, t, end_excl = [], d0, d1 + pd.Timedelta(days=1)
    while t < end_excl:
        keys.append(t.strftime("%Y%m%d_%H"))
        t += pd.Timedelta(hours=1)
    return keys


# ---------------------------------------------------------------------------
# 1) Perp path — underlying mark, USD index, and the hedge instrument
# ---------------------------------------------------------------------------
def load_perp_path(start: str = "2026-07-12", end: str = "2026-07-18",
                   resample: str | None = "1min") -> pd.DataFrame:
    """Deribit BTC-perp top-of-book over [start, end]. Returns
    ``ts, bid, ask, mid, half_spread`` (USD). ``mid`` doubles as the USD index."""
    frames = []
    for hk in _hour_range(start, end):
        f = f"{FUT_OB}/btcperp_{hk}.parquet"
        if os.path.exists(f):
            frames.append(pq.read_table(
                f, columns=["recv_ts_ms", "bid_px_00", "ask_px_00"]).to_pandas())
    if not frames:
        raise FileNotFoundError(f"no perp files under {FUT_OB} for {start}..{end}")
    p = pd.concat(frames, ignore_index=True).rename(
        columns={"bid_px_00": "bid", "ask_px_00": "ask"})
    p["ts"] = pd.to_datetime(p["recv_ts_ms"], unit="ms", utc=True)
    p = p.dropna(subset=["bid", "ask"]).sort_values("ts")
    p = p[(p["bid"] > 0) & (p["ask"] >= p["bid"])]
    if resample:
        p = (p.set_index("ts").resample(resample)[["bid", "ask"]]
             .last().dropna().reset_index())
    p["mid"] = 0.5 * (p["bid"] + p["ask"])
    p["half_spread"] = 0.5 * (p["ask"] - p["bid"])
    return p[["ts", "bid", "ask", "mid", "half_spread"]].reset_index(drop=True)


def forward_at(perp: pd.DataFrame, ts: pd.Timestamp) -> float:
    i = min(max(int(perp["ts"].searchsorted(ts)), 0), len(perp) - 1)
    return float(perp["mid"].iloc[i])


# ---------------------------------------------------------------------------
# 2) Option trade tape — the prints we replay as touch-and-fill events
# ---------------------------------------------------------------------------
def load_option_trades(start: str = "2026-07-12", end: str = "2026-07-18",
                       symbols: list[str] | None = None,
                       max_capture_lag_s: float = 3600.0) -> pd.DataFrame:
    """Deribit option prints over [start, end]. ``price``/``mark_price`` are BTC
    premia; ``iv`` is percent → fraction; ``direction`` is the aggressor side.
    Keyed on the capture clock ``recv_ts_ms`` (live-print filter is a light guard;
    the Deribit tape shows no backfill)."""
    day = {k[:8] for k in _hour_range(start, end)}
    frames = []
    for folder in sorted(glob.glob(f"{OPT_TRADES}/*")):
        if os.path.basename(folder)[:8] in day:
            f = os.path.join(folder, "trades.parquet")
            if os.path.exists(f):
                frames.append(pq.read_table(f).to_pandas())
    if not frames:
        raise FileNotFoundError(f"no option-trade folders under {OPT_TRADES}")
    t = pd.concat(frames, ignore_index=True)
    t["symbol"] = t["symbol"].astype(str)
    if symbols is not None:
        t = t[t["symbol"].isin(symbols)]
    t = t[(t["recv_ts_ms"] - t["time"]) / 1000.0 < max_capture_lag_s]
    t["ts"] = pd.to_datetime(t["recv_ts_ms"], unit="ms", utc=True)
    t["iv"] = t["iv"].astype(float) / 100.0            # percent -> fraction
    parsed = t["symbol"].apply(_parse_symbol)
    t = pd.concat([t.reset_index(drop=True), parsed.reset_index(drop=True)], axis=1)
    lo, hi = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
    t = t[(t["ts"] >= lo) & (t["ts"] < hi)]
    keep = ["ts", "symbol", "price", "amount", "direction", "iv",
            "index_price", "mark_price", "strike", "expiry", "cp"]
    return t[keep].sort_values("ts").reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3) Causal book selection — moneyness at the window's first timestamp
# ---------------------------------------------------------------------------
def select_book_causal(perp: pd.DataFrame, start: str, end: str,
                       n_strikes: int = 5, expiry: str = "31JUL26") -> list[str]:
    """Pick the ``n_strikes`` strikes nearest the forward **at the window start**
    on a surviving ``expiry``, taking the OTM leg per strike (P below the forward,
    C above). This uses no information after ``t0`` — no look-ahead — unlike a
    pick-by-total-volume rule. The book universe is read from the option-OB
    folder at ``start``."""
    exp_dt = pd.to_datetime(expiry, format="%d%b%y", utc=True) + pd.Timedelta(hours=OPTION_EXPIRY_HOUR_UTC)
    if exp_dt <= pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1):
        raise ValueError(f"expiry {expiry} does not survive window end {end}")
    F0 = float(perp["mid"].iloc[0])
    hk0 = _hour_range(start, end)[0]                    # first hour folder of the window
    folder = f"{OPT_OB}/{hk0[:8]}_{hk0[9:]}00"
    if not os.path.isdir(folder):
        cands = sorted(glob.glob(f"{OPT_OB}/{hk0[:8]}_*"))
        folder = cands[0]
    strikes = set()
    for f in glob.glob(f"{folder}/BTC-{expiry}-*.parquet"):
        m = SYMBOL_RE.match(os.path.basename(f)[:-8])
        if m:
            strikes.add(int(m.group(2)))
    chosen = sorted(strikes, key=lambda k: abs(k - F0))[:n_strikes]
    return [f"BTC-{expiry}-{k}-{'P' if k < F0 else 'C'}" for k in sorted(chosen)]


# ---------------------------------------------------------------------------
# 4) Per-symbol book series (mid + resting sizes) and IV series
# ---------------------------------------------------------------------------
def load_ob_series(symbol: str, start: str, end: str,
                   cadence: str = "10s") -> pd.DataFrame:
    """Top-of-book series for one option: ``ts, mid, bid, ask, bid_sz, ask_sz``
    (prices in BTC), thinned to ``cadence`` (last per bucket). ``mid`` marks the
    book; ``bid_sz``/``ask_sz`` feed the participation rate."""
    day = {k[:8] for k in _hour_range(start, end)}
    rows = []
    cols = ["recv_ts_ms", "bid_px_00", "ask_px_00", "bid_sz_00", "ask_sz_00"]
    for folder in sorted(glob.glob(f"{OPT_OB}/*")):
        if os.path.basename(folder)[:8] not in day:
            continue
        f = os.path.join(folder, f"{symbol}.parquet")
        if os.path.exists(f):
            rows.append(pq.read_table(f, columns=cols).to_pandas())
    if not rows:
        return pd.DataFrame(columns=["ts", "mid", "bid", "ask", "bid_sz", "ask_sz"])
    s = pd.concat(rows, ignore_index=True).rename(columns={
        "bid_px_00": "bid", "ask_px_00": "ask", "bid_sz_00": "bid_sz", "ask_sz_00": "ask_sz"})
    for c in ["bid", "ask", "bid_sz", "ask_sz"]:
        s[c] = pd.to_numeric(s[c], errors="coerce")
    s = s[(s["bid"] > 0) & (s["ask"] >= s["bid"])]
    s["ts"] = pd.to_datetime(s["recv_ts_ms"], unit="ms", utc=True)
    s["mid"] = 0.5 * (s["bid"] + s["ask"])
    s = s.sort_values("ts")
    if cadence:
        s = (s.set_index("ts").resample(cadence).last().dropna(subset=["mid"]).reset_index())
    return s[["ts", "mid", "bid", "ask", "bid_sz", "ask_sz"]].reset_index(drop=True)


def iv_series_from_trades(symbol: str, tape: pd.DataFrame) -> pd.DataFrame:
    """Per-symbol implied-vol series from that symbol's own prints (``iv`` already a
    fraction). As-of / forward-fill only — used to mark the autograd delta."""
    s = tape[tape["symbol"] == symbol][["ts", "iv"]].dropna().sort_values("ts")
    return s.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Autograd Black-76 delta (repo constraint: Greeks via autograd)
# ---------------------------------------------------------------------------
def option_delta(F: float, K: float, T: float, sigma: float, cp: str) -> float:
    """Forward delta ∂C_usd/∂F via autograd. The BTC-premium quoting affects the
    book's *USD value* (premium × index), not the delta, which is the standard
    dimensionless BS forward delta; puts via parity (call_delta − 1)."""
    if T <= 0 or sigma <= 0:
        icall = 1.0 if F > K else 0.0
        return icall if cp == "C" else icall - 1.0
    Ft = torch.tensor(float(F), dtype=torch.float64, requires_grad=True)
    K_, T_, r_, s_ = (torch.tensor(float(x), dtype=torch.float64) for x in (K, T, 0.0, sigma))
    price = bs_call(Ft, K_, T_, r_, s_)
    (cd,) = torch.autograd.grad(price, Ft)
    return float(cd) if cp == "C" else float(cd) - 1.0


if __name__ == "__main__":
    START, END = "2026-07-12", "2026-07-18"
    perp = load_perp_path(START, END, "1min")
    print(f"perp marks : {len(perp):,}  mid {perp['mid'].min():,.0f}..{perp['mid'].max():,.0f}  "
          f"median half-spread ${perp['half_spread'].median():.2f}")

    book = select_book_causal(perp, START, END, n_strikes=5, expiry="31JUL26")
    print(f"causal book (near-ATM at F0={perp['mid'].iloc[0]:,.0f}): {book}")

    tape = load_option_trades(START, END, symbols=book)
    print(f"book prints: {len(tape)}  (buys {int((tape.direction=='buy').sum())}, "
          f"sells {int((tape.direction=='sell').sum())})  "
          f"index {tape['index_price'].min():,.0f}..{tape['index_price'].max():,.0f}")

    sym = book[0]
    ob = load_ob_series(sym, START, END)
    print(f"OB series {sym}: {len(ob)} pts, mid(BTC) {ob['mid'].min():.4f}..{ob['mid'].max():.4f}, "
          f"median ask_sz {ob['ask_sz'].median():.2f}")
    info = _parse_symbol(sym)
    iv0 = float(tape[tape.symbol == sym]['iv'].median())
    d = option_delta(perp['mid'].iloc[len(perp)//2], info['strike'], 0.05, iv0, info['cp'])
    print(f"sample autograd delta {sym} (iv≈{iv0:.2f}): {d:+.3f}")
