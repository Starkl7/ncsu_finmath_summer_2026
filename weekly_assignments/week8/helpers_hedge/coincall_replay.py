"""
Week 8 — Coincall replay loaders for the Problem-5 threshold hedger.

Problem 8.5 asks us to run a delta-hedger *on one week of data* and integrated
with the Week-5 engine. This module is the data layer: it turns the recorded
Coincall capture into the three streams the engine consumes.

  1. ``load_futures_path``    — BTC futures top-of-book mid over the week; this
                                is both the underlying mark and the perp we hedge
                                on (the hedge crosses the futures bid/ask spread).
  2. ``load_trade_tape``      — the real executed option prints; each print at a
                                tracked strike is replayed as a *fill against the
                                market maker's resting quote* (passive side).
  3. ``load_option_iv_series``— per-strike implied-vol time series inverted from
                                the sparse option top-of-book snapshots, used to
                                mark each contract's Black-76 delta.

Design choices (see ps8_p5 notebook §1):
  * Options are Black-76 (forward-based, r=0), matching the repo's Week-6 code.
    With r=0, ``bs_greeks.bs_call`` with S=F *is* the Black-76 call, so its
    autograd derivative w.r.t. F is exactly the forward delta we hedge — we
    reuse the repo's autograd pricer rather than hand-rolling a delta.
  * The option order-book stream is snapshot-only (one sparse row per symbol
    per hourly folder), so the *trade tape* is the fill source; the OB snapshots
    only mark IV.

Reuses Week-6 ``helpers_vol/coincall_snapshot`` (``black76_price``,
``implied_vol``, ``_parse_symbol``, ``OPTION_EXPIRY_HOUR_UTC``) and Week-6
``bs_greeks`` (autograd delta), located via the repo root.

    python coincall_replay.py
"""
from __future__ import annotations

import glob
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

# --- Coincall capture layout (external drive; see CLAUDE.md) -----------------
OB_BASE = "/Volumes/SEAGATE/Crypto/Coincall_OB/options_ob_ws"
FUT_BASE = "/Volumes/SEAGATE/Crypto/Coincall_OB/futures_ws"
TRADES_BASE = "/Volumes/SEAGATE/Crypto/Coincall_OB/options_trades_ws"

# Trade-side convention on the Coincall tape: 1 = aggressor BUY (lifts the ask),
# 2 = aggressor SELL (hits the bid). A resting market maker takes the opposite
# side, so aggressor-buy => MM sells (short), aggressor-sell => MM buys (long).
AGGRESSOR_BUY = 1
AGGRESSOR_SELL = 2

# --- Reuse Week-6 helpers via the repo root ----------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[3]
for _p in (_REPO_ROOT / "code" / "week06_vol_surface",
           _REPO_ROOT / "weekly_assignments" / "week6" / "helpers_vol",
           _REPO_ROOT / "code" / "week08_hedging"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from bs_greeks import bs_call                                   # noqa: E402  autograd pricer
from coincall_snapshot import (                                 # noqa: E402
    black76_price, implied_vol, _parse_symbol, OPTION_EXPIRY_HOUR_UTC,
)

_YEAR_SECONDS = 365.25 * 24 * 3600


# ---------------------------------------------------------------------------
# 1) Futures top-of-book — the underlying mark and the hedge instrument
# ---------------------------------------------------------------------------
def _hour_range(start: str, end: str) -> list[str]:
    """Inclusive list of ``YYYYMMDD_HH`` hour keys between two ``YYYY-MM-DD`` days."""
    d0 = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    d1 = datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    keys, t = [], d0
    end_excl = d1 + pd.Timedelta(days=1)
    while t < end_excl:
        keys.append(t.strftime("%Y%m%d_%H"))
        t += pd.Timedelta(hours=1)
    return keys


def load_futures_path(start: str = "2026-06-16", end: str = "2026-06-22",
                      resample: str | None = "1min") -> pd.DataFrame:
    """BTC futures top-of-book over [start, end] (UTC days, inclusive).

    Returns columns ``ts`` (UTC), ``bid``, ``ask``, ``mid``, ``half_spread``.
    ``resample`` (a pandas offset like ``"1min"``) thins the ~kHz book to a
    manageable mark cadence; ``None`` keeps every recorded update.
    """
    frames = []
    for hk in _hour_range(start, end):
        f = f"{FUT_BASE}/btcusd_{hk}.parquet"
        if not os.path.exists(f):
            continue
        frames.append(pq.read_table(
            f, columns=["recv_ts_ms", "bid_px_00", "ask_px_00"]).to_pandas())
    if not frames:
        raise FileNotFoundError(f"no futures files under {FUT_BASE} for {start}..{end}")
    fut = pd.concat(frames, ignore_index=True)
    fut["ts"] = pd.to_datetime(fut["recv_ts_ms"], unit="ms", utc=True)
    fut = fut.rename(columns={"bid_px_00": "bid", "ask_px_00": "ask"})
    fut = fut.dropna(subset=["bid", "ask"]).sort_values("ts")
    fut = fut[(fut["bid"] > 0) & (fut["ask"] >= fut["bid"])]
    if resample:
        fut = (fut.set_index("ts").resample(resample)[["bid", "ask"]]
               .last().dropna().reset_index())
    fut["mid"] = 0.5 * (fut["bid"] + fut["ask"])
    fut["half_spread"] = 0.5 * (fut["ask"] - fut["bid"])
    return fut[["ts", "bid", "ask", "mid", "half_spread"]].reset_index(drop=True)


def forward_at(fut: pd.DataFrame, ts: pd.Timestamp) -> float:
    """Futures mid nearest ``ts`` — the forward proxy used to mark option IV/delta."""
    i = fut["ts"].searchsorted(ts)
    i = min(max(int(i), 0), len(fut) - 1)
    return float(fut["mid"].iloc[i])


# ---------------------------------------------------------------------------
# 2) Trade tape — the replayed option fills
# ---------------------------------------------------------------------------
def load_trade_tape(start: str = "2026-06-16", end: str = "2026-06-22",
                    symbols: list[str] | None = None,
                    max_capture_lag_s: float = 3600.0) -> pd.DataFrame:
    """Real executed option prints over [start, end], parsed and time-sorted.

    Returns ``ts, symbol, price, qty, trade_side, strike, expiry, cp``. If
    ``symbols`` is given, restrict to those contracts.

    Two clocks live in the raw tape: ``time`` is the exchange match time and
    ``recv_ts_ms`` is when our capture saw it. About half the rows are *stale
    backfilled* prints (the WS snapshot replays historical trades — some tens of
    days old), whose ``time`` is far behind ``recv_ts_ms``. Those are not live
    fills and are not aligned with the futures/OB capture clock, so we keep only
    genuinely live prints (capture lag < ``max_capture_lag_s``) and key every
    stream on the **capture clock** (``recv_ts_ms``) so fills, futures marks, and
    IV snapshots are all on one timeline.
    """
    day_prefixes = {k[:8] for k in _hour_range(start, end)}
    frames = []
    for folder in sorted(glob.glob(f"{TRADES_BASE}/*")):
        name = os.path.basename(folder)
        if name[:8] not in day_prefixes:
            continue
        f = os.path.join(folder, "trades.parquet")
        if os.path.exists(f):
            frames.append(pq.read_table(f).to_pandas())
    if not frames:
        raise FileNotFoundError(f"no trade folders under {TRADES_BASE} for {start}..{end}")
    tape = pd.concat(frames, ignore_index=True)
    tape["symbol"] = tape["symbol"].astype(str)
    if symbols is not None:
        tape = tape[tape["symbol"].isin(symbols)]
    tape = tape[(tape["recv_ts_ms"] - tape["time"]) / 1000.0 < max_capture_lag_s]  # live only
    tape["ts"] = pd.to_datetime(tape["recv_ts_ms"], unit="ms", utc=True)           # capture clock
    parsed = tape["symbol"].apply(_parse_symbol)
    tape = pd.concat([tape.reset_index(drop=True), parsed.reset_index(drop=True)], axis=1)
    lo = pd.Timestamp(start, tz="UTC")
    hi = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
    tape = tape[(tape["ts"] >= lo) & (tape["ts"] < hi)]
    return (tape[["ts", "symbol", "price", "qty", "trade_side", "strike", "expiry", "cp"]]
            .sort_values("ts").reset_index(drop=True))


def select_book(tape: pd.DataFrame, n_strikes: int = 5,
                expiry: pd.Timestamp | None = None,
                window_end: str = "2026-06-22") -> list[str]:
    """Pick a small near-ATM book: the ``n_strikes`` most-traded contracts of a
    single expiry.

    Defaults to the most-traded expiry that *survives the whole backtest window*
    (expires after ``window_end``), so no tracked contract expires mid-run — the
    book's delta stays well defined for the full week. Explicit ``expiry`` skips
    the survival filter.
    """
    if expiry is None:
        alive = tape[tape["expiry"] > pd.Timestamp(window_end, tz="UTC") + pd.Timedelta(days=1)]
        expiry = alive.groupby("expiry")["qty"].count().idxmax()
    sub = tape[tape["expiry"] == expiry]
    return sub["symbol"].value_counts().head(n_strikes).index.tolist()


# ---------------------------------------------------------------------------
# 3) Per-strike implied-vol series from the option OB snapshots
# ---------------------------------------------------------------------------
def _option_ob_series(symbol: str, start: str, end: str) -> pd.DataFrame:
    """Concatenate a single option's sparse top-of-book snapshots over the week.

    Handles both capture schema versions (single-level ``bid1_px`` through
    2026-06-17, 10-level ``bid_px_00`` from 06-18), mirroring the Week-6
    normalization.
    """
    day_prefixes = {k[:8] for k in _hour_range(start, end)}
    rows = []
    for folder in sorted(glob.glob(f"{OB_BASE}/*")):
        name = os.path.basename(folder)
        if name[:8] not in day_prefixes:
            continue
        f = os.path.join(folder, f"{symbol}.parquet")
        if not os.path.exists(f):
            continue
        df = pq.read_table(f).to_pandas()
        if "bid1_px" in df.columns and df["bid1_px"].notna().any():
            b, a = df["bid1_px"], df["ask1_px"]
        elif "bid_px_00" in df.columns:
            b, a = df.get("bid_px_00"), df.get("ask_px_00")
        else:
            continue
        out = pd.DataFrame({"recv_ts_ms": df["recv_ts_ms"],
                            "bid": pd.to_numeric(b, errors="coerce"),
                            "ask": pd.to_numeric(a, errors="coerce"),
                            "n_bids": df.get("n_bids", 0), "n_asks": df.get("n_asks", 0)})
        rows.append(out)
    if not rows:
        return pd.DataFrame(columns=["ts", "mid"])
    s = pd.concat(rows, ignore_index=True)
    s = s[(s["n_bids"] > 0) & (s["n_asks"] > 0) & s["bid"].notna() & s["ask"].notna()]
    s["ts"] = pd.to_datetime(s["recv_ts_ms"], unit="ms", utc=True)
    s["mid"] = 0.5 * (s["bid"] + s["ask"])
    return s[["ts", "mid"]].sort_values("ts").reset_index(drop=True)


def load_option_iv_series(symbol: str, fut: pd.DataFrame,
                          start: str = "2026-06-16", end: str = "2026-06-22",
                          fallback_iv: float = 0.6, cadence: str = "5min") -> pd.DataFrame:
    """Implied-vol time series for one option, inverted from its OB-snapshot mid.

    The raw snapshot stream is very dense (tens of thousands of updates per
    active symbol), far finer than we need to *mark* a delta, so we thin it to
    ``cadence`` (last snapshot per bucket) before inverting. For each retained
    snapshot we take the futures forward at that instant and invert the Black-76
    price (Week-6 ``implied_vol``). Sub-intrinsic / one-sided snapshots yield NaN
    and are forward-filled; if the option never quotes two-sidedly we fall back
    to ``fallback_iv`` so the delta mark is always defined. Returns ``ts, mid, iv``.
    """
    info = _parse_symbol(symbol)
    K, cp, expiry = float(info["strike"]), info["cp"], info["expiry"]
    ser = _option_ob_series(symbol, start, end)
    if ser.empty:
        return ser.assign(iv=pd.Series(dtype=float))
    if cadence:
        ser = (ser.set_index("ts").resample(cadence)["mid"].last()
               .dropna().reset_index())
    ivs = []
    for ts, mid in zip(ser["ts"], ser["mid"]):
        F = forward_at(fut, ts)
        T = max((expiry - ts).total_seconds() / _YEAR_SECONDS, 1e-6)
        ivs.append(implied_vol(mid, F, K, T, cp))
    ser["iv"] = pd.Series(ivs).ffill().bfill().fillna(fallback_iv)
    ser["iv"] = ser["iv"].clip(lower=0.05, upper=3.0)
    return ser


# ---------------------------------------------------------------------------
# Autograd Black-76 delta (repo constraint: Greeks via autograd)
# ---------------------------------------------------------------------------
def option_delta(F: float, K: float, T: float, sigma: float, cp: str) -> float:
    """Black-76 forward delta via PyTorch autograd.

    With r=0, ``bs_call(S=F, K, T, 0, sigma)`` is the Black-76 call, so
    d/dF gives the call's forward delta; the put delta follows from parity
    (put_delta = call_delta - 1). Delta is per 1 BTC of underlying notional.
    """
    if T <= 0 or sigma <= 0:                                    # expired / degenerate
        intrinsic_call = 1.0 if F > K else 0.0
        return intrinsic_call if cp == "C" else intrinsic_call - 1.0
    Ft = torch.tensor(float(F), dtype=torch.float64, requires_grad=True)
    args = (torch.tensor(float(x), dtype=torch.float64) for x in (K, T, 0.0, sigma))
    K_, T_, r_, s_ = args
    price = bs_call(Ft, K_, T_, r_, s_)
    call_delta, = torch.autograd.grad(price, Ft)
    d = float(call_delta)
    return d if cp == "C" else d - 1.0


if __name__ == "__main__":
    START, END = "2026-06-16", "2026-06-22"
    print(f"loading futures path {START}..{END} ...")
    fut = load_futures_path(START, END, resample="1min")
    print(f"  futures marks: {len(fut)} rows, "
          f"mid {fut['mid'].min():.0f}..{fut['mid'].max():.0f}, "
          f"median half-spread {fut['half_spread'].median():.2f}")

    print("loading trade tape ...")
    tape = load_trade_tape(START, END)
    print(f"  trades: {len(tape)}   expiries: "
          f"{sorted(tape['expiry'].dt.strftime('%d%b').unique())[:6]}")

    book = select_book(tape, n_strikes=5)
    exp = _parse_symbol(book[0])["expiry"].strftime("%d%b%y")
    print(f"  selected book (expiry {exp}): {book}")
    fills = tape[tape["symbol"].isin(book)]
    print(f"  replayable fills in book: {len(fills)} "
          f"(buys {int((fills.trade_side==AGGRESSOR_BUY).sum())}, "
          f"sells {int((fills.trade_side==AGGRESSOR_SELL).sum())})")

    sym = book[0]
    iv = load_option_iv_series(sym, fut, START, END)
    print(f"  IV series for {sym}: {len(iv)} snapshots, "
          f"iv {iv['iv'].min():.2f}..{iv['iv'].max():.2f} (median {iv['iv'].median():.2f})")
    info = _parse_symbol(sym)
    d = option_delta(fut['mid'].iloc[len(fut)//2], float(info['strike']),
                     0.03, float(iv['iv'].median()), info['cp'])
    print(f"  sample autograd delta for {sym}: {d:+.3f}")
