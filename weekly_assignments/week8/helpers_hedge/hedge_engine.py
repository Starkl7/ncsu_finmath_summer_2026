"""
Week 8 · Problem 5 — Market maker + threshold delta-hedger, on one week of data.

This is the Week-5 event-driven engine (``code/week05_engine/backtest.py``)
carried over to an *options* book and wired to a delta-hedger. The loop advances
on a merged stream of futures marks and replayed option fills, marks the book,
aggregates it to a net delta, and — when ``|Δ| > threshold`` — rebalances a BTC
perp hedge, exactly the Avellaneda–Stoikov inventory idea with a vector-valued
inventory reduced to its delta.

Two stages, so the threshold *sweep* is cheap:

  ``build_event_frame`` walks the timeline once, replaying fills and marking the
      book. The book's net delta is ``d/dF [ Σ qᵢ · priceᵢ(F) ]`` — a *single*
      PyTorch-autograd backward per event yields the whole book's delta (repo
      rule: Greeks via autograd). All the expensive work lives here.

  ``run_hedge`` replays the precomputed net-delta path through the threshold
      hedger (``code/week08_hedging/multi_greek_hedge.py::threshold_hedge``) and
      accounts P&L. ``threshold=None`` is the unhedged baseline. Cheap, so we
      call it once per trigger in the sweep.

Fill convention: a printed trade at a tracked strike fills the MM on the passive
side at the printed price (aggressor buy ⇒ MM short; aggressor sell ⇒ MM long).

    python hedge_engine.py
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# resolve sibling ``coincall_replay`` whether run as a script or imported as
# ``helpers_hedge.hedge_engine`` (the notebook does the latter)
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
from coincall_replay import (
    AGGRESSOR_BUY, OB_BASE, FUT_BASE, TRADES_BASE,   # noqa: F401  (paths re-exported)
    black76_price, bs_call, _parse_symbol,
    load_futures_path, load_trade_tape, select_book, load_option_iv_series,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT / "code" / "week08_hedging") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "code" / "week08_hedging"))
from multi_greek_hedge import threshold_hedge     # noqa: E402  the 0.1-BTC trigger primitive

_YEAR_SECONDS = 365.25 * 24 * 3600


@dataclass
class HedgeConfig:
    """Backtest knobs. Defaults reproduce the Problem-5 base case."""
    threshold: float = 0.1          # |net delta| trigger, in BTC
    fee_bps: float = 2.5            # taker fee on the perp hedge, bps of notional
    contract_size: float = 1.0     # BTC of underlying per option contract (Coincall = 1)
    qty_cap: float | None = None   # optional cap on a single replayed fill (contracts)
    start: str = "2026-06-16"
    end: str = "2026-06-22"


# ---------------------------------------------------------------------------
# Book marking: analytic prices, autograd net delta
# ---------------------------------------------------------------------------
def _book_value_and_delta(F: float, pos: np.ndarray, K: np.ndarray,
                          T: np.ndarray, iv: np.ndarray, cp: np.ndarray) -> tuple[float, float]:
    """Return (book mark-to-market value, net forward delta) at forward ``F``.

    Value uses the analytic Black-76 price; the net delta is the autograd
    derivative of the *summed* book value w.r.t. F — one backward for the whole
    book. Puts are handled via forward parity (put = call − (F − K), r=0).
    """
    value = 0.0
    for p, k, t, s, c in zip(pos, K, T, iv, cp):
        if p == 0.0:
            continue
        value += p * black76_price(F, float(k), float(t), float(s), c)

    Ft = torch.tensor(float(F), dtype=torch.float64, requires_grad=True)
    total = torch.zeros((), dtype=torch.float64)
    for p, k, t, s, c in zip(pos, K, T, iv, cp):
        if p == 0.0:
            continue
        kk = torch.tensor(float(k), dtype=torch.float64)
        tt = torch.tensor(float(t), dtype=torch.float64)
        ss = torch.tensor(float(s), dtype=torch.float64)
        r0 = torch.tensor(0.0, dtype=torch.float64)
        call = bs_call(Ft, kk, tt, r0, ss)             # Black-76 call (r=0, S=F)
        price = call if c == "C" else call - (Ft - kk)  # put via forward parity
        total = total + float(p) * price
    if total.requires_grad:
        (net_delta,) = torch.autograd.grad(total, Ft)
        return value, float(net_delta)
    return value, 0.0


def _ns(ts_series: pd.Series) -> np.ndarray:
    """int64 nanoseconds-since-epoch, resolution-independent.

    pandas 2.x may carry a datetime column at ``[ms]`` resolution, whose
    ``.astype("int64")`` yields *milliseconds*; ``Timestamp.value`` and this
    helper always yield nanoseconds, so time deltas against expiries stay
    consistent (a ms/ns mismatch silently turns weeks-to-expiry into decades).
    """
    return (ts_series.dt.tz_convert("UTC").dt.tz_localize(None)
            .astype("datetime64[ns]").astype("int64").to_numpy())


def _asof(ts_grid: np.ndarray, val_grid: np.ndarray, ts: np.int64, default: float,
          clamp: bool = False) -> float:
    """Most recent ``val_grid`` at or before ``ts`` (both keyed by int64 ns).

    With ``clamp``, a timestamp before the first grid point returns the first
    value (nearest available) rather than ``default`` — used for IV marks so an
    early fill is marked at the earliest observed vol, never a blind fallback.
    """
    if len(ts_grid) == 0:
        return default
    i = int(np.searchsorted(ts_grid, ts, side="right")) - 1
    if i < 0:
        return float(val_grid[0]) if clamp else default
    return float(val_grid[i])


def build_event_frame(fut: pd.DataFrame, tape: pd.DataFrame, book: list[str],
                      iv_map: dict[str, pd.DataFrame], cfg: HedgeConfig) -> pd.DataFrame:
    """Walk the merged (futures-mark + fill) timeline once, marking the book.

    Returns one row per event with the columns ``run_hedge`` consumes:
    ``ts, F, half_spread, is_fill, net_delta, option_value, option_cash,
    spread_credit``. All autograd work happens here.
    """
    info = {s: _parse_symbol(s) for s in book}
    K = np.array([float(info[s]["strike"]) for s in book])
    cp = np.array([info[s]["cp"] for s in book])
    expiry_ns = np.array([info[s]["expiry"].value for s in book])
    iv_grids = {s: (_ns(iv_map[s]["ts"]), iv_map[s]["iv"].to_numpy()) for s in book}
    fallback_iv = 0.6

    fills = tape[tape["symbol"].isin(book)].copy()
    # unified event stream: 0 = futures mark, 1 = fill
    ev_mark = pd.DataFrame({"ts": fut["ts"], "F": fut["mid"],
                            "half_spread": fut["half_spread"], "kind": 0,
                            "symbol": None, "price": np.nan, "signed_qty": np.nan})
    signed = np.where(fills["trade_side"].to_numpy() == AGGRESSOR_BUY,
                      -1.0, 1.0) * fills["qty"].to_numpy()      # MM passive side
    if cfg.qty_cap is not None:
        signed = np.clip(signed, -cfg.qty_cap, cfg.qty_cap)
    ev_fill = pd.DataFrame({"ts": fills["ts"], "F": np.nan, "half_spread": np.nan,
                            "kind": 1, "symbol": fills["symbol"].to_numpy(),
                            "price": fills["price"].to_numpy(), "signed_qty": signed})
    ev = pd.concat([ev_mark, ev_fill], ignore_index=True).sort_values(
        "ts", kind="stable").reset_index(drop=True)
    ev_ts = _ns(ev["ts"])
    fut_ts = _ns(fut["ts"])
    fut_mid = fut["mid"].to_numpy()
    fut_hs = fut["half_spread"].to_numpy()

    pos = np.zeros(len(book))
    sym_idx = {s: i for i, s in enumerate(book)}
    option_cash = 0.0
    out = []
    for r in range(len(ev)):
        ts = ev_ts[r]
        F = ev["F"].iat[r]
        if not np.isfinite(F):                         # fill row: take forward as-of
            F = _asof(fut_ts, fut_mid, ts, fut_mid[0])
        half_spread = ev["half_spread"].iat[r]
        if not np.isfinite(half_spread):
            half_spread = _asof(fut_ts, fut_hs, ts, float(np.median(fut_hs)))
        iv = np.array([_asof(*iv_grids[s], ts, fallback_iv, clamp=True) for s in book])
        T = np.maximum((expiry_ns - ts) / 1e9 / _YEAR_SECONDS, 1e-6)

        spread_credit = 0.0
        if ev["kind"].iat[r] == 1:                     # replayed fill
            j = sym_idx[ev["symbol"].iat[r]]
            sq = float(ev["signed_qty"].iat[r])
            fill_price = float(ev["price"].iat[r])
            fair_mid = black76_price(F, float(K[j]), float(T[j]), float(iv[j]), cp[j])
            # passive edge vs. fair mid: buy below mid / sell above mid earns
            spread_credit = sq * (fair_mid - fill_price)
            option_cash -= sq * fill_price
            pos[j] += sq

        value, net_delta = _book_value_and_delta(F, pos, K, T, iv, cp)
        out.append((ts, F, half_spread, int(ev["kind"].iat[r]),
                    net_delta * cfg.contract_size, value, option_cash, spread_credit))

    frame = pd.DataFrame(out, columns=["ts_ns", "F", "half_spread", "is_fill",
                                       "net_delta", "option_value", "option_cash",
                                       "spread_credit"])
    frame["ts"] = pd.to_datetime(frame["ts_ns"], utc=True)
    return frame


# ---------------------------------------------------------------------------
# Cheap replay of the precomputed net-delta path through the hedger
# ---------------------------------------------------------------------------
def run_hedge(ev: pd.DataFrame, threshold: float | None, cfg: HedgeConfig) -> dict:
    """Replay the net-delta path through the threshold hedger and account P&L.

    ``threshold=None`` runs unhedged (perp stays flat). Returns metrics plus a
    per-event time series. Attribution streams (spread, inventory, hedge P&L,
    −hedge cost) sum to total by construction.
    """
    n = len(ev)
    net_delta = ev["net_delta"].to_numpy()
    F = ev["F"].to_numpy()
    half_spread = ev["half_spread"].to_numpy()
    option_value = ev["option_value"].to_numpy()
    option_cash = ev["option_cash"].to_numpy()
    spread_cum = ev["spread_credit"].cumsum().to_numpy()

    perp = 0.0
    perp_cash = 0.0
    hedge_cost = 0.0
    n_hedges = 0
    turnover = 0.0
    perp_path = np.empty(n)
    total_delta = np.empty(n)
    equity = np.empty(n)
    hcost_path = np.empty(n)

    for i in range(n):
        if threshold is not None:
            port_delta = net_delta[i] + perp          # option book + current perp
            perp, trade = threshold_hedge(port_delta, perp, threshold)
            if trade != 0.0:
                perp_cash -= trade * F[i]             # buy costs cash, sell credits
                cost = abs(trade) * (half_spread[i] + F[i] * cfg.fee_bps * 1e-4)
                hedge_cost += cost
                n_hedges += 1
                turnover += abs(trade)
        perp_path[i] = perp
        total_delta[i] = net_delta[i] + perp
        # equity = option MTM + option cash + perp MTM + perp cash − frictions
        equity[i] = option_value[i] + option_cash[i] + perp * F[i] + perp_cash - hedge_cost
        hcost_path[i] = hedge_cost

    options_pnl = option_value + option_cash
    hedge_pnl = perp_path * F + perp_cash
    inventory_pnl = options_pnl - spread_cum          # option MTM drift (gamma/theta/vega)

    ret = np.diff(equity)
    rms_option_delta = float(np.sqrt(np.mean(net_delta ** 2)))
    rms_total_delta = float(np.sqrt(np.mean(total_delta ** 2)))
    dd = equity - np.maximum.accumulate(equity)

    ts = ev["ts"]
    series = pd.DataFrame({
        "ts": ts, "F": F, "net_option_delta": net_delta, "perp": perp_path,
        "total_delta": total_delta, "equity": equity, "spread_pnl": spread_cum,
        "inventory_pnl": inventory_pnl, "hedge_pnl": hedge_pnl,
        "hedge_cost": hcost_path, "is_fill": ev["is_fill"].to_numpy()})

    metrics = {
        "threshold": threshold if threshold is not None else float("nan"),
        "hedged": threshold is not None,
        "total_pnl": float(equity[-1]),
        "spread_pnl": float(spread_cum[-1]),
        "inventory_pnl": float(inventory_pnl[-1]),
        "hedge_pnl": float(hedge_pnl[-1]),
        "hedge_cost": float(hedge_cost),
        "n_hedges": int(n_hedges),
        "perp_turnover_btc": float(turnover),
        "rms_option_delta": rms_option_delta,          # unhedged directional risk
        "rms_total_delta": rms_total_delta,            # residual after hedging
        "delta_risk_reduction": 1.0 - rms_total_delta / (rms_option_delta + 1e-12),
        "pnl_vol": float(ret.std() * np.sqrt(len(ret))) if len(ret) else 0.0,
        "sharpe": float(ret.mean() / (ret.std() + 1e-12) * np.sqrt(len(ret))) if len(ret) else 0.0,
        "max_drawdown": float(dd.min()),
        "final_perp": float(perp),
    }
    # attribution identity: streams sum to total P&L
    recon = metrics["spread_pnl"] + metrics["inventory_pnl"] + metrics["hedge_pnl"] - metrics["hedge_cost"]
    metrics["attribution_residual"] = float(metrics["total_pnl"] - recon)
    return {"metrics": metrics, "series": series}


def run_sweep(ev: pd.DataFrame, thresholds, cfg: HedgeConfig) -> pd.DataFrame:
    """Unhedged baseline + one hedged run per trigger; return a tidy metrics table."""
    rows = [run_hedge(ev, None, cfg)["metrics"]]
    for thr in thresholds:
        rows.append(run_hedge(ev, thr, cfg)["metrics"])
    return pd.DataFrame(rows)


def load_all(cfg: HedgeConfig, n_strikes: int = 5, iv_cadence: str = "5min"):
    """Convenience loader: futures path, book, per-symbol IV, and the event frame."""
    fut = load_futures_path(cfg.start, cfg.end, resample="1min")
    tape = load_trade_tape(cfg.start, cfg.end)
    book = select_book(tape, n_strikes=n_strikes, window_end=cfg.end)
    iv_map = {s: load_option_iv_series(s, fut, cfg.start, cfg.end, cadence=iv_cadence)
              for s in book}
    ev = build_event_frame(fut, tape, book, iv_map, cfg)
    return fut, tape, book, iv_map, ev


if __name__ == "__main__":
    cfg = HedgeConfig()
    print(f"loading + marking book on {cfg.start}..{cfg.end} ...")
    fut, tape, book, iv_map, ev = load_all(cfg)
    print(f"  book: {book}")
    print(f"  events: {len(ev)}  ({int(ev['is_fill'].sum())} fills)")

    base = run_hedge(ev, cfg.threshold, cfg)["metrics"]
    unh = run_hedge(ev, None, cfg)["metrics"]
    print(f"\n{'metric':22s}{'unhedged':>14s}{'hedged @0.1':>14s}")
    for k in ("total_pnl", "spread_pnl", "inventory_pnl", "hedge_pnl", "hedge_cost",
              "rms_option_delta", "rms_total_delta", "delta_risk_reduction",
              "pnl_vol", "max_drawdown", "n_hedges", "perp_turnover_btc"):
        print(f"{k:22s}{unh.get(k, float('nan')):>14.4f}{base.get(k, float('nan')):>14.4f}")
    print(f"\nattribution residual (hedged): {base['attribution_residual']:.2e}")

    # determinism + monotonic-cost sanity across the sweep
    sweep = run_sweep(ev, [0.02, 0.05, 0.1, 0.2, 0.5], cfg)
    hed = sweep[sweep["hedged"]].sort_values("threshold", ascending=False)
    print("\nthreshold sweep (cost rises as trigger tightens):")
    print(hed[["threshold", "hedge_cost", "n_hedges", "rms_total_delta",
               "delta_risk_reduction", "total_pnl"]].to_string(index=False))
    assert run_hedge(ev, 0.1, cfg)["metrics"] == base, "run must be deterministic"
    print("\ndeterministic replay: OK")
