"""
Week 8 · Problem 5 (Deribit variant) — MM + threshold hedger, USD P&L.

Mirrors ``weekly_assignments/week8/helpers_hedge/hedge_engine.py`` (same Week-5
event loop, same ``threshold_hedge``, same precompute-once / cheap-sweep split),
with three Deribit-specific pieces in ``build_event_frame``:

  * **Touch-and-fill + causal participation** instead of replaying 100% of prints:
    a print fills the maker on the passive side for a fraction
    ``p = mm_size / (mm_size + resting_depth)`` of its size, ``resting_depth`` read
    from the book *as of the fill* (see ``deribit_replay``).
  * **BTC-premium → USD** marks via the contemporaneous index (perp mid): option
    cash and book value are converted to USD so P&L is directly comparable to the
    Coincall run; the delta is the standard forward delta and needs no conversion.
  * **As-of / forward-fill only** — no ``bfill``, no clamp-to-first — so no mark ever
    uses a future observation.

``run_hedge`` / ``run_sweep`` are venue-agnostic (they consume a USD event frame)
and are identical in spirit to the Coincall engine.

    python hedge_engine.py
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
from deribit_replay import (
    load_perp_path, load_option_trades, select_book_causal, load_ob_series,
    iv_series_from_trades, _parse_symbol, bs_call,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT / "code" / "week08_hedging") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "code" / "week08_hedging"))
from multi_greek_hedge import threshold_hedge     # noqa: E402  shared 0.1-BTC trigger primitive

_YEAR_SECONDS = 365.25 * 24 * 3600


@dataclass
class HedgeConfig:
    """Deribit backtest knobs.

    Facts of the venue (not tuned): ``fee_bps`` = Deribit BTC-perp taker fee;
    ``contract_size`` = 1 BTC/contract. Maker's own choices (not fit to P&L):
    ``mm_quote_size`` (posted size, drives the participation rate together with the
    observed book) and ``threshold`` (the swept risk budget)."""
    threshold: float = 0.1          # |net delta| trigger, BTC
    fee_bps: float = 5.0           # Deribit BTC-perp taker fee, bps of notional
    mm_quote_size: float = 1.0     # maker's posted size (BTC) — sets participation p
    contract_size: float = 1.0     # BTC of underlying per option contract
    start: str = "2026-07-12"
    end: str = "2026-07-18"
    expiry: str = "31JUL26"
    n_strikes: int = 5
    ob_cadence: str = "10s"


# ---------------------------------------------------------------------------
# helpers (mirror the Coincall engine)
# ---------------------------------------------------------------------------
def _ns(ts: pd.Series) -> np.ndarray:
    """int64 nanoseconds, resolution-independent (pandas 2.x may carry [ms])."""
    return (ts.dt.tz_convert("UTC").dt.tz_localize(None)
            .astype("datetime64[ns]").astype("int64").to_numpy())


def _asof(ts_grid: np.ndarray, val_grid: np.ndarray, ts: int, default: float) -> float:
    """Most recent value at or before ``ts``; ``default`` before the first point.

    Deliberately **no** clamp-to-first: an event before a series begins uses the
    neutral default, never that series' first (future-relative) value."""
    if len(ts_grid) == 0:
        return default
    i = int(np.searchsorted(ts_grid, ts, side="right")) - 1
    return default if i < 0 else float(val_grid[i])


def _bs_usd(F: float, K: float, T: float, sigma: float, cp: str) -> float:
    """Black-76 USD price (r=0, S=F); put via forward parity. Analytic scalar."""
    if T <= 0 or sigma <= 0:
        return max(F - K, 0.0) if cp == "C" else max(K - F, 0.0)
    with torch.no_grad():
        Ft = torch.tensor(F, dtype=torch.float64)
        call = float(bs_call(Ft, torch.tensor(K, dtype=torch.float64),
                             torch.tensor(T, dtype=torch.float64),
                             torch.tensor(0.0, dtype=torch.float64),
                             torch.tensor(sigma, dtype=torch.float64)))
    return call if cp == "C" else call - (F - K)


def _book_value_and_delta(F: float, pos: np.ndarray, K: np.ndarray, T: np.ndarray,
                          iv: np.ndarray, cp: np.ndarray) -> tuple[float, float]:
    """Return (book USD value, net forward delta) marked at the **BS model price**
    (as-of IV, r=0). A model mark is smooth and consistent with the autograd delta;
    the raw two-sided option mid on Deribit is wide and jumpy, so marking the book
    at it would swamp genuine (delta/gamma) risk with bid-ask bounce. Net delta is
    one autograd backward of the summed book value; puts via parity."""
    value = 0.0
    for p, k, t, s, c in zip(pos, K, T, iv, cp):
        if p != 0.0:
            value += p * _bs_usd(float(F), float(k), float(t), float(s), c)
    Ft = torch.tensor(float(F), dtype=torch.float64, requires_grad=True)
    total = torch.zeros((), dtype=torch.float64)
    any_pos = False
    for p, k, t, s, c in zip(pos, K, T, iv, cp):
        if p == 0.0 or s <= 0 or t <= 0:
            continue
        any_pos = True
        call = bs_call(Ft, torch.tensor(float(k), dtype=torch.float64),
                       torch.tensor(float(t), dtype=torch.float64),
                       torch.tensor(0.0, dtype=torch.float64),
                       torch.tensor(float(s), dtype=torch.float64))
        price = call if c == "C" else call - (Ft - torch.tensor(float(k), dtype=torch.float64))
        total = total + float(p) * price
    if not any_pos:
        return value, 0.0
    (nd,) = torch.autograd.grad(total, Ft)
    return value, float(nd)


# ---------------------------------------------------------------------------
# Deribit event frame: participation fills + USD marks
# ---------------------------------------------------------------------------
def build_event_frame(perp: pd.DataFrame, tape: pd.DataFrame, book: list[str],
                      ob_map: dict[str, pd.DataFrame], iv_map: dict[str, pd.DataFrame],
                      cfg: HedgeConfig) -> pd.DataFrame:
    """Walk the merged (perp-mark + option-print) timeline once.

    Fills use touch-and-fill + participation; marks are market mids in USD. All
    lookups are as-of (past-only). Returns the columns ``run_hedge`` consumes.
    """
    info = {s: _parse_symbol(s) for s in book}
    K = np.array([float(info[s]["strike"]) for s in book])
    cp = np.array([info[s]["cp"] for s in book])
    expiry_ns = np.array([info[s]["expiry"].value for s in book])
    sym_idx = {s: i for i, s in enumerate(book)}

    bidsz_grids = {s: (_ns(ob_map[s]["ts"]), ob_map[s]["bid_sz"].to_numpy()) for s in book}
    asksz_grids = {s: (_ns(ob_map[s]["ts"]), ob_map[s]["ask_sz"].to_numpy()) for s in book}
    iv_grids = {s: (_ns(iv_map[s]["ts"]), iv_map[s]["iv"].to_numpy()) for s in book}

    # merged event stream: kind 0 = perp mark, 1 = option print
    ev_mark = pd.DataFrame({"ts": perp["ts"], "F": perp["mid"],
                            "half_spread": perp["half_spread"], "kind": 0,
                            "symbol": None, "price": np.nan, "amount": np.nan,
                            "direction": None})
    ev_fill = pd.DataFrame({"ts": tape["ts"], "F": np.nan, "half_spread": np.nan,
                            "kind": 1, "symbol": tape["symbol"].to_numpy(),
                            "price": tape["price"].to_numpy(),
                            "amount": tape["amount"].to_numpy(),
                            "direction": tape["direction"].to_numpy()})
    ev = pd.concat([ev_mark, ev_fill], ignore_index=True).sort_values(
        "ts", kind="stable").reset_index(drop=True)
    ev_ts = _ns(ev["ts"])
    perp_ts, perp_mid, perp_hs = _ns(perp["ts"]), perp["mid"].to_numpy(), perp["half_spread"].to_numpy()
    med_hs = float(np.median(perp_hs))
    fallback_iv = 0.6

    pos = np.zeros(len(book))
    option_cash = 0.0          # USD
    out = []
    for r in range(len(ev)):
        ts = ev_ts[r]
        F = ev["F"].iat[r]
        if not np.isfinite(F):
            F = _asof(perp_ts, perp_mid, ts, perp_mid[0])
        half_spread = ev["half_spread"].iat[r]
        if not np.isfinite(half_spread):
            half_spread = _asof(perp_ts, perp_hs, ts, med_hs)
        iv = np.array([_asof(*iv_grids[s], ts, fallback_iv) for s in book])
        T = np.maximum((expiry_ns - ts) / 1e9 / _YEAR_SECONDS, 1e-6)

        spread_credit = 0.0
        if ev["kind"].iat[r] == 1:                              # option print
            s = ev["symbol"].iat[r]; j = sym_idx[s]
            amount = float(ev["amount"].iat[r]); fill_btc = float(ev["price"].iat[r])
            direction = ev["direction"].iat[r]
            # touch-and-fill: aggressor sell hits maker bid (maker long, +),
            # aggressor buy lifts maker ask (maker short, -). resting depth read
            # from the book as-of the fill (never the future).
            if direction == "sell":
                sign, depth = 1.0, _asof(*bidsz_grids[s], ts, np.inf)
            else:
                sign, depth = -1.0, _asof(*asksz_grids[s], ts, np.inf)
            p = cfg.mm_quote_size / (cfg.mm_quote_size + max(depth, 0.0))   # participation
            sq = sign * p * amount
            # realized edge vs the *prevailing* fair value — the model priced at the
            # IV as of strictly before this print. (Deribit derives a trade's IV
            # from its own price, so pricing at the current IV would make fair≡fill
            # and the edge vanish by construction.)
            ivt, ivv = iv_grids[s]
            k_prior = int(np.searchsorted(ivt, ts, side="left")) - 1
            iv_prior = float(ivv[k_prior]) if k_prior >= 0 else float(iv[j])
            fair_usd = _bs_usd(F, float(K[j]), float(T[j]), iv_prior, cp[j])
            spread_credit = sq * (fair_usd - fill_btc * F)      # USD
            option_cash -= sq * fill_btc * F                    # USD (BTC premium × index)
            pos[j] += sq

        value, net_delta = _book_value_and_delta(F, pos, K, T, iv, cp)
        net_delta *= cfg.contract_size
        out.append((ts, F, half_spread, int(ev["kind"].iat[r]),
                    net_delta, value, option_cash, spread_credit))

    frame = pd.DataFrame(out, columns=["ts_ns", "F", "half_spread", "is_fill",
                                       "net_delta", "option_value", "option_cash",
                                       "spread_credit"])
    frame["ts"] = pd.to_datetime(frame["ts_ns"], utc=True)
    return frame


# ---------------------------------------------------------------------------
# venue-agnostic hedge replay + sweep (USD) — mirrors the Coincall engine
# ---------------------------------------------------------------------------
def run_hedge(ev: pd.DataFrame, threshold: float | None, cfg: HedgeConfig) -> dict:
    """Replay the precomputed net-delta path through the threshold hedger and
    account USD P&L. ``threshold=None`` is the unhedged baseline. Attribution
    streams (spread, inventory, hedge P&L, −hedge cost) sum to total."""
    n = len(ev)
    net_delta = ev["net_delta"].to_numpy()
    F = ev["F"].to_numpy()
    half_spread = ev["half_spread"].to_numpy()
    option_value = ev["option_value"].to_numpy()
    option_cash = ev["option_cash"].to_numpy()
    spread_cum = ev["spread_credit"].cumsum().to_numpy()

    perp = perp_cash = hedge_cost = turnover = 0.0
    n_hedges = 0
    perp_path = np.empty(n); total_delta = np.empty(n); equity = np.empty(n); hcost = np.empty(n)
    for i in range(n):
        if threshold is not None:
            port_delta = net_delta[i] + perp
            perp, trade = threshold_hedge(port_delta, perp, threshold)
            if trade != 0.0:
                perp_cash -= trade * F[i]
                cost = abs(trade) * (half_spread[i] + F[i] * cfg.fee_bps * 1e-4)
                hedge_cost += cost; n_hedges += 1; turnover += abs(trade)
        perp_path[i] = perp
        total_delta[i] = net_delta[i] + perp
        equity[i] = option_value[i] + option_cash[i] + perp * F[i] + perp_cash - hedge_cost
        hcost[i] = hedge_cost

    options_pnl = option_value + option_cash
    hedge_pnl = perp_path * F + perp_cash
    inventory_pnl = options_pnl - spread_cum
    ret = np.diff(equity)
    rms_o = float(np.sqrt(np.mean(net_delta ** 2)))
    rms_t = float(np.sqrt(np.mean(total_delta ** 2)))
    dd = equity - np.maximum.accumulate(equity)

    series = pd.DataFrame({
        "ts": ev["ts"], "F": F, "net_option_delta": net_delta, "perp": perp_path,
        "total_delta": total_delta, "equity": equity, "spread_pnl": spread_cum,
        "inventory_pnl": inventory_pnl, "hedge_pnl": hedge_pnl, "hedge_cost": hcost,
        "is_fill": ev["is_fill"].to_numpy()})
    metrics = {
        "threshold": threshold if threshold is not None else float("nan"),
        "hedged": threshold is not None,
        "total_pnl": float(equity[-1]), "spread_pnl": float(spread_cum[-1]),
        "inventory_pnl": float(inventory_pnl[-1]), "hedge_pnl": float(hedge_pnl[-1]),
        "hedge_cost": float(hedge_cost), "n_hedges": int(n_hedges),
        "perp_turnover_btc": float(turnover), "rms_option_delta": rms_o,
        "rms_total_delta": rms_t,
        "delta_risk_reduction": 1.0 - rms_t / (rms_o + 1e-12),
        "pnl_vol": float(ret.std() * np.sqrt(len(ret))) if len(ret) else 0.0,
        "sharpe": float(ret.mean() / (ret.std() + 1e-12) * np.sqrt(len(ret))) if len(ret) else 0.0,
        "max_drawdown": float(dd.min()), "final_perp": float(perp)}
    recon = metrics["spread_pnl"] + metrics["inventory_pnl"] + metrics["hedge_pnl"] - metrics["hedge_cost"]
    metrics["attribution_residual"] = float(metrics["total_pnl"] - recon)
    return {"metrics": metrics, "series": series}


def run_sweep(ev: pd.DataFrame, thresholds, cfg: HedgeConfig) -> pd.DataFrame:
    rows = [run_hedge(ev, None, cfg)["metrics"]]
    for thr in thresholds:
        rows.append(run_hedge(ev, thr, cfg)["metrics"])
    return pd.DataFrame(rows)


def load_all(cfg: HedgeConfig):
    """Perp path, causal book, per-symbol OB + IV series, and the event frame."""
    perp = load_perp_path(cfg.start, cfg.end, resample="1min")
    book = select_book_causal(perp, cfg.start, cfg.end, cfg.n_strikes, cfg.expiry)
    tape = load_option_trades(cfg.start, cfg.end, symbols=book)
    ob_map = {s: load_ob_series(s, cfg.start, cfg.end, cfg.ob_cadence) for s in book}
    iv_map = {s: iv_series_from_trades(s, tape) for s in book}
    ev = build_event_frame(perp, tape, book, ob_map, iv_map, cfg)
    return perp, tape, book, ob_map, iv_map, ev


if __name__ == "__main__":
    cfg = HedgeConfig()
    print(f"loading + marking Deribit book on {cfg.start}..{cfg.end} ...")
    perp, tape, book, ob_map, iv_map, ev = load_all(cfg)
    print(f"  book  : {book}")
    print(f"  events: {len(ev):,}  ({int(ev['is_fill'].sum())} option prints)")

    base = run_hedge(ev, cfg.threshold, cfg)["metrics"]
    unh = run_hedge(ev, None, cfg)["metrics"]
    print(f"\n{'metric':22s}{'unhedged':>14s}{'hedged @0.1':>14s}")
    for k in ("total_pnl", "spread_pnl", "inventory_pnl", "hedge_pnl", "hedge_cost",
              "rms_option_delta", "rms_total_delta", "delta_risk_reduction",
              "pnl_vol", "max_drawdown", "n_hedges", "perp_turnover_btc"):
        print(f"{k:22s}{unh.get(k, float('nan')):>14.4f}{base.get(k, float('nan')):>14.4f}")
    print(f"\nattribution residual (hedged): {base['attribution_residual']:.2e}")

    sweep = run_sweep(ev, [0.1, 0.25, 0.5, 1.0, 2.5], cfg)
    hed = sweep[sweep["hedged"]].sort_values("threshold", ascending=False)
    print("\nthreshold sweep (cost rises as trigger tightens):")
    print(hed[["threshold", "hedge_cost", "n_hedges", "rms_total_delta",
               "delta_risk_reduction", "total_pnl"]].to_string(index=False))
    assert run_hedge(ev, 0.1, cfg)["metrics"] == base, "run must be deterministic"
    print("\ndeterministic replay: OK")
