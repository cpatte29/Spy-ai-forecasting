"""
flow_features.py
────────────────
Compute derived flow features from an option chain DataFrame.

All computations are purely from the chain snapshot — no ML model required.

Returns
───────
A flat dict of float / int / bool / str values consumed by
unusual_flow_detector and flow_scorer.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_flow_features(chain_df: pd.DataFrame, stock_price: float = 0.0) -> dict:
    """
    Derive flow features from the raw option chain snapshot DataFrame.

    Parameters
    ──────────
    chain_df      DataFrame returned by option_chain_loader.load_chain().
    stock_price   Current underlying price (used for moneyness, confirmation).

    Returns
    ───────
    dict with all flow feature values.  Safe to pass directly to detector
    and scorer; missing / unavailable values default to neutral (0 or None).
    """
    if chain_df is None or chain_df.empty:
        return _empty_features(stock_price)

    calls = chain_df[chain_df["contract_type"] == "call"]
    puts  = chain_df[chain_df["contract_type"] == "put"]

    # ── Volume aggregates ─────────────────────────────────────────────────────
    call_vol  = int(calls["volume"].sum())
    put_vol   = int(puts["volume"].sum())
    total_vol = call_vol + put_vol

    cp_ratio  = (call_vol / max(put_vol, 1)) if put_vol > 0 else (10.0 if call_vol > 0 else 1.0)
    pc_ratio  = (put_vol  / max(call_vol, 1)) if call_vol > 0 else (10.0 if put_vol > 0 else 1.0)

    # ── Open interest ─────────────────────────────────────────────────────────
    call_oi  = int(calls["open_interest"].sum())
    put_oi   = int(puts["open_interest"].sum())
    total_oi = call_oi + put_oi

    # Aggregate vol/OI ratios — weighted by volume
    def _weighted_vol_oi(sub: pd.DataFrame) -> float:
        if sub.empty or sub["volume"].sum() == 0:
            return 0.0
        vol = sub["volume"].astype(float)
        voi = sub["vol_oi_ratio"].astype(float)
        return float((vol * voi).sum() / vol.sum())

    call_vol_oi = _weighted_vol_oi(calls)
    put_vol_oi  = _weighted_vol_oi(puts)

    # ── Premium estimates ─────────────────────────────────────────────────────
    call_premium = float(calls["premium_est"].sum())
    put_premium  = float(puts["premium_est"].sum())
    total_premium = call_premium + put_premium

    premium_skew = (
        (call_premium - put_premium) / max(total_premium, 1)
        if total_premium > 0 else 0.0
    )

    # ── Top contracts by volume ───────────────────────────────────────────────
    top_call = (
        calls.sort_values("volume", ascending=False).iloc[0].to_dict()
        if not calls.empty else {}
    )
    top_put = (
        puts.sort_values("volume", ascending=False).iloc[0].to_dict()
        if not puts.empty else {}
    )

    top_contract = (
        chain_df.sort_values("volume", ascending=False).iloc[0].to_dict()
        if not chain_df.empty else {}
    )

    # ── Strike concentration (Herfindahl index, 0=spread, 1=one strike) ──────
    def _concentration(sub: pd.DataFrame, col: str = "strike") -> float:
        if sub.empty or sub["volume"].sum() == 0:
            return 0.0
        vol_by = sub.groupby(col)["volume"].sum()
        shares = vol_by / vol_by.sum()
        return float((shares ** 2).sum())

    call_strike_conc = _concentration(calls, "strike")
    put_strike_conc  = _concentration(puts, "strike")
    call_expiry_conc = _concentration(calls, "expiry")
    put_expiry_conc  = _concentration(puts, "expiry")

    # ── Days-to-expiry distribution ───────────────────────────────────────────
    def _weighted_dte(sub: pd.DataFrame) -> float:
        if sub.empty or sub["volume"].sum() == 0:
            return 0.0
        vol = sub["volume"].astype(float)
        return float((vol * sub["dte"]).sum() / vol.sum())

    call_avg_dte = _weighted_dte(calls)
    put_avg_dte  = _weighted_dte(puts)

    # Volume in each DTE bucket (volume-weighted %)
    def _dte_bucket_pct(sub: pd.DataFrame, lo: int, hi: int) -> float:
        if sub.empty or sub["volume"].sum() == 0:
            return 0.0
        mask = (sub["dte"] >= lo) & (sub["dte"] <= hi)
        return float(sub.loc[mask, "volume"].sum() / sub["volume"].sum())

    call_0_7_dte_pct   = _dte_bucket_pct(calls, 0, 7)
    call_8_30_dte_pct  = _dte_bucket_pct(calls, 8, 30)
    call_31_90_dte_pct = _dte_bucket_pct(calls, 31, 90)

    # ── Near-the-money vs far-OTM split ──────────────────────────────────────
    def _moneyness_pct(sub: pd.DataFrame, label: str) -> float:
        if sub.empty or sub["volume"].sum() == 0:
            return 0.0
        mask = sub["moneyness"] == label
        return float(sub.loc[mask, "volume"].sum() / sub["volume"].sum())

    call_ntm_pct  = _moneyness_pct(calls, "NTM")
    call_otm_pct  = _moneyness_pct(calls, "OTM")
    call_lotto_pct = (
        _moneyness_pct(calls, "FAR_OTM") + _moneyness_pct(calls, "LOTTO")
    )
    put_ntm_pct   = _moneyness_pct(puts, "NTM")
    put_lotto_pct = (
        _moneyness_pct(puts, "FAR_OTM") + _moneyness_pct(puts, "LOTTO")
    )

    far_otm_dominated = (call_lotto_pct > 0.60) or (put_lotto_pct > 0.60)

    # ── Implied volatility ────────────────────────────────────────────────────
    def _avg_iv(sub: pd.DataFrame) -> float:
        if sub.empty or sub["volume"].sum() == 0:
            return 0.0
        vol = sub["volume"].astype(float)
        iv  = sub["iv"].astype(float)
        return float((vol * iv).sum() / vol.sum())

    call_avg_iv = _avg_iv(calls)
    put_avg_iv  = _avg_iv(puts)

    # ── Gamma exposure proxy (top contracts) ──────────────────────────────────
    chain_df_cp = chain_df.copy()
    chain_df_cp["gex"] = (
        chain_df_cp["gamma"]
        * chain_df_cp["open_interest"]
        * chain_df_cp["volume"]
        * (stock_price ** 2)
        * 0.01
    )
    net_gex = float(
        chain_df_cp.loc[chain_df_cp["contract_type"] == "call", "gex"].sum()
        - chain_df_cp.loc[chain_df_cp["contract_type"] == "put", "gex"].sum()
    )

    # ── Sweep / block detection heuristic ────────────────────────────────────
    # Contracts where vol > 10x OI are likely opening sweeps (not closing rolls)
    call_sweeps = calls[calls["vol_oi_ratio"] > 10.0]
    put_sweeps  = puts[puts["vol_oi_ratio"]  > 10.0]
    call_sweep_vol = int(call_sweeps["volume"].sum())
    put_sweep_vol  = int(put_sweeps["volume"].sum())

    # Largest single contract by volume (block indicator)
    call_max_contract_vol = int(calls["volume"].max()) if not calls.empty else 0
    put_max_contract_vol  = int(puts["volume"].max())  if not puts.empty else 0

    # ── Unusual vol/OI flag ───────────────────────────────────────────────────
    # Contracts where vol/OI > 1.0 are notably unusual (more volume than existing OI)
    high_vol_oi_calls = calls[calls["vol_oi_ratio"] > 1.0]
    high_vol_oi_puts  = puts[puts["vol_oi_ratio"] > 1.0]
    unusual_call_vol = int(high_vol_oi_calls["volume"].sum())
    unusual_put_vol  = int(high_vol_oi_puts["volume"].sum())

    # ── Directional signal ────────────────────────────────────────────────────
    # Positive = bullish skew, Negative = bearish skew
    # Combines cp_ratio and premium skew
    directional_score_raw = (
        0.5 * (cp_ratio - 1.0) / max(cp_ratio + 1.0, 1.0)
        + 0.5 * premium_skew
    )
    directional_score = float(np.clip(directional_score_raw, -1.0, 1.0))

    return {
        # Volume
        "call_vol":           call_vol,
        "put_vol":            put_vol,
        "total_vol":          total_vol,
        "cp_ratio":           round(cp_ratio, 3),
        "pc_ratio":           round(pc_ratio, 3),
        # Open interest
        "call_oi":            call_oi,
        "put_oi":             put_oi,
        "total_oi":           total_oi,
        # Vol/OI ratios (volume-weighted)
        "call_vol_oi":        round(call_vol_oi, 3),
        "put_vol_oi":         round(put_vol_oi, 3),
        # Unusual activity
        "unusual_call_vol":   unusual_call_vol,
        "unusual_put_vol":    unusual_put_vol,
        "call_sweep_vol":     call_sweep_vol,
        "put_sweep_vol":      put_sweep_vol,
        "call_max_contract_vol": call_max_contract_vol,
        "put_max_contract_vol":  put_max_contract_vol,
        # Premium
        "call_premium":       round(call_premium, 2),
        "put_premium":        round(put_premium, 2),
        "total_premium":      round(total_premium, 2),
        "premium_skew":       round(premium_skew, 4),
        # Concentration
        "call_strike_conc":   round(call_strike_conc, 4),
        "put_strike_conc":    round(put_strike_conc, 4),
        "call_expiry_conc":   round(call_expiry_conc, 4),
        "put_expiry_conc":    round(put_expiry_conc, 4),
        # DTE
        "call_avg_dte":       round(call_avg_dte, 1),
        "put_avg_dte":        round(put_avg_dte, 1),
        "call_0_7_dte_pct":   round(call_0_7_dte_pct, 3),
        "call_8_30_dte_pct":  round(call_8_30_dte_pct, 3),
        "call_31_90_dte_pct": round(call_31_90_dte_pct, 3),
        # Moneyness
        "call_ntm_pct":       round(call_ntm_pct, 3),
        "call_otm_pct":       round(call_otm_pct, 3),
        "call_lotto_pct":     round(call_lotto_pct, 3),
        "put_ntm_pct":        round(put_ntm_pct, 3),
        "put_lotto_pct":      round(put_lotto_pct, 3),
        "far_otm_dominated":  far_otm_dominated,
        # IV
        "call_avg_iv":        round(call_avg_iv, 4),
        "put_avg_iv":         round(put_avg_iv, 4),
        # GEX
        "net_gex":            round(net_gex, 2),
        # Directional composite
        "directional_score":  round(directional_score, 4),
        # Top contracts (used by report)
        "top_call":           top_call,
        "top_put":            top_put,
        "top_contract":       top_contract,
        # Stock price reference
        "stock_price":        stock_price,
    }


def _empty_features(stock_price: float = 0.0) -> dict:
    """Return a zeroed-out features dict when no chain data is available."""
    return {
        "call_vol": 0, "put_vol": 0, "total_vol": 0,
        "cp_ratio": 1.0, "pc_ratio": 1.0,
        "call_oi": 0, "put_oi": 0, "total_oi": 0,
        "call_vol_oi": 0.0, "put_vol_oi": 0.0,
        "unusual_call_vol": 0, "unusual_put_vol": 0,
        "call_sweep_vol": 0, "put_sweep_vol": 0,
        "call_max_contract_vol": 0, "put_max_contract_vol": 0,
        "call_premium": 0.0, "put_premium": 0.0,
        "total_premium": 0.0, "premium_skew": 0.0,
        "call_strike_conc": 0.0, "put_strike_conc": 0.0,
        "call_expiry_conc": 0.0, "put_expiry_conc": 0.0,
        "call_avg_dte": 0.0, "put_avg_dte": 0.0,
        "call_0_7_dte_pct": 0.0, "call_8_30_dte_pct": 0.0, "call_31_90_dte_pct": 0.0,
        "call_ntm_pct": 0.0, "call_otm_pct": 0.0, "call_lotto_pct": 0.0,
        "put_ntm_pct": 0.0, "put_lotto_pct": 0.0,
        "far_otm_dominated": False,
        "call_avg_iv": 0.0, "put_avg_iv": 0.0,
        "net_gex": 0.0,
        "directional_score": 0.0,
        "top_call": {}, "top_put": {}, "top_contract": {},
        "stock_price": stock_price,
    }
