"""
unusual_flow_detector.py
────────────────────────
Rules-based detection of unusual bullish and bearish options flow.

Each detection returns a list of signal dicts describing what pattern
was found and why.  All detections are inferred from market data —
direction is never guaranteed.

Detection categories
─────────────────────
  BULLISH
    CALL_SWEEP          — high vol/OI ratio on calls, same expiry/strike
    CALL_BLOCK          — single large call contract dominates volume
    CALL_ACCUMULATION   — call volume clustered across multiple strikes
    CALL_PREMIUM_SPIKE  — total call premium far exceeds put premium
    GAMMA_SQUEEZE_SETUP — high gamma at NTM strikes from call activity

  BEARISH
    PUT_SWEEP           — high vol/OI ratio on puts, same expiry/strike
    PUT_BLOCK           — single large put contract dominates volume
    PUT_ACCUMULATION    — put volume clustered across multiple strikes
    PUT_PREMIUM_SPIKE   — total put premium far exceeds call premium
    HEDGE_OR_PROTECTION — high put/call ratio, large OTM puts (may be hedge)

  NEUTRAL / AMBIGUOUS
    MIXED_FLOW          — both calls and puts showing unusual activity
    LOTTO_ACTIVITY      — mostly far-OTM contracts (speculative, low conviction)
    LOW_PARTICIPATION   — total volume too low to draw conclusions
"""

from __future__ import annotations

# ── thresholds ─────────────────────────────────────────────────────────────────

_MIN_TOTAL_VOL      = 500      # below this → LOW_PARTICIPATION, no strong signal
_MIN_PREMIUM        = 50_000   # below this → noise, skip
_CP_BULLISH_THRESH  = 1.5      # cp_ratio > 1.5 = call-skewed
_CP_BEARISH_THRESH  = 0.67     # cp_ratio < 0.67 = put-skewed
_VOL_OI_NOTABLE     = 0.5      # vol/OI > 0.5 = notable
_VOL_OI_SIGNIFICANT = 1.0      # vol/OI > 1.0 = significant
_VOL_OI_EXTREME     = 2.0      # vol/OI > 2.0 = extreme sweep
_CONC_HIGH          = 0.50     # Herfindahl > 0.50 = concentrated in 1-2 strikes
_PREMIUM_SKEW_STRONG = 0.30    # |premium_skew| > 0.30 = lopsided premium
_SWEEP_VOL_THRESH   = 1000     # single-contract volume above this = block
_LOTTO_DOMINANT     = 0.50     # if >50% volume is lotto/far-OTM → flag


def detect_unusual_flow(features: dict) -> list[dict]:
    """
    Run all detection rules against the feature dict.

    Parameters
    ──────────
    features   Output of flow_features.compute_flow_features().

    Returns
    ───────
    list[dict], each with keys:
        signal_type   str   — one of the categories above
        bias          str   — "BULLISH", "BEARISH", "NEUTRAL", "AMBIGUOUS"
        confidence    str   — "HIGH", "MEDIUM", "LOW"
        reason        str   — human-readable explanation
        inferred      bool  — True when direction is inferred, not guaranteed
    """
    detections: list[dict] = []

    total_vol   = features.get("total_vol", 0)
    total_prem  = features.get("total_premium", 0.0)

    # ── guard: not enough activity ────────────────────────────────────────────
    if total_vol < _MIN_TOTAL_VOL:
        detections.append({
            "signal_type": "LOW_PARTICIPATION",
            "bias":        "NEUTRAL",
            "confidence":  "LOW",
            "reason":      f"Total option volume only {total_vol:,} — insufficient to draw conclusions.",
            "inferred":    True,
        })
        return detections

    if total_prem < _MIN_PREMIUM:
        detections.append({
            "signal_type": "LOW_PARTICIPATION",
            "bias":        "NEUTRAL",
            "confidence":  "LOW",
            "reason":      f"Total premium only ${total_prem:,.0f} — likely noise.",
            "inferred":    True,
        })
        return detections

    # ── far-OTM dominated ─────────────────────────────────────────────────────
    if features.get("far_otm_dominated", False):
        lotto_pct = max(
            features.get("call_lotto_pct", 0),
            features.get("put_lotto_pct", 0),
        )
        detections.append({
            "signal_type": "LOTTO_ACTIVITY",
            "bias":        "AMBIGUOUS",
            "confidence":  "LOW",
            "reason":      (
                f"{lotto_pct:.0%} of volume in far-OTM contracts. "
                "Lotto flow has poor predictive value — treat as speculative."
            ),
            "inferred":    True,
        })

    cp_ratio   = features.get("cp_ratio", 1.0)
    call_vol_oi = features.get("call_vol_oi", 0.0)
    put_vol_oi  = features.get("put_vol_oi", 0.0)
    call_conc  = features.get("call_strike_conc", 0.0) + features.get("call_expiry_conc", 0.0)
    put_conc   = features.get("put_strike_conc", 0.0)  + features.get("put_expiry_conc", 0.0)
    prem_skew  = features.get("premium_skew", 0.0)
    call_sweep = features.get("call_sweep_vol", 0)
    put_sweep  = features.get("put_sweep_vol", 0)
    call_block = features.get("call_max_contract_vol", 0)
    put_block  = features.get("put_max_contract_vol", 0)
    unusual_cv = features.get("unusual_call_vol", 0)
    unusual_pv = features.get("unusual_put_vol", 0)
    call_vol   = features.get("call_vol", 0)
    put_vol    = features.get("put_vol", 0)

    bullish_signals = 0
    bearish_signals = 0

    # ── CALL SWEEP ────────────────────────────────────────────────────────────
    if call_vol_oi >= _VOL_OI_EXTREME and cp_ratio >= _CP_BULLISH_THRESH:
        conf = "HIGH" if call_vol_oi >= 3.0 else "MEDIUM"
        detections.append({
            "signal_type": "CALL_SWEEP",
            "bias":        "BULLISH",
            "confidence":  conf,
            "reason":      (
                f"Call vol/OI ratio {call_vol_oi:.2f}x with CP ratio {cp_ratio:.2f}x. "
                "Volume well above existing open interest — consistent with aggressive new call buying."
            ),
            "inferred":    True,
        })
        bullish_signals += 2

    elif call_vol_oi >= _VOL_OI_SIGNIFICANT and cp_ratio >= _CP_BULLISH_THRESH:
        detections.append({
            "signal_type": "CALL_SWEEP",
            "bias":        "BULLISH",
            "confidence":  "MEDIUM",
            "reason":      (
                f"Call vol/OI {call_vol_oi:.2f}x, CP ratio {cp_ratio:.2f}x. "
                "Notable call buying relative to open interest."
            ),
            "inferred":    True,
        })
        bullish_signals += 1

    # ── PUT SWEEP ─────────────────────────────────────────────────────────────
    if put_vol_oi >= _VOL_OI_EXTREME and cp_ratio <= _CP_BEARISH_THRESH:
        conf = "HIGH" if put_vol_oi >= 3.0 else "MEDIUM"
        detections.append({
            "signal_type": "PUT_SWEEP",
            "bias":        "BEARISH",
            "confidence":  conf,
            "reason":      (
                f"Put vol/OI ratio {put_vol_oi:.2f}x with CP ratio {cp_ratio:.2f}x. "
                "Aggressive new put buying — may signal downside hedging or directional bet."
            ),
            "inferred":    True,
        })
        bearish_signals += 2

    elif put_vol_oi >= _VOL_OI_SIGNIFICANT and cp_ratio <= _CP_BEARISH_THRESH:
        detections.append({
            "signal_type": "PUT_SWEEP",
            "bias":        "BEARISH",
            "confidence":  "MEDIUM",
            "reason":      (
                f"Put vol/OI {put_vol_oi:.2f}x, CP ratio {cp_ratio:.2f}x. "
                "Notable put buying relative to open interest."
            ),
            "inferred":    True,
        })
        bearish_signals += 1

    # ── CALL BLOCK ────────────────────────────────────────────────────────────
    if call_block >= _SWEEP_VOL_THRESH and call_vol > 0:
        block_pct = call_block / max(call_vol, 1)
        if block_pct >= 0.20:   # one contract is ≥20% of all call volume
            detections.append({
                "signal_type": "CALL_BLOCK",
                "bias":        "BULLISH",
                "confidence":  "MEDIUM",
                "reason":      (
                    f"Single call contract with {call_block:,} vol "
                    f"({block_pct:.0%} of all call volume). "
                    "Large block trades may indicate institutional positioning."
                ),
                "inferred":    True,
            })
            bullish_signals += 1

    # ── PUT BLOCK ─────────────────────────────────────────────────────────────
    if put_block >= _SWEEP_VOL_THRESH and put_vol > 0:
        block_pct = put_block / max(put_vol, 1)
        if block_pct >= 0.20:
            detections.append({
                "signal_type": "PUT_BLOCK",
                "bias":        "BEARISH",
                "confidence":  "MEDIUM",
                "reason":      (
                    f"Single put contract with {put_block:,} vol "
                    f"({block_pct:.0%} of all put volume). "
                    "Large block — may be hedge or directional bet."
                ),
                "inferred":    True,
            })
            bearish_signals += 1

    # ── CALL ACCUMULATION ─────────────────────────────────────────────────────
    if unusual_cv >= 0.40 * call_vol and call_vol > 0 and call_vol_oi >= _VOL_OI_NOTABLE:
        detections.append({
            "signal_type": "CALL_ACCUMULATION",
            "bias":        "BULLISH",
            "confidence":  "MEDIUM",
            "reason":      (
                f"{unusual_cv:,} call contracts with vol/OI > 1.0 "
                f"({unusual_cv / max(call_vol, 1):.0%} of call activity). "
                "Broad-based unusual call buying across multiple strikes."
            ),
            "inferred":    True,
        })
        bullish_signals += 1

    # ── PUT ACCUMULATION ─────────────────────────────────────────────────────
    if unusual_pv >= 0.40 * put_vol and put_vol > 0 and put_vol_oi >= _VOL_OI_NOTABLE:
        detections.append({
            "signal_type": "PUT_ACCUMULATION",
            "bias":        "BEARISH",
            "confidence":  "MEDIUM",
            "reason":      (
                f"{unusual_pv:,} put contracts with vol/OI > 1.0 "
                f"({unusual_pv / max(put_vol, 1):.0%} of put activity). "
                "Broad-based unusual put buying."
            ),
            "inferred":    True,
        })
        bearish_signals += 1

    # ── CALL PREMIUM SPIKE ────────────────────────────────────────────────────
    if prem_skew >= _PREMIUM_SKEW_STRONG:
        call_prem = features.get("call_premium", 0.0)
        detections.append({
            "signal_type": "CALL_PREMIUM_SPIKE",
            "bias":        "BULLISH",
            "confidence":  "MEDIUM",
            "reason":      (
                f"${call_prem:,.0f} call premium vs "
                f"${features.get('put_premium', 0):,.0f} put premium "
                f"(skew {prem_skew:+.0%}). "
                "Call premium significantly outweighs put premium."
            ),
            "inferred":    True,
        })
        bullish_signals += 1

    # ── PUT PREMIUM SPIKE ─────────────────────────────────────────────────────
    if prem_skew <= -_PREMIUM_SKEW_STRONG:
        put_prem = features.get("put_premium", 0.0)
        detections.append({
            "signal_type": "PUT_PREMIUM_SPIKE",
            "bias":        "BEARISH",
            "confidence":  "MEDIUM",
            "reason":      (
                f"${put_prem:,.0f} put premium vs "
                f"${features.get('call_premium', 0):,.0f} call premium "
                f"(skew {prem_skew:+.0%}). "
                "Put premium significantly outweighs calls — may be hedging or directional."
            ),
            "inferred":    True,
        })
        bearish_signals += 1

    # ── HEDGE OR PROTECTION flag ──────────────────────────────────────────────
    put_lotto = features.get("put_lotto_pct", 0.0)
    if (
        cp_ratio <= 0.5
        and put_vol_oi >= _VOL_OI_NOTABLE
        and put_lotto >= 0.30
    ):
        detections.append({
            "signal_type": "HEDGE_OR_PROTECTION",
            "bias":        "BEARISH",
            "confidence":  "LOW",
            "reason":      (
                f"CP ratio {cp_ratio:.2f}x with {put_lotto:.0%} of put volume in far-OTM. "
                "Far-OTM put buying can indicate portfolio hedging rather than directional bets — "
                "direction is uncertain."
            ),
            "inferred":    True,
        })
        bearish_signals += 1

    # ── MIXED FLOW ────────────────────────────────────────────────────────────
    if bullish_signals >= 1 and bearish_signals >= 1:
        detections.append({
            "signal_type": "MIXED_FLOW",
            "bias":        "AMBIGUOUS",
            "confidence":  "LOW",
            "reason":      (
                f"Both bullish ({bullish_signals} signal(s)) and bearish "
                f"({bearish_signals} signal(s)) indicators present. "
                "Could reflect hedging, straddle positioning, or disagreement."
            ),
            "inferred":    True,
        })

    return detections
