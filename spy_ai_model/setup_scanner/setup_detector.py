"""
setup_detector.py
─────────────────
Detection logic for all six setup types.

Each detector function receives a normalised snapshot dict and returns
a (conditions_met, missing_confirms, invalidation_level, target_zone,
guidance, direction_override) tuple.

Public entry point
──────────────────
  results = detect_setups(snapshot)

where `snapshot` is produced by build_snapshot() in this module.

Snapshot schema (all keys optional with sensible defaults)
──────────────────────────────────────────────────────────
  # bar data
  current_price       float
  current_open        float
  current_high        float
  current_low         float
  bar_ts              pd.Timestamp | None

  # model outputs
  dir_prob            float   P(close[t+60] > close[t])
  pred_range          float   normalized range forecast

  # zone context (from get_zone_context)
  zone_ctx            dict

  # analog (from run_analog_analysis, or {} if unavailable)
  analog              dict

  # confluence (from compute_confluence, or {} if unavailable)
  confluence          dict

  # feature row (optional – used for structure signals)
  features            dict | None

  # chop flag (optional)
  chop_detected       bool
  chop_score          float   0-1 (higher = more choppy)
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from setup_scanner.setup_definitions import (
    SetupDirection,
    SetupType,
)

logger = logging.getLogger(__name__)

# ── zone proximity thresholds ─────────────────────────────────────────────────
_INSIDE_OR_AT   = 0.001   # 0.1 % – effectively inside / touching
_CLOSE_TO_ZONE  = 0.005   # 0.5 %
_NEAR_ZONE      = 0.015   # 1.5 %

# ── model bias thresholds ─────────────────────────────────────────────────────
_MODEL_BULLISH_SOFT  = 0.50   # neutral / slight bullish above this
_MODEL_BULLISH_HARD  = 0.55   # clearly bullish above this
_MODEL_BEARISH_SOFT  = 0.50   # neutral / slight bearish below this
_MODEL_BEARISH_HARD  = 0.45   # clearly bearish below this

# ── trend definition (slope_ema_20_5) ─────────────────────────────────────────
_TREND_UP_MIN   =  0.0003
_TREND_DOWN_MAX = -0.0003

# ── candlestick structure ─────────────────────────────────────────────────────
_BULLISH_BODY_MIN   = 0.3   # body_to_range >= 0.3 → bullish body
_BEARISH_BODY_MIN   = 0.3
_LOWER_WICK_STRONG  = 0.35  # lower_wick_to_range >= 0.35 → pinbar / hammer
_UPPER_WICK_STRONG  = 0.35  # upper_wick_to_range >= 0.35 → shooting-star / rejection
_CLOSE_LOCATION_HIGH = 0.6  # close_location >= 0.6 → closed near top of range
_CLOSE_LOCATION_LOW  = 0.4  # close_location <= 0.4 → closed near bottom

# ── analog thresholds ─────────────────────────────────────────────────────────
_ANALOG_MIN_MATCHES     = 5
_ANALOG_REJECTION_BIAS  = 0.50   # rejection_rate >= this → analog favors rejection
_ANALOG_BREAKOUT_BIAS   = 0.45   # breakout_rate  >= this → analog favors breakout
_ANALOG_STRONG_REJ      = 0.65
_ANALOG_STRONG_BRK      = 0.60

# ── range forecast ────────────────────────────────────────────────────────────
_RANGE_LOW_CUTOFF  = 0.002   # < 0.2% → very low, reduces confidence

# ── VWAP ─────────────────────────────────────────────────────────────────────
_DIST_VWAP_NEAR = 0.003   # within 0.3% of VWAP

# ── distance formatting ───────────────────────────────────────────────────────

def _pct(v: float | None) -> str:
    if v is None:
        return "n/a"
    return f"{v*100:.2f}%"


def _fmt_price(v: float | None) -> str:
    if v is None:
        return "n/a"
    return f"{v:.2f}"


# ── snapshot builder ──────────────────────────────────────────────────────────

def build_snapshot(
    df_bars:       pd.DataFrame,
    dir_prob:      float,
    pred_range:    float,
    zone_ctx:      dict,
    analog:        dict | None = None,
    confluence:    dict | None = None,
    features:      dict | None = None,
    chop_detected: bool        = False,
    chop_score:    float       = 0.0,
    volume_ctx:    dict | None = None,
) -> dict:
    """
    Assemble a normalised snapshot dict from existing system outputs.

    Parameters
    ──────────
    df_bars     Full OHLCV bar DataFrame up to and including the scored bar.
    dir_prob    Direction model P(up).
    pred_range  Range model normalized forecast.
    zone_ctx    Output of get_zone_context().
    analog      Output of run_analog_analysis() (or None / {} if unavailable).
    confluence  Output of compute_confluence() (or None / {} if unavailable).
    features    Dict of feature values for the scored bar (from build_features).
    chop_detected  Boolean flag from an external chop detector.
    chop_score  0–1 choppiness level (1 = maximum chop).
    volume_ctx  Output of compute_volume_context() (or None if unavailable).
    """
    if df_bars.empty:
        raise ValueError("df_bars must not be empty")

    last = df_bars.iloc[-1]
    feat = features or {}

    return {
        # bar OHLCV
        "current_price": float(last["close"]),
        "current_open":  float(last["open"]),
        "current_high":  float(last["high"]),
        "current_low":   float(last["low"]),
        "bar_ts":        df_bars.index[-1],

        # model
        "dir_prob":   dir_prob,
        "pred_range": pred_range,

        # zone context
        "zone_ctx":   zone_ctx or {},

        # analog / confluence
        "analog":     analog     or {},
        "confluence": confluence or {},

        # structure features (raw values, defaulting to None)
        "body_to_range":       feat.get("body_to_range"),
        "upper_wick_to_range": feat.get("upper_wick_to_range"),
        "lower_wick_to_range": feat.get("lower_wick_to_range"),
        "close_location":      feat.get("close_location"),
        "dist_vwap":           feat.get("dist_vwap"),
        "vwap_reclaim_flag":   feat.get("vwap_reclaim_flag"),
        "vwap_loss_flag":      feat.get("vwap_loss_flag"),
        "vwap_trend_strength": feat.get("vwap_trend_strength"),
        "slope_ema_20_5":      feat.get("slope_ema_20_5"),
        "slope_ema_9_5":       feat.get("slope_ema_9_5"),
        "dist_ema_20":         feat.get("dist_ema_20"),
        "dist_ema_9":          feat.get("dist_ema_9"),
        "rv_15":               feat.get("rv_15"),
        "vol_rel_5":           feat.get("vol_rel_5"),
        "ret_5":               feat.get("ret_5"),
        "ret_15":              feat.get("ret_15"),
        "orb_break_flag":      feat.get("orb_break_flag"),
        "orb_reject_flag":     feat.get("orb_reject_flag"),

        # chop
        "chop_detected": chop_detected,
        "chop_score":    chop_score,

        # volume (from volume_integration.compute_volume_context)
        "volume_ctx":    volume_ctx or {},
    }


# ── helpers ───────────────────────────────────────────────────────────────────

def _get_supply(sn: dict) -> dict:
    return sn["zone_ctx"].get("supply", {})


def _get_demand(sn: dict) -> dict:
    return sn["zone_ctx"].get("demand", {})


def _zone_bias(sn: dict) -> str:
    return sn["zone_ctx"].get("bias", "NEUTRAL")


def _supply_dist(sn: dict) -> float | None:
    d = _get_supply(sn).get("distance_to_zone_pct")
    return float(d) if d is not None else None


def _demand_dist(sn: dict) -> float | None:
    d = _get_demand(sn).get("distance_to_zone_pct")
    return float(d) if d is not None else None


def _analog_active(sn: dict) -> bool:
    return int(sn["analog"].get("n_matches", 0)) >= _ANALOG_MIN_MATCHES


def _bullish_candle(sn: dict) -> bool:
    return sn["current_price"] > sn["current_open"]


def _bearish_candle(sn: dict) -> bool:
    return sn["current_price"] < sn["current_open"]


def _strong_lower_wick(sn: dict) -> bool:
    lw = sn.get("lower_wick_to_range")
    if lw is None:
        bar_range = sn["current_high"] - sn["current_low"]
        if bar_range <= 0:
            return False
        lower_wick = min(sn["current_open"], sn["current_price"]) - sn["current_low"]
        lw = lower_wick / bar_range
    return lw >= _LOWER_WICK_STRONG


def _strong_upper_wick(sn: dict) -> bool:
    uw = sn.get("upper_wick_to_range")
    if uw is None:
        bar_range = sn["current_high"] - sn["current_low"]
        if bar_range <= 0:
            return False
        upper_wick = sn["current_high"] - max(sn["current_open"], sn["current_price"])
        uw = upper_wick / bar_range
    return uw >= _UPPER_WICK_STRONG


def _closed_high_in_range(sn: dict) -> bool:
    cl = sn.get("close_location")
    if cl is None:
        bar_range = sn["current_high"] - sn["current_low"]
        if bar_range <= 0:
            return False
        cl = (sn["current_price"] - sn["current_low"]) / bar_range
    return cl >= _CLOSE_LOCATION_HIGH


def _closed_low_in_range(sn: dict) -> bool:
    cl = sn.get("close_location")
    if cl is None:
        bar_range = sn["current_high"] - sn["current_low"]
        if bar_range <= 0:
            return False
        cl = (sn["current_price"] - sn["current_low"]) / bar_range
    return cl <= _CLOSE_LOCATION_LOW


def _uptrend(sn: dict) -> bool:
    slope = sn.get("slope_ema_20_5")
    if slope is not None:
        return float(slope) >= _TREND_UP_MIN
    # fallback: ret_15
    r15 = sn.get("ret_15")
    return r15 is not None and float(r15) > 0.001


def _downtrend(sn: dict) -> bool:
    slope = sn.get("slope_ema_20_5")
    if slope is not None:
        return float(slope) <= _TREND_DOWN_MAX
    r15 = sn.get("ret_15")
    return r15 is not None and float(r15) < -0.001


def _near_vwap(sn: dict) -> bool:
    dv = sn.get("dist_vwap")
    if dv is None:
        return False
    return abs(float(dv)) <= _DIST_VWAP_NEAR


def _near_ema20(sn: dict) -> bool:
    de = sn.get("dist_ema_20")
    if de is None:
        return False
    return abs(float(de)) <= _DIST_VWAP_NEAR


# ── demand bounce ─────────────────────────────────────────────────────────────

def _detect_demand_bounce(sn: dict) -> tuple:
    """
    DEMAND_BOUNCE: bullish reaction off a demand zone.
    """
    demand   = _get_demand(sn)
    bias     = _zone_bias(sn)
    price    = sn["current_price"]
    dir_prob = sn["dir_prob"]
    analog   = sn["analog"]

    conditions: list[str] = []
    missing:    list[str] = []

    # ── Condition 1: at / inside demand zone ──────────────────────────────
    inside_demand  = demand.get("inside_zone_flag", False)
    demand_dist    = _demand_dist(sn)
    near_demand    = (
        inside_demand
        or bias in ("AT_DEMAND", "DEMAND_BELOW")
        or (demand_dist is not None and demand_dist <= _CLOSE_TO_ZONE)
    )

    if inside_demand:
        conditions.append(f"price inside demand zone ({_fmt_price(demand.get('zone_low'))}–{_fmt_price(demand.get('zone_high'))})")
    elif near_demand:
        conditions.append(f"price near demand zone ({_pct(demand_dist)} away)")
    else:
        missing.append("price not near any demand zone")

    # ── Condition 2: bullish reaction forming ─────────────────────────────
    bullish_reaction = (
        _strong_lower_wick(sn)
        or (_bullish_candle(sn) and _closed_high_in_range(sn))
        or sn.get("vwap_reclaim_flag") == 1
    )
    if bullish_reaction:
        if _strong_lower_wick(sn):
            conditions.append("strong lower wick / hammer structure")
        elif _bullish_candle(sn):
            conditions.append("bullish bar closing near highs")
        else:
            conditions.append("VWAP reclaim on this bar")
    else:
        missing.append("no clear bullish reaction candle yet")

    # ── Condition 3: model not bearish ────────────────────────────────────
    if dir_prob >= _MODEL_BULLISH_SOFT:
        conditions.append(f"model not bearish (P(up)={dir_prob:.2f})")
    elif dir_prob >= _MODEL_BEARISH_HARD:
        missing.append(f"model is weakly bearish (P(up)={dir_prob:.2f})")
    else:
        missing.append(f"model is clearly bearish (P(up)={dir_prob:.2f})")

    # ── Condition 4: analog does not strongly favor breakdown ─────────────
    if _analog_active(sn):
        rej_rate = float(analog.get("rejection_rate", 0.5))
        brk_rate = float(analog.get("breakout_rate",  0.25))
        # For demand bounce: "rejection" of demand = breakdown; "breakout" = bounce
        if brk_rate >= _ANALOG_BREAKOUT_BIAS:
            conditions.append(f"analog favors bounce (breakout_rate={brk_rate:.0%})")
        elif rej_rate >= _ANALOG_STRONG_REJ:
            missing.append(f"analog strongly favors breakdown (rejection_rate={rej_rate:.0%})")
        else:
            conditions.append(f"analog neutral ({analog.get('n_matches',0)} matches)")
    else:
        conditions.append("analog unavailable (not penalised)")

    # ── Condition 5: chop not strongly flagging against trade ─────────────
    if sn["chop_detected"] and sn["chop_score"] >= 0.7:
        missing.append(f"chop detector warning (score={sn['chop_score']:.2f})")
    elif sn["chop_detected"]:
        missing.append(f"mild chop detected (score={sn['chop_score']:.2f})")

    # ── Invalidation ──────────────────────────────────────────────────────
    invalidation = None
    dz_low = demand.get("zone_low")
    if dz_low is not None:
        invalidation = round(float(dz_low) - (price * 0.001), 2)   # 0.1% below zone low

    # ── Target zone ───────────────────────────────────────────────────────
    target = None
    supply = _get_supply(sn)
    if supply.get("zone_low") is not None:
        target = (float(supply["zone_low"]), float(supply.get("zone_high", supply["zone_low"] * 1.002)))

    guidance = (
        "Watch for clean bounce with volume above demand zone. "
        "Look for consecutive bullish bars or VWAP reclaim before committing."
    )

    # Primary condition: must be near demand to be a valid DEMAND_BOUNCE
    valid = near_demand and (len(missing) - (0 if sn["chop_detected"] else 0)) <= 2

    return (
        conditions, missing, invalidation, target, guidance,
        SetupDirection.LONG, valid
    )


# ── supply rejection ──────────────────────────────────────────────────────────

def _detect_supply_rejection(sn: dict) -> tuple:
    """
    SUPPLY_REJECTION: bearish stall / wick off a supply zone.
    """
    supply   = _get_supply(sn)
    bias     = _zone_bias(sn)
    price    = sn["current_price"]
    dir_prob = sn["dir_prob"]
    analog   = sn["analog"]

    conditions: list[str] = []
    missing:    list[str] = []

    # ── Condition 1: at / inside supply zone ──────────────────────────────
    inside_supply = supply.get("inside_zone_flag", False)
    sup_dist      = _supply_dist(sn)
    near_supply   = (
        inside_supply
        or bias in ("AT_SUPPLY", "SUPPLY_OVERHEAD")
        or (sup_dist is not None and sup_dist <= _CLOSE_TO_ZONE)
    )

    if inside_supply:
        conditions.append(f"price inside supply zone ({_fmt_price(supply.get('zone_low'))}–{_fmt_price(supply.get('zone_high'))})")
    elif near_supply:
        conditions.append(f"price approaching supply zone ({_pct(sup_dist)} away)")
    else:
        missing.append("price not near any supply zone")

    # ── Condition 2: rejection / stall / wick ─────────────────────────────
    rejection_candle = (
        _strong_upper_wick(sn)
        or (_bearish_candle(sn) and _closed_low_in_range(sn))
        or sn.get("vwap_loss_flag") == 1
    )
    if rejection_candle:
        if _strong_upper_wick(sn):
            conditions.append("strong upper wick / shooting-star rejection")
        elif _bearish_candle(sn):
            conditions.append("bearish bar closing near lows")
        else:
            conditions.append("VWAP loss on this bar")
    else:
        missing.append("no clear rejection candle yet (watch for wick / bearish close)")

    # ── Condition 3: model not strongly bullish ───────────────────────────
    if dir_prob <= _MODEL_BEARISH_SOFT:
        conditions.append(f"model supports short (P(up)={dir_prob:.2f})")
    elif dir_prob <= _MODEL_BULLISH_HARD:
        missing.append(f"model is weakly bullish (P(up)={dir_prob:.2f})")
    else:
        missing.append(f"model is strongly bullish (P(up)={dir_prob:.2f})")

    # ── Condition 4: analog supports rejection ────────────────────────────
    if _analog_active(sn):
        rej_rate = float(analog.get("rejection_rate", 0.5))
        brk_rate = float(analog.get("breakout_rate",  0.25))
        if rej_rate >= _ANALOG_REJECTION_BIAS:
            conditions.append(f"analog favors rejection (rejection_rate={rej_rate:.0%})")
        elif brk_rate >= _ANALOG_STRONG_BRK:
            missing.append(f"analog favors breakout (breakout_rate={brk_rate:.0%})")
        else:
            conditions.append(f"analog neutral ({analog.get('n_matches',0)} matches)")
    else:
        conditions.append("analog unavailable (not penalised)")

    # ── Condition 5: structure not in strong uptrend ──────────────────────
    if _uptrend(sn):
        slope = sn.get("slope_ema_20_5")
        missing.append(f"structure in uptrend (slope_ema_20={slope:.5f})" if slope else "structure in uptrend")
    else:
        conditions.append("no strong uptrend opposing the rejection")

    # ── Chop ──────────────────────────────────────────────────────────────
    if sn["chop_detected"] and sn["chop_score"] >= 0.7:
        missing.append(f"chop detector warning (score={sn['chop_score']:.2f})")

    # ── Invalidation ──────────────────────────────────────────────────────
    invalidation = None
    sz_high = supply.get("zone_high")
    if sz_high is not None:
        invalidation = round(float(sz_high) + (price * 0.001), 2)

    # ── Target zone ───────────────────────────────────────────────────────
    target = None
    demand = _get_demand(sn)
    if demand.get("zone_high") is not None:
        target = (float(demand.get("zone_low", demand["zone_high"] * 0.998)), float(demand["zone_high"]))

    guidance = (
        "Wait for clean bearish candle closing below supply zone low. "
        "Volume expansion on the rejection bar increases conviction."
    )

    valid = near_supply and (len(missing)) <= 2

    return (
        conditions, missing, invalidation, target, guidance,
        SetupDirection.SHORT, valid
    )


# ── breakout above supply ─────────────────────────────────────────────────────

def _detect_breakout_above_supply(sn: dict) -> tuple:
    """
    BREAKOUT_ABOVE_SUPPLY: price pushes through supply and holds.
    """
    supply   = _get_supply(sn)
    price    = sn["current_price"]
    dir_prob = sn["dir_prob"]
    analog   = sn["analog"]

    conditions: list[str] = []
    missing:    list[str] = []

    # ── Condition 1: price above supply zone high ─────────────────────────
    sz_high = supply.get("zone_high")
    sz_low  = supply.get("zone_low")
    if sz_high is not None and price > float(sz_high):
        conditions.append(f"price broke above supply zone high ({_fmt_price(sz_high)})")
        # Check hold: closed above, not just wicked
        if sn["current_price"] >= float(sz_high):
            conditions.append("close holding above former supply high")
        else:
            missing.append("close has not held above supply zone high yet")
    elif sz_low is not None and price > float(sz_low):
        conditions.append(f"price inside supply zone (testing breakout level)")
        missing.append("needs clean close above supply zone high")
    else:
        missing.append("no supply zone breakout detected")

    # ── Condition 2: forecast model supports upside ───────────────────────
    if dir_prob >= _MODEL_BULLISH_HARD:
        conditions.append(f"model supports upside (P(up)={dir_prob:.2f})")
    elif dir_prob >= _MODEL_BULLISH_SOFT:
        missing.append(f"model is neutral/weak bullish (P(up)={dir_prob:.2f})")
    else:
        missing.append(f"model does not support upside (P(up)={dir_prob:.2f})")

    # ── Condition 3: analog does not strongly favor rejection ─────────────
    if _analog_active(sn):
        rej_rate = float(analog.get("rejection_rate", 0.5))
        brk_rate = float(analog.get("breakout_rate",  0.25))
        if brk_rate >= _ANALOG_BREAKOUT_BIAS:
            conditions.append(f"analog favors continuation (breakout_rate={brk_rate:.0%})")
        elif rej_rate >= _ANALOG_STRONG_REJ:
            missing.append(f"analog strongly favors rejection (rejection_rate={rej_rate:.0%})")
        else:
            conditions.append(f"analog neutral or mixed")
    else:
        conditions.append("analog unavailable")

    # ── Condition 4: range forecast supports meaningful move ──────────────
    if sn["pred_range"] >= _RANGE_LOW_CUTOFF * 2:
        conditions.append(f"range forecast supports move ({sn['pred_range']*100:.2f}%)")
    else:
        missing.append(f"range forecast is low ({sn['pred_range']*100:.2f}%) – may not have fuel")

    # ── Condition 5: bullish close structure ──────────────────────────────
    if _bullish_candle(sn) and _closed_high_in_range(sn):
        conditions.append("breakout bar is bullish with high close location")
    else:
        missing.append("breakout bar structure is not clearly bullish yet")

    # ── Chop ──────────────────────────────────────────────────────────────
    if sn["chop_detected"] and sn["chop_score"] >= 0.7:
        missing.append("chop detected – breakout risk of false break")

    # ── Invalidation ──────────────────────────────────────────────────────
    invalidation = None
    if sz_low is not None:
        invalidation = round(float(sz_low) - (price * 0.001), 2)

    # ── Target zone ───────────────────────────────────────────────────────
    target = None
    if sz_high is not None:
        # target = extension above supply high
        target = (float(sz_high), round(float(sz_high) * 1.005, 2))

    guidance = (
        "Look for a retest of former supply (now support) as a lower-risk entry. "
        "Volume on the breakout bar matters — thin volume = false break risk."
    )

    valid = (sz_high is not None and price >= float(sz_high) * 0.998)

    return (
        conditions, missing, invalidation, target, guidance,
        SetupDirection.LONG, valid
    )


# ── breakdown below demand ────────────────────────────────────────────────────

def _detect_breakdown_below_demand(sn: dict) -> tuple:
    """
    BREAKDOWN_BELOW_DEMAND: price breaks below demand and fails to recover.
    """
    demand   = _get_demand(sn)
    price    = sn["current_price"]
    dir_prob = sn["dir_prob"]
    analog   = sn["analog"]

    conditions: list[str] = []
    missing:    list[str] = []

    # ── Condition 1: price below demand zone low ──────────────────────────
    dz_low  = demand.get("zone_low")
    dz_high = demand.get("zone_high")
    if dz_low is not None and price < float(dz_low):
        conditions.append(f"price broke below demand zone low ({_fmt_price(dz_low)})")
        if sn["current_price"] <= float(dz_low):
            conditions.append("close confirming break below demand")
        else:
            missing.append("close has not confirmed break below demand")
    elif dz_high is not None and price < float(dz_high):
        conditions.append("price inside demand zone (testing breakdown level)")
        missing.append("needs close below demand zone low to confirm breakdown")
    else:
        missing.append("no demand zone breakdown detected")

    # ── Condition 2: model not supportive of longs ────────────────────────
    if dir_prob <= _MODEL_BEARISH_HARD:
        conditions.append(f"model supports downside (P(up)={dir_prob:.2f})")
    elif dir_prob <= _MODEL_BEARISH_SOFT:
        missing.append(f"model is weakly bearish (P(up)={dir_prob:.2f})")
    else:
        missing.append(f"model is bullish — conflicts with breakdown (P(up)={dir_prob:.2f})")

    # ── Condition 3: analog supports breakdown or weak bounce ─────────────
    if _analog_active(sn):
        rej_rate = float(analog.get("rejection_rate", 0.5))
        brk_rate = float(analog.get("breakout_rate",  0.25))
        if rej_rate >= _ANALOG_REJECTION_BIAS:
            conditions.append(f"analog supports breakdown (rejection_rate={rej_rate:.0%})")
        elif brk_rate >= _ANALOG_STRONG_BRK:
            missing.append(f"analog favors bounce/recovery (breakout_rate={brk_rate:.0%})")
        else:
            conditions.append(f"analog neutral ({analog.get('n_matches',0)} matches)")
    else:
        conditions.append("analog unavailable")

    # ── Condition 4: bearish structure ────────────────────────────────────
    if _bearish_candle(sn) and _closed_low_in_range(sn):
        conditions.append("breakdown bar is bearish with low close location")
    elif _bearish_candle(sn):
        conditions.append("bearish bar on breakdown")
    else:
        missing.append("breakdown bar is not clearly bearish")

    # ── Condition 5: lower-high structure (downtrend context) ─────────────
    if _downtrend(sn):
        conditions.append("price action in downtrend context")
    else:
        missing.append("no clear downtrend context — isolated breakdown risk")

    # ── Chop ──────────────────────────────────────────────────────────────
    if sn["chop_detected"] and sn["chop_score"] >= 0.7:
        missing.append("chop detected — breakdown may be a fake flush")

    # ── Invalidation ──────────────────────────────────────────────────────
    invalidation = None
    if dz_high is not None:
        invalidation = round(float(dz_high) + (price * 0.001), 2)

    # ── Target zone ───────────────────────────────────────────────────────
    target = None
    # next demand is below — approximate using recent low extension
    if dz_low is not None:
        target = (round(float(dz_low) * 0.995, 2), float(dz_low))

    guidance = (
        "Watch for back-test of former demand (now resistance) as entry. "
        "A bouncing wick that stalls at the broken zone confirms the breakdown."
    )

    valid = (dz_low is not None and price <= float(dz_low) * 1.002)

    return (
        conditions, missing, invalidation, target, guidance,
        SetupDirection.SHORT, valid
    )


# ── trend pullback continuation ───────────────────────────────────────────────

def _detect_trend_pullback(sn: dict) -> tuple:
    """
    TREND_PULLBACK_CONTINUATION: pullback into VWAP/EMA in a trending context.
    """
    dir_prob = sn["dir_prob"]
    analog   = sn["analog"]

    conditions: list[str] = []
    missing:    list[str] = []

    is_uptrend   = _uptrend(sn)
    is_downtrend = _downtrend(sn)
    trend_defined = is_uptrend or is_downtrend
    direction = SetupDirection.LONG if is_uptrend else SetupDirection.SHORT

    # ── Condition 1: trend defined ────────────────────────────────────────
    if is_uptrend:
        slope = sn.get("slope_ema_20_5")
        conditions.append(f"uptrend defined (slope_ema_20={slope:.5f})" if slope else "uptrend defined")
    elif is_downtrend:
        slope = sn.get("slope_ema_20_5")
        conditions.append(f"downtrend defined (slope_ema_20={slope:.5f})" if slope else "downtrend defined")
    else:
        missing.append("no clear trend — price is in a range or consolidating")

    # ── Condition 2: pullback into VWAP/EMA ──────────────────────────────
    near_vwap  = _near_vwap(sn)
    near_ema20 = _near_ema20(sn)
    pullback_to_structure = near_vwap or near_ema20

    if near_vwap:
        conditions.append(f"price near VWAP (dist={_pct(sn.get('dist_vwap'))})")
    if near_ema20:
        conditions.append(f"price near EMA-20 (dist={_pct(sn.get('dist_ema_20'))})")
    if not pullback_to_structure:
        missing.append("price not near VWAP or EMA-20 — pullback not to key structure")

    # ── Condition 3: continuation structure forming ───────────────────────
    if is_uptrend:
        continuation = _bullish_candle(sn) or sn.get("vwap_reclaim_flag") == 1
        if continuation:
            conditions.append("bullish continuation structure forming at pullback level")
        else:
            missing.append("no bullish continuation candle at pullback level yet")
    elif is_downtrend:
        continuation = _bearish_candle(sn) or sn.get("vwap_loss_flag") == 1
        if continuation:
            conditions.append("bearish continuation structure forming at pullback level")
        else:
            missing.append("no bearish continuation candle at pullback level yet")
    else:
        missing.append("trend unclear — continuation direction undetermined")

    # ── Condition 4: model aligned with trend ────────────────────────────
    if is_uptrend and dir_prob >= _MODEL_BULLISH_SOFT:
        conditions.append(f"model supports upside in uptrend (P(up)={dir_prob:.2f})")
    elif is_downtrend and dir_prob <= _MODEL_BEARISH_SOFT:
        conditions.append(f"model supports downside in downtrend (P(up)={dir_prob:.2f})")
    elif is_uptrend:
        missing.append(f"model not supporting upside continuation (P(up)={dir_prob:.2f})")
    elif is_downtrend:
        missing.append(f"model not supporting downside continuation (P(up)={dir_prob:.2f})")

    # ── Condition 5: chop detector not flagging chop ──────────────────────
    if sn["chop_detected"] and sn["chop_score"] >= 0.6:
        missing.append(f"chop detected — pullback may be noise not trend ({sn['chop_score']:.2f})")
    elif not trend_defined:
        missing.append("chop/range likely — no trend to continue")

    # ── Condition 6: analog (optional) ───────────────────────────────────
    if _analog_active(sn):
        rej_rate = float(analog.get("rejection_rate", 0.5))
        brk_rate = float(analog.get("breakout_rate",  0.25))
        if is_uptrend and brk_rate >= _ANALOG_BREAKOUT_BIAS:
            conditions.append(f"analog supports continuation (breakout_rate={brk_rate:.0%})")
        elif is_downtrend and rej_rate >= _ANALOG_REJECTION_BIAS:
            conditions.append(f"analog supports continuation (rejection_rate={rej_rate:.0%})")
        else:
            conditions.append("analog mixed")

    # ── Invalidation ──────────────────────────────────────────────────────
    price       = sn["current_price"]
    dist_ema20  = sn.get("dist_ema_20")
    invalidation = None
    if dist_ema20 is not None:
        ema20_price = price / (1.0 + float(dist_ema20))
        # Invalidation for uptrend: close below EMA-20
        if is_uptrend:
            invalidation = round(ema20_price * 0.999, 2)
        else:
            invalidation = round(ema20_price * 1.001, 2)

    # ── Target zone ───────────────────────────────────────────────────────
    target = None
    supply = _get_supply(sn)
    demand = _get_demand(sn)
    if is_uptrend and supply.get("zone_low") is not None:
        target = (float(supply["zone_low"]), float(supply.get("zone_high", supply["zone_low"] * 1.002)))
    elif is_downtrend and demand.get("zone_high") is not None:
        target = (float(demand.get("zone_low", demand["zone_high"] * 0.998)), float(demand["zone_high"]))

    trend_word = "uptrend" if is_uptrend else "downtrend" if is_downtrend else "range"
    guidance = (
        f"In {trend_word} context: wait for confirmation candle at VWAP/EMA before entry. "
        "High-probability setups show clean pullbacks with contracting volume."
    )

    valid = trend_defined and pullback_to_structure

    return (
        conditions, missing, invalidation, target, guidance,
        direction, valid
    )


# ── no-setup fallback ─────────────────────────────────────────────────────────

def _detect_no_setup(sn: dict) -> tuple:
    bias     = _zone_bias(sn)
    dir_prob = sn["dir_prob"]

    notes = [f"zone bias: {bias}", f"P(up)={dir_prob:.2f}"]
    if sn["chop_detected"]:
        notes.append(f"chop detected (score={sn['chop_score']:.2f})")

    guidance = (
        "No high-confidence setup present. "
        "Consider waiting for price to reach a key zone or for a clearer trend to form."
    )
    return (
        notes,                  # conditions (just descriptive)
        ["no qualifying setup conditions met"],
        None, None, guidance,
        SetupDirection.NEUTRAL, True
    )


# ── public entry point ────────────────────────────────────────────────────────

# Map each setup to its detector
_DETECTORS = {
    SetupType.DEMAND_BOUNCE:               _detect_demand_bounce,
    SetupType.SUPPLY_REJECTION:            _detect_supply_rejection,
    SetupType.BREAKOUT_ABOVE_SUPPLY:       _detect_breakout_above_supply,
    SetupType.BREAKDOWN_BELOW_DEMAND:      _detect_breakdown_below_demand,
    SetupType.TREND_PULLBACK_CONTINUATION: _detect_trend_pullback,
    SetupType.NO_SETUP:                    _detect_no_setup,
}


def detect_setups(snapshot: dict) -> dict[SetupType, dict]:
    """
    Run all detectors against the snapshot and return a per-setup detection dict.

    Returns
    ───────
    dict mapping SetupType → {
        "conditions_met":     list[str]
        "missing_confirms":   list[str]
        "invalidation_level": float | None
        "target_zone":        tuple | None
        "guidance":           str
        "direction":          SetupDirection
        "valid":              bool   – True if primary conditions are met
    }
    """
    results: dict[SetupType, dict] = {}

    for setup_type, detector in _DETECTORS.items():
        try:
            conds, missing, inv, target, guidance, direction, valid = detector(snapshot)
        except Exception as exc:
            logger.warning("Detector %s failed: %s", setup_type.value, exc)
            conds, missing, inv, target, guidance = ["detector error"], [str(exc)], None, None, ""
            direction, valid = SetupDirection.NEUTRAL, False

        results[setup_type] = {
            "conditions_met":     conds,
            "missing_confirms":   missing,
            "invalidation_level": inv,
            "target_zone":        target,
            "guidance":           guidance,
            "direction":          direction,
            "valid":              valid,
        }

    return results
