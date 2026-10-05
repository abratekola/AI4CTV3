from __future__ import annotations
from datetime import datetime
from typing import Iterable


def clamp(x, lo=0.0, hi=100.0):
    return max(lo, min(hi, float(x)))


def _slope(points):
    """Least-squares slope in risk points/day for [(date, value), ...]."""
    if len(points) < 2:
        return 0.0
    xs=[]; ys=[]
    d0=min(p[0] for p in points)
    for d,v in points:
        xs.append((d-d0).days)
        ys.append(float(v))
    if len(set(xs)) < 2:
        return 0.0
    xbar=sum(xs)/len(xs); ybar=sum(ys)/len(ys)
    den=sum((x-xbar)**2 for x in xs)
    return sum((x-xbar)*(y-ybar) for x,y in zip(xs,ys))/den if den else 0.0


def forecast_risk(metric_rows: Iterable, horizons=(7,14,30)):
    rows=list(metric_rows)
    if not rows:
        return {h:0.0 for h in horizons}, 0.0, "insufficient data"
    pts=[]
    for r in rows[:8]:
        try:
            pts.append((datetime.fromisoformat(str(r["report_date"])), float(r["overall_score"])))
        except Exception:
            pass
    pts=sorted(pts, key=lambda x:x[0])
    latest=pts[-1][1] if pts else 0.0
    slope=_slope(pts)
    # Cap extrapolation so a short noisy series cannot create absurd forecasts.
    slope=max(-2.5, min(2.5, slope))
    forecasts={h:round(clamp(latest+slope*h),1) for h in horizons}
    if len(pts)<3:
        confidence="low — fewer than 3 reporting periods"
    elif abs(slope)<0.1:
        confidence="stable trend"
    else:
        confidence="trend-based"
    return forecasts, round(slope,3), confidence


def forecast_recruitment(history_data, target_total: float):
    """Simple transparent enrollment forecast from recent 30-day rates."""
    if not history_data:
        return {"rate_per_day":0,"days_to_target":None,"projected_date":None,"status":"No data"}
    latest=history_data[0]
    enrolled=float(latest.get("enrolled_total",0) or 0)
    recent=[]
    for d in history_data[:4]:
        r=float(d.get("enrollment_30d",0) or 0)/30.0
        if r>=0: recent.append(r)
    rate=sum(recent)/len(recent) if recent else 0
    remaining=max(0,float(target_total or 0)-enrolled)
    days=remaining/rate if rate>0 and target_total else None
    return {
        "rate_per_day":round(rate,2),
        "days_to_target":round(days) if days is not None else None,
        "projected_date":None,
        "status":"On target" if remaining==0 else ("Forecast available" if days is not None else "Insufficient recruitment rate")
    }
