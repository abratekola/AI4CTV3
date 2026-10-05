from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Tuple
import math
import numpy as np

from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

FEATURE_FIELDS = [
    "enrolled_total","enrollment_30d","enrollment_target_30d","screen_fail_pct",
    "retention_pct","missed_visit_pct","deviations_30d","major_deviations_30d",
    "overdue_crfs","open_queries","median_query_age_days","overdue_sae",
    "unreviewed_aes","safety_training_overdue","ip_days_stock","stockout_events_90d",
    "temp_excursions_30d","lab_tat_hours","rejected_samples_pct","critical_result_overdue",
    "staff_vacancy_pct","training_overdue_count","turnover_90d","monitoring_actions_overdue",
    "days_since_monitoring"
]

FRIENDLY = {
    "enrolled_total":"cumulative enrollment", "enrollment_30d":"30-day enrollment",
    "enrollment_target_30d":"30-day recruitment target", "screen_fail_pct":"screening-failure rate",
    "retention_pct":"participant retention", "missed_visit_pct":"missed-visit rate",
    "deviations_30d":"protocol deviations", "major_deviations_30d":"major protocol deviations",
    "overdue_crfs":"overdue CRFs", "open_queries":"open data queries",
    "median_query_age_days":"median query age", "overdue_sae":"overdue SAE reports",
    "unreviewed_aes":"unreviewed adverse events", "safety_training_overdue":"overdue safety training",
    "ip_days_stock":"investigational-product stock days", "stockout_events_90d":"recent stockout events",
    "temp_excursions_30d":"temperature excursions", "lab_tat_hours":"laboratory turnaround time",
    "rejected_samples_pct":"rejected-sample rate", "critical_result_overdue":"overdue critical results",
    "staff_vacancy_pct":"key-staff vacancy", "training_overdue_count":"overdue staff training",
    "turnover_90d":"recent staff turnover", "monitoring_actions_overdue":"overdue monitoring actions",
    "days_since_monitoring":"days since monitoring"
}

HIGH_BAD = {
    "screen_fail_pct","missed_visit_pct","deviations_30d","major_deviations_30d","overdue_crfs",
    "open_queries","median_query_age_days","overdue_sae","unreviewed_aes","safety_training_overdue",
    "stockout_events_90d","temp_excursions_30d","lab_tat_hours","rejected_samples_pct",
    "critical_result_overdue","staff_vacancy_pct","training_overdue_count","turnover_90d",
    "monitoring_actions_overdue","days_since_monitoring"
}
LOW_BAD = {"retention_pct","ip_days_stock","enrollment_30d"}


@dataclass
class ModelBundle:
    classifier: RandomForestClassifier
    anomaly: IsolationForest
    train_X: np.ndarray
    feature_names: List[str]
    medians: np.ndarray
    scales: np.ndarray
    mode: str
    n_samples: int
    positive_rate: float
    internal_auc: Optional[float]
    trained_at: str
    note: str


def _num(d: dict, k: str) -> float:
    try:
        v = d.get(k, 0)
        return float(v or 0)
    except Exception:
        return 0.0


def _trend_features(current: dict, previous: Optional[dict]) -> List[float]:
    if not previous:
        return [0.0] * 8
    keys = [
        "enrollment_30d","retention_pct","missed_visit_pct","overdue_crfs",
        "median_query_age_days","ip_days_stock","lab_tat_hours","staff_vacancy_pct"
    ]
    return [_num(current,k)-_num(previous,k) for k in keys]


def feature_names() -> List[str]:
    return FEATURE_FIELDS + [
        "trend_enrollment_30d","trend_retention_pct","trend_missed_visit_pct","trend_overdue_crfs",
        "trend_query_age_days","trend_ip_days_stock","trend_lab_tat_hours","trend_staff_vacancy_pct"
    ]


def vectorize(current: dict, previous: Optional[dict]=None) -> np.ndarray:
    return np.asarray([_num(current,k) for k in FEATURE_FIELDS] + _trend_features(current,previous), dtype=float)


def _synthetic_training(n: int=1800, seed: int=73021) -> Tuple[np.ndarray,np.ndarray]:
    """Generate transparent bootstrap training data for a non-validated demo model.

    The synthetic set exists only so AI functions can run before a trial has accumulated
    enough labelled longitudinal data. The UI clearly labels this mode as bootstrap/demo.
    """
    rng=np.random.default_rng(seed)
    X=[]; y=[]
    for _ in range(n):
        stress=float(rng.beta(1.6,2.2))
        trend=float(rng.normal(0,0.18))
        target=max(5, rng.normal(18,4))
        enrollment=max(0, target*(1.15-0.9*stress)+rng.normal(0,2.2))
        d={
            "enrolled_total":max(0,rng.normal(65,25)),
            "enrollment_30d":enrollment,
            "enrollment_target_30d":target,
            "screen_fail_pct":np.clip(10+34*stress+rng.normal(0,5),2,70),
            "retention_pct":np.clip(99-16*stress+rng.normal(0,2),65,100),
            "missed_visit_pct":np.clip(1+20*stress+rng.normal(0,3),0,45),
            "deviations_30d":max(0,rng.poisson(0.8+6*stress)),
            "major_deviations_30d":max(0,rng.poisson(0.05+1.6*stress)),
            "overdue_crfs":max(0,rng.poisson(1+20*stress)),
            "open_queries":max(0,rng.poisson(3+28*stress)),
            "median_query_age_days":np.clip(4+27*stress+rng.normal(0,4),1,60),
            "overdue_sae":rng.binomial(2,min(.45,0.03+0.35*stress)),
            "unreviewed_aes":max(0,rng.poisson(.4+5*stress)),
            "safety_training_overdue":max(0,rng.poisson(.1+1.8*stress)),
            "ip_days_stock":np.clip(65-58*stress+rng.normal(0,8),0,100),
            "stockout_events_90d":rng.binomial(2,min(.5,.02+.32*stress)),
            "temp_excursions_30d":rng.binomial(3,min(.45,.02+.25*stress)),
            "lab_tat_hours":np.clip(16+70*stress+rng.normal(0,9),4,160),
            "rejected_samples_pct":np.clip(.5+11*stress+rng.normal(0,1.5),0,30),
            "critical_result_overdue":max(0,rng.poisson(.1+2.7*stress)),
            "staff_vacancy_pct":np.clip(1+40*stress+rng.normal(0,6),0,80),
            "training_overdue_count":max(0,rng.poisson(.15+4*stress)),
            "turnover_90d":max(0,rng.poisson(.05+2.2*stress)),
            "monitoring_actions_overdue":max(0,rng.poisson(.3+7*stress)),
            "days_since_monitoring":np.clip(35+100*stress+rng.normal(0,15),5,220),
        }
        prev=dict(d)
        # Produce plausible recent direction of travel around the current state.
        prev["enrollment_30d"] = max(0, d["enrollment_30d"] - rng.normal(-trend*6,2))
        prev["retention_pct"] = np.clip(d["retention_pct"] - rng.normal(-trend*3,1),50,100)
        prev["missed_visit_pct"] = max(0, d["missed_visit_pct"] - rng.normal(trend*5,1.5))
        prev["overdue_crfs"] = max(0, d["overdue_crfs"] - rng.normal(trend*8,2))
        prev["median_query_age_days"] = max(0, d["median_query_age_days"] - rng.normal(trend*8,2))
        prev["ip_days_stock"] = max(0, d["ip_days_stock"] - rng.normal(-trend*12,3))
        prev["lab_tat_hours"] = max(0, d["lab_tat_hours"] - rng.normal(trend*15,4))
        prev["staff_vacancy_pct"] = max(0, d["staff_vacancy_pct"] - rng.normal(trend*10,3))
        x=vectorize(d,prev)
        # Non-linear latent probability of entering a high/critical operational state.
        achievement=d["enrollment_30d"]/max(d["enrollment_target_30d"],1)
        logit=(-4.0 + 4.8*stress + 1.0*max(0,0.75-achievement) +
               0.9*(d["overdue_sae"]>0) + 0.7*(d["ip_days_stock"]<14) +
               0.7*(d["major_deviations_30d"]>=2) + 0.65*max(0,trend))
        p=1/(1+math.exp(-logit))
        label=int(rng.random()<p)
        X.append(x); y.append(label)
    return np.asarray(X,dtype=float),np.asarray(y,dtype=int)


def local_training_samples(histories: Iterable[List[dict]]) -> Tuple[np.ndarray,np.ndarray]:
    X=[]; y=[]
    for hist in histories:
        if len(hist)<2: continue
        # hist must be chronological; each item contains data and next-state score/severity.
        for i in range(len(hist)-1):
            cur=hist[i]
            prev=hist[i-1] if i>0 else None
            nxt=hist[i+1]
            X.append(vectorize(cur["data"], prev["data"] if prev else None))
            y.append(1 if float(nxt.get("score",0))>=50 else 0)
    if not X: return np.empty((0,len(feature_names()))),np.empty((0,),dtype=int)
    return np.asarray(X,dtype=float),np.asarray(y,dtype=int)


def _fit(X: np.ndarray, y: np.ndarray, mode: str, note: str) -> ModelBundle:
    clf=RandomForestClassifier(
        n_estimators=240, max_depth=8, min_samples_leaf=3,
        class_weight="balanced_subsample", random_state=431, n_jobs=-1
    )
    auc=None
    if len(X)>=80 and len(np.unique(y))>1 and min(np.bincount(y))>=10:
        Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.25,random_state=431,stratify=y)
        clf.fit(Xtr,ytr)
        try: auc=float(roc_auc_score(yte,clf.predict_proba(Xte)[:,1]))
        except Exception: auc=None
        # Refit on all data after internal assessment.
        clf.fit(X,y)
    else:
        clf.fit(X,y)
    iso=IsolationForest(n_estimators=180,contamination=.08,random_state=431,n_jobs=-1)
    iso.fit(X)
    med=np.nanmedian(X,axis=0)
    q75=np.nanpercentile(X,75,axis=0); q25=np.nanpercentile(X,25,axis=0)
    scale=np.where((q75-q25)>1e-6,q75-q25,np.nanstd(X,axis=0)+1e-6)
    return ModelBundle(
        classifier=clf, anomaly=iso, train_X=X, feature_names=feature_names(),
        medians=med, scales=scale, mode=mode, n_samples=int(len(X)),
        positive_rate=float(np.mean(y)) if len(y) else 0.0, internal_auc=auc,
        trained_at=datetime.now().isoformat(timespec="seconds"), note=note
    )


def train_bundle(histories: Iterable[List[dict]], min_local_samples: int=40) -> ModelBundle:
    Xlocal,ylocal=local_training_samples(histories)
    enough=(len(Xlocal)>=min_local_samples and len(np.unique(ylocal))>1 and min(np.bincount(ylocal))>=8)
    if enough:
        return _fit(
            Xlocal,ylocal,"local-history",
            "Model trained on this installation's longitudinal site history. Internal discrimination is not external validation."
        )
    X,y=_synthetic_training()
    return _fit(
        X,y,"bootstrap-demo",
        "Model trained on synthetic operational scenarios because there are not yet enough labelled local transitions. Use for software evaluation only until locally validated."
    )


def _hazard_scale(p30: float, days: int) -> float:
    p30=min(max(float(p30),0.00001),0.99999)
    return 1-(1-p30)**(days/30.0)


def predict(bundle: ModelBundle, current: dict, previous: Optional[dict]=None) -> dict:
    x=vectorize(current,previous).reshape(1,-1)
    p30=float(bundle.classifier.predict_proba(x)[0,1])
    probs={7:round(100*_hazard_scale(p30,7),1),14:round(100*_hazard_scale(p30,14),1),30:round(100*p30,1)}

    # Percentile-like anomaly score relative to training data.
    train_raw=-bundle.anomaly.score_samples(bundle.train_X)
    raw=float(-bundle.anomaly.score_samples(x)[0])
    anomaly=float(np.mean(train_raw<=raw)*100)

    importance=getattr(bundle.classifier,"feature_importances_",np.ones(x.shape[1])/x.shape[1])
    z=(x[0]-bundle.medians)/(bundle.scales+1e-9)
    contrib=np.abs(z)*importance
    order=np.argsort(contrib)[::-1]
    drivers=[]
    for idx in order[:8]:
        if contrib[idx]<=0: continue
        name=bundle.feature_names[idx]
        if name.startswith("trend_"):
            label=name.replace("trend_","").replace("_"," ")+" trend"
            direction="worsening" if z[idx]>0 else "improving"
        else:
            label=FRIENDLY.get(name,name.replace("_"," "))
            val=x[0,idx]
            if name in HIGH_BAD: direction="higher than learned baseline" if z[idx]>0 else "lower than learned baseline"
            elif name in LOW_BAD: direction="lower than learned baseline" if z[idx]<0 else "higher than learned baseline"
            else: direction="different from learned baseline"
        drivers.append({
            "feature":name,"label":label,"direction":direction,
            "value":round(float(x[0,idx]),2),"importance":round(float(importance[idx]),4),
            "driver_score":round(float(contrib[idx]),3)
        })
    risk_band="Critical" if probs[30]>=75 else "High" if probs[30]>=50 else "Medium" if probs[30]>=25 else "Low"
    anomaly_band="Very unusual" if anomaly>=97 else "Unusual" if anomaly>=90 else "Watch" if anomaly>=75 else "Typical"
    return {
        "p7":probs[7],"p14":probs[14],"p30":probs[30],"risk_band":risk_band,
        "anomaly_score":round(anomaly,1),"anomaly_band":anomaly_band,
        "drivers":drivers,"model_mode":bundle.mode,"trained_at":bundle.trained_at,
        "n_samples":bundle.n_samples,"internal_auc":round(bundle.internal_auc,3) if bundle.internal_auc is not None else None,
        "model_note":bundle.note
    }


def feature_importance(bundle: ModelBundle, top_n: int=15) -> List[dict]:
    imp=getattr(bundle.classifier,"feature_importances_",np.zeros(len(bundle.feature_names)))
    order=np.argsort(imp)[::-1][:top_n]
    out=[]
    for i in order:
        name=bundle.feature_names[i]
        out.append({"feature":name,"label":FRIENDLY.get(name,name.replace("trend_","").replace("_"," ")),"importance":round(float(imp[i])*100,2)})
    return out
