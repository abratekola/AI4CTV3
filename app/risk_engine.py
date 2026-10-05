from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List, Tuple


def clamp(x, lo=0.0, hi=100.0):
    return max(lo, min(hi, float(x)))


def sev(score: float) -> str:
    if score >= 70: return "Critical"
    if score >= 50: return "High"
    if score >= 30: return "Medium"
    return "Low"


@dataclass
class RiskResult:
    overall_score: float
    severity: str
    domains: dict
    alerts: list
    brief: str
    rule_signals: list


def recruitment_risk(enrollment_30d: float, target_30d: float, screen_fail_pct: float) -> float:
    if target_30d <= 0:
        achievement = 100
    else:
        achievement = enrollment_30d / target_30d * 100
    if achievement >= 100: performance = 0
    elif achievement >= 85: performance = 20
    elif achievement >= 70: performance = 45
    elif achievement >= 50: performance = 70
    else: performance = 95
    sf = clamp((screen_fail_pct - 15) * 2.0) if screen_fail_pct > 15 else 0
    return clamp(0.75 * performance + 0.25 * sf)


def retention_risk(retention_pct: float, missed_visit_pct: float) -> float:
    if retention_pct >= 97: ret = 0
    elif retention_pct >= 95: ret = 20
    elif retention_pct >= 90: ret = 50
    elif retention_pct >= 85: ret = 75
    else: ret = 95
    missed = clamp(missed_visit_pct * 7)
    return clamp(0.7 * ret + 0.3 * missed)


def protocol_risk(deviations_30d: float, major_deviations_30d: float, enrolled_total: float) -> float:
    denom=max(enrolled_total,10)
    rate100=deviations_30d/denom*100
    major_rate100=major_deviations_30d/denom*100
    return clamp(rate100*8 + major_rate100*20)


def data_risk(overdue_crfs: float, open_queries: float, query_age_days: float) -> float:
    crf=clamp(overdue_crfs*3.5); q=clamp(open_queries*1.5)
    age=clamp((query_age_days-7)*3.5) if query_age_days>7 else 0
    return clamp(0.4*crf + 0.35*q + 0.25*age)


def safety_risk(overdue_sae: float, unreviewed_aes: float, safety_training_overdue: float) -> float:
    sae=100 if overdue_sae>=2 else (80 if overdue_sae>=1 else 0)
    ae=clamp(unreviewed_aes*8); tr=clamp(safety_training_overdue*20)
    return clamp(0.65*sae + 0.2*ae + 0.15*tr)


def supply_risk(ip_days_stock: float, stockout_events_90d: float, temp_excursions_30d: float) -> float:
    if ip_days_stock<=7: stock=100
    elif ip_days_stock<=14: stock=80
    elif ip_days_stock<=30: stock=55
    elif ip_days_stock<=45: stock=25
    else: stock=0
    return clamp(0.7*stock + 0.2*clamp(stockout_events_90d*40) + 0.1*clamp(temp_excursions_30d*30))


def lab_risk(lab_tat_hours: float, rejected_samples_pct: float, critical_result_overdue: float) -> float:
    if lab_tat_hours>72: tat=90
    elif lab_tat_hours>48: tat=65
    elif lab_tat_hours>24: tat=35
    else: tat=0
    reject=clamp(rejected_samples_pct*10); critical=clamp(critical_result_overdue*35)
    return clamp(0.55*tat + 0.25*reject + 0.2*critical)


def staffing_risk(vacancy_pct: float, training_overdue_count: float, turnover_90d: float) -> float:
    return clamp(0.55*clamp(vacancy_pct*4) + 0.25*clamp(training_overdue_count*15) + 0.2*clamp(turnover_90d*20))


def monitoring_risk(overdue_actions: float, days_since_monitoring: float) -> float:
    actions=clamp(overdue_actions*12)
    if days_since_monitoring>120: interval=85
    elif days_since_monitoring>90: interval=60
    elif days_since_monitoring>60: interval=30
    else: interval=0
    return clamp(0.65*actions + 0.35*interval)


def score_rule(value: float, rule: dict) -> float:
    direction=rule.get("direction","high_bad")
    amber=float(rule.get("amber",0)); red=float(rule.get("red",0)); critical=float(rule.get("critical",0))
    v=float(value or 0)
    if direction=="low_bad":
        if v<=critical: return 100
        if v<=red: return 75
        if v<=amber: return 45
        return 0
    if v>=critical: return 100
    if v>=red: return 75
    if v>=amber: return 45
    return 0


def evaluate(metrics: dict, previous: Optional[dict]=None, custom_rules: Optional[list]=None) -> RiskResult:
    d={
        "Recruitment": recruitment_risk(metrics.get("enrollment_30d",0),metrics.get("enrollment_target_30d",0),metrics.get("screen_fail_pct",0)),
        "Retention": retention_risk(metrics.get("retention_pct",100),metrics.get("missed_visit_pct",0)),
        "Protocol": protocol_risk(metrics.get("deviations_30d",0),metrics.get("major_deviations_30d",0),metrics.get("enrolled_total",0)),
        "Data quality": data_risk(metrics.get("overdue_crfs",0),metrics.get("open_queries",0),metrics.get("median_query_age_days",0)),
        "Safety": safety_risk(metrics.get("overdue_sae",0),metrics.get("unreviewed_aes",0),metrics.get("safety_training_overdue",0)),
        "Supply": supply_risk(metrics.get("ip_days_stock",999),metrics.get("stockout_events_90d",0),metrics.get("temp_excursions_30d",0)),
        "Laboratory": lab_risk(metrics.get("lab_tat_hours",0),metrics.get("rejected_samples_pct",0),metrics.get("critical_result_overdue",0)),
        "Staffing": staffing_risk(metrics.get("staff_vacancy_pct",0),metrics.get("training_overdue_count",0),metrics.get("turnover_90d",0)),
        "Monitoring": monitoring_risk(metrics.get("monitoring_actions_overdue",0),metrics.get("days_since_monitoring",0)),
    }
    rule_signals=[]
    domain_rule_scores={}
    for r in custom_rules or []:
        if not int(r.get("enabled",1)) or r.get("scope","site")!="site": continue
        metric=r.get("metric")
        if metric not in metrics: continue
        rs=score_rule(metrics.get(metric,0),r)
        if rs>0:
            rule_signals.append({"rule_id":r.get("id"),"rule_type":r.get("rule_type","KRI"),"domain":r.get("domain","Other"),"metric":metric,
                                 "value":metrics.get(metric,0),"score":rs,"severity":sev(rs),"label":r.get("label",metric),"action":r.get("action","")})
        domain_rule_scores.setdefault(r.get("domain","Other"),[]).append(rs)
    # Blend configurable protocol/KRI signals into transparent base domain score.
    for domain, scores in domain_rule_scores.items():
        if domain in d and scores:
            custom=max(scores)
            d[domain]=clamp(0.65*d[domain]+0.35*custom)

    weights={"Recruitment":0.15,"Retention":0.11,"Protocol":0.12,"Data quality":0.12,"Safety":0.16,"Supply":0.10,"Laboratory":0.08,"Staffing":0.08,"Monitoring":0.08}
    overall=sum(d[k]*weights[k] for k in d)
    trend_flags: List[Tuple[str,float]]=[]
    if previous:
        old=evaluate(previous,None,custom_rules).domains
        for k,v in d.items():
            delta=v-old.get(k,0)
            if delta>=20: trend_flags.append((k,delta))
        if trend_flags: overall=clamp(overall+min(12,max(x[1] for x in trend_flags)*0.2))

    action_map={
        "Recruitment":"Review screening logs and referral pathways; compare actual versus target enrollment and implement a site-specific recruitment recovery plan.",
        "Retention":"Review missed-visit cases, participant contact procedures and visit-window tracking; initiate targeted retention follow-up.",
        "Protocol":"Perform root-cause review of deviations, retrain affected staff and verify corrective/preventive actions.",
        "Data quality":"Prioritize overdue CRFs and aged queries; assign owners and deadlines and verify source-data completion.",
        "Safety":"Escalate immediately to the PI/safety lead; resolve overdue SAE reporting or unreviewed safety events and document actions.",
        "Supply":"Verify investigational-product inventory and resupply lead time; initiate replenishment and contingency planning before stockout.",
        "Laboratory":"Review sample workflow, transport, processing and critical-result communication; address recurrent bottlenecks.",
        "Staffing":"Review staffing gaps, delegated duties and training currency; arrange coverage and priority training.",
        "Monitoring":"Close overdue monitoring actions and reassess monitoring frequency according to current site risk.",
    }
    alerts=[]
    for domain,score in sorted(d.items(),key=lambda x:x[1],reverse=True):
        if score>=30:
            alerts.append({"domain":domain,"score":round(score,1),"severity":sev(score),"message":f"{domain} risk is {sev(score).lower()} ({score:.0f}/100).","action":action_map[domain]})
    for domain,delta in trend_flags:
        alerts.append({"domain":domain,"score":round(d[domain],1),"severity":"Trend","message":f"{domain} risk worsened by {delta:.0f} points since the previous reporting period.","action":action_map[domain]})
    # Specific configurable KRI signals are explicit and traceable.
    for s in rule_signals:
        if s["score"]>=45:
            alerts.append({"domain":s["domain"],"score":s["score"],"severity":s["severity"],
                           "message":f"{s['rule_type']} threshold triggered: {s['label']} = {s['value']}.",
                           "action":s["action"] or action_map.get(s["domain"],"Review the configured threshold and initiate documented risk mitigation.")})

    top=sorted(d.items(),key=lambda x:x[1],reverse=True)[:3]
    top_text=", ".join(f"{k} ({v:.0f})" for k,v in top)
    if overall>=70: intro="Immediate operational escalation is recommended."
    elif overall>=50: intro="The site requires prompt risk-mitigation review."
    elif overall>=30: intro="The site shows emerging operational risk that should be actively managed."
    else: intro="No major operational warning is currently detected; continue routine surveillance."
    trend_text=" Rapid deterioration was detected in "+", ".join(x[0] for x in trend_flags)+"." if trend_flags else ""
    brief=f"{intro} Overall risk is {overall:.0f}/100 ({sev(overall)}). Highest-risk domains: {top_text}.{trend_text}"
    return RiskResult(round(overall,1),sev(overall),{k:round(v,1) for k,v in d.items()},alerts,brief,rule_signals)
