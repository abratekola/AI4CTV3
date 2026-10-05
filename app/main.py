from __future__ import annotations
import csv, io, json, os, re, shutil, sqlite3
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Optional

import httpx
from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .risk_engine import evaluate, score_rule, sev
from .predictive import forecast_risk, forecast_recruitment
from .protocol_parser import extract_pdf_pages, suggest_operational_rules
from .ml_engine import train_bundle, predict as ml_predict, feature_importance
from .ai_copilot import load_settings as load_ai_settings, save_settings as save_ai_settings, provider_label, analyze as copilot_analyze

BASE=Path(__file__).resolve().parent.parent
DB_PATH=BASE/"trial_ops.sqlite3"
PROTOCOL_DIR=BASE/"data"/"protocols"
PROTOCOL_DIR.mkdir(parents=True,exist_ok=True)
AI_SETTINGS_PATH=BASE/"ai_settings.local.json"
MODEL_CACHE={}

app=FastAPI(title="AI4CT — Clinical Trial Operations Early-Warning AI",version="0.3.0")
app.mount("/static",StaticFiles(directory=BASE/"app"/"static"),name="static")
templates=Jinja2Templates(directory=BASE/"app"/"templates")

FIELDS=[
 "enrolled_total","enrollment_30d","enrollment_target_30d","screen_fail_pct",
 "retention_pct","missed_visit_pct","deviations_30d","major_deviations_30d",
 "overdue_crfs","open_queries","median_query_age_days","overdue_sae",
 "unreviewed_aes","safety_training_overdue","ip_days_stock","stockout_events_90d",
 "temp_excursions_30d","lab_tat_hours","rejected_samples_pct","critical_result_overdue",
 "staff_vacancy_pct","training_overdue_count","turnover_90d","monitoring_actions_overdue",
 "days_since_monitoring"
]

ROLE_DOMAINS={
 "All roles":None,
 "Trial Manager":None,
 "Principal Investigator":{"Recruitment","Retention","Protocol","Safety","Staffing"},
 "CRA / Monitor":{"Protocol","Data quality","Monitoring","Staffing"},
 "Data Manager":{"Data quality"},
 "Safety Lead":{"Safety"},
 "Pharmacy / IP Lead":{"Supply"},
 "Laboratory Lead":{"Laboratory"},
}


def conn():
    c=sqlite3.connect(DB_PATH); c.row_factory=sqlite3.Row; return c

def now(): return datetime.now().isoformat(timespec="seconds")

def _cols(c,table): return {r["name"] for r in c.execute(f"PRAGMA table_info({table})").fetchall()}

def _ensure_col(c,table,col,definition):
    if col not in _cols(c,table): c.execute(f"ALTER TABLE {table} ADD COLUMN {col} {definition}")


def init_db():
    c=conn()
    c.executescript("""
    CREATE TABLE IF NOT EXISTS trials(
      id INTEGER PRIMARY KEY, code TEXT UNIQUE, title TEXT, phase TEXT, sponsor TEXT,
      start_date TEXT, planned_end_date TEXT, created_at TEXT
    );
    CREATE TABLE IF NOT EXISTS sites(
      id INTEGER PRIMARY KEY, trial_id INTEGER, code TEXT, name TEXT, country TEXT, city TEXT,
      pi_name TEXT, status TEXT DEFAULT 'Active', created_at TEXT, UNIQUE(trial_id,code)
    );
    CREATE TABLE IF NOT EXISTS metrics(
      id INTEGER PRIMARY KEY, site_id INTEGER, report_date TEXT, data_json TEXT,
      overall_score REAL, severity TEXT, domains_json TEXT, brief TEXT, created_at TEXT,
      UNIQUE(site_id,report_date)
    );
    CREATE TABLE IF NOT EXISTS alerts(
      id INTEGER PRIMARY KEY, site_id INTEGER, metric_id INTEGER, report_date TEXT,
      domain TEXT, score REAL, severity TEXT, message TEXT, action TEXT,
      status TEXT DEFAULT 'Open', owner TEXT DEFAULT '', due_date TEXT DEFAULT '',
      note TEXT DEFAULT '', created_at TEXT, updated_at TEXT
    );
    CREATE TABLE IF NOT EXISTS audit(
      id INTEGER PRIMARY KEY, at TEXT, actor TEXT, action TEXT, details TEXT
    );
    CREATE TABLE IF NOT EXISTS risk_rules(
      id INTEGER PRIMARY KEY, trial_id INTEGER, rule_type TEXT DEFAULT 'KRI', scope TEXT DEFAULT 'site',
      domain TEXT, metric TEXT, label TEXT, direction TEXT DEFAULT 'high_bad',
      amber REAL, red REAL, critical REAL, unit TEXT DEFAULT '', weight REAL DEFAULT 1,
      enabled INTEGER DEFAULT 1, action TEXT DEFAULT '', source TEXT DEFAULT 'Manual', source_page INTEGER,
      created_at TEXT
    );
    CREATE TABLE IF NOT EXISTS protocol_signals(
      id INTEGER PRIMARY KEY, trial_id INTEGER, protocol_name TEXT, page INTEGER, domain TEXT, metric TEXT,
      direction TEXT, amber REAL, red REAL, critical REAL, message TEXT, snippet TEXT,
      adopted INTEGER DEFAULT 0, created_at TEXT
    );
    CREATE TABLE IF NOT EXISTS ml_predictions(
      id INTEGER PRIMARY KEY, site_id INTEGER, metric_id INTEGER, report_date TEXT,
      p7 REAL, p14 REAL, p30 REAL, anomaly_score REAL, risk_band TEXT, anomaly_band TEXT,
      model_mode TEXT, model_trained_at TEXT, drivers_json TEXT, model_note TEXT, created_at TEXT,
      UNIQUE(site_id,metric_id)
    );
    CREATE TABLE IF NOT EXISTS ai_briefs(
      id INTEGER PRIMARY KEY, trial_id INTEGER, site_id INTEGER, narrative_type TEXT, narrative TEXT,
      result_json TEXT, provider TEXT, created_at TEXT
    );
    """)
    for col,defn in [
      ("target_sample_size","REAL DEFAULT 0"),("recruitment_end_date","TEXT DEFAULT ''"),
      ("monitoring_interval_days","REAL DEFAULT 90"),("safety_sae_hours","REAL DEFAULT 24"),
      ("protocol_path","TEXT DEFAULT ''"),("protocol_name","TEXT DEFAULT ''")]: _ensure_col(c,"trials",col,defn)
    for col,defn in [("enrollment_target","REAL DEFAULT 0"),("region","TEXT DEFAULT ''")]: _ensure_col(c,"sites",col,defn)
    for col,defn in [
      ("root_cause","TEXT DEFAULT ''"),("capa_action","TEXT DEFAULT ''"),("verification_plan","TEXT DEFAULT ''"),
      ("effectiveness_status","TEXT DEFAULT ''"),("escalated_to","TEXT DEFAULT ''")]: _ensure_col(c,"alerts",col,defn)
    c.commit(); c.close(); seed_demo()


def audit(action,details,actor="local-user"):
    c=conn(); c.execute("INSERT INTO audit(at,actor,action,details) VALUES(?,?,?,?)",(now(),actor,action,details)); c.commit(); c.close()


def default_rules(trial_id:int):
    rules=[
      ("KRI","site","Retention","retention_pct","Participant retention","low_bad",95,90,85,"%","Review retention drivers and participant follow-up procedures."),
      ("KRI","site","Retention","missed_visit_pct","Missed visits","high_bad",5,10,20,"%","Review visit-window tracking and participant contact procedures."),
      ("KRI","site","Protocol","major_deviations_30d","Major protocol deviations (30d)","high_bad",1,2,3,"count","Perform root-cause review and implement CAPA."),
      ("KRI","site","Data quality","overdue_crfs","Overdue CRFs","high_bad",5,15,30,"count","Prioritize source completion and overdue CRF entry."),
      ("KRI","site","Data quality","median_query_age_days","Median query age","high_bad",10,20,30,"days","Assign query owners and clear aged queries."),
      ("KRI","site","Safety","overdue_sae","Overdue SAE reports","high_bad",1,1,2,"count","Immediate PI/safety escalation and reporting correction."),
      ("KRI","site","Supply","ip_days_stock","IP stock remaining","low_bad",30,14,7,"days","Initiate resupply and contingency planning."),
      ("KRI","site","Laboratory","lab_tat_hours","Laboratory turnaround","high_bad",24,48,72,"hours","Review specimen transport/processing bottlenecks."),
      ("KRI","site","Staffing","staff_vacancy_pct","Key staff vacancy","high_bad",10,20,35,"%","Arrange coverage, recruitment and delegation review."),
      ("KRI","site","Monitoring","monitoring_actions_overdue","Overdue monitoring actions","high_bad",2,5,8,"count","Close overdue monitoring actions and verify CAPA."),
      ("QTL","trial","Trial","mean_operational_risk","Mean site operational risk","high_bad",35,50,70,"score","Sponsor/trial leadership review of systemic operational risk."),
      ("QTL","trial","Trial","critical_warnings","Open critical warnings","high_bad",1,2,4,"count","Escalate systemic critical warnings to trial leadership."),
      ("QTL","trial","Recruitment","recruitment_achievement_pct","Trial recruitment achievement","low_bad",85,70,50,"%","Activate trial-wide recruitment recovery plan."),
    ]
    c=conn()
    if c.execute("SELECT COUNT(*) n FROM risk_rules WHERE trial_id=?",(trial_id,)).fetchone()["n"]==0:
        for r in rules:
            c.execute("""INSERT INTO risk_rules(trial_id,rule_type,scope,domain,metric,label,direction,amber,red,critical,unit,action,created_at)
                         VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",(trial_id,*r,now()))
        c.commit()
    c.close()


def seed_demo():
    c=conn()
    if c.execute("SELECT COUNT(*) n FROM trials").fetchone()["n"]:
        tids=[r["id"] for r in c.execute("SELECT id FROM trials").fetchall()]; c.close()
        for tid in tids: default_rules(tid)
        return
    c.execute("""INSERT INTO trials(code,title,phase,sponsor,start_date,planned_end_date,target_sample_size,recruitment_end_date,monitoring_interval_days,safety_sae_hours,created_at)
                 VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
      ("DEMO-RCT-01","Multicentre pragmatic clinical trial — demonstration","III","Demo Sponsor","2026-01-15","2027-06-30",500,"2027-01-31",90,24,now()))
    tid=c.execute("SELECT id FROM trials WHERE code='DEMO-RCT-01'").fetchone()["id"]
    sites=[
      ("ETH-01","Central Hospital","Ethiopia","Addis Ababa","Addis Ababa","Dr A",100),
      ("ETH-02","North Hospital","Ethiopia","Amhara","Bahir Dar","Dr B",100),
      ("ETH-03","East Hospital","Ethiopia","Harari","Harar","Dr C",100),
      ("ETH-04","South Hospital","Ethiopia","Sidama","Hawassa","Dr D",100),
      ("ETH-05","West Hospital","Ethiopia","Oromia","Jimma","Dr E",100),
    ]
    for code,name,country,region,city,pi,target in sites:
        c.execute("INSERT INTO sites(trial_id,code,name,country,region,city,pi_name,enrollment_target,created_at) VALUES(?,?,?,?,?,?,?,?,?)",
                  (tid,code,name,country,region,city,pi,target,now()))
    c.commit(); c.close(); default_rules(tid)

    current={
      "ETH-01":dict(enrolled_total=82,enrollment_30d=16,enrollment_target_30d=18,screen_fail_pct=18,retention_pct=96,missed_visit_pct=3,deviations_30d=2,major_deviations_30d=0,overdue_crfs=3,open_queries=5,median_query_age_days=6,overdue_sae=0,unreviewed_aes=1,safety_training_overdue=0,ip_days_stock=52,stockout_events_90d=0,temp_excursions_30d=0,lab_tat_hours=22,rejected_samples_pct=1,critical_result_overdue=0,staff_vacancy_pct=0,training_overdue_count=0,turnover_90d=0,monitoring_actions_overdue=1,days_since_monitoring=45),
      "ETH-02":dict(enrolled_total=55,enrollment_30d=7,enrollment_target_30d=18,screen_fail_pct=31,retention_pct=91,missed_visit_pct=9,deviations_30d=7,major_deviations_30d=2,overdue_crfs=18,open_queries=29,median_query_age_days=21,overdue_sae=1,unreviewed_aes=4,safety_training_overdue=1,ip_days_stock=10,stockout_events_90d=1,temp_excursions_30d=0,lab_tat_hours=68,rejected_samples_pct=6,critical_result_overdue=2,staff_vacancy_pct=25,training_overdue_count=3,turnover_90d=2,monitoring_actions_overdue=6,days_since_monitoring=104),
      "ETH-03":dict(enrolled_total=71,enrollment_30d=13,enrollment_target_30d=16,screen_fail_pct=22,retention_pct=95,missed_visit_pct=4,deviations_30d=3,major_deviations_30d=0,overdue_crfs=8,open_queries=12,median_query_age_days=10,overdue_sae=0,unreviewed_aes=0,safety_training_overdue=0,ip_days_stock=35,stockout_events_90d=0,temp_excursions_30d=1,lab_tat_hours=31,rejected_samples_pct=2,critical_result_overdue=0,staff_vacancy_pct=8,training_overdue_count=1,turnover_90d=0,monitoring_actions_overdue=2,days_since_monitoring=66),
      "ETH-04":dict(enrolled_total=91,enrollment_30d=21,enrollment_target_30d=18,screen_fail_pct=12,retention_pct=98,missed_visit_pct=1,deviations_30d=1,major_deviations_30d=0,overdue_crfs=1,open_queries=3,median_query_age_days=4,overdue_sae=0,unreviewed_aes=0,safety_training_overdue=0,ip_days_stock=61,stockout_events_90d=0,temp_excursions_30d=0,lab_tat_hours=18,rejected_samples_pct=.5,critical_result_overdue=0,staff_vacancy_pct=0,training_overdue_count=0,turnover_90d=0,monitoring_actions_overdue=0,days_since_monitoring=39),
      "ETH-05":dict(enrolled_total=64,enrollment_30d=10,enrollment_target_30d=17,screen_fail_pct=25,retention_pct=93,missed_visit_pct=7,deviations_30d=4,major_deviations_30d=1,overdue_crfs=11,open_queries=18,median_query_age_days=15,overdue_sae=0,unreviewed_aes=2,safety_training_overdue=0,ip_days_stock=24,stockout_events_90d=0,temp_excursions_30d=0,lab_tat_hours=44,rejected_samples_pct=4,critical_result_overdue=1,staff_vacancy_pct=15,training_overdue_count=2,turnover_90d=1,monitoring_actions_overdue=3,days_since_monitoring=78)
    }
    high_bad={"screen_fail_pct","missed_visit_pct","deviations_30d","major_deviations_30d","overdue_crfs","open_queries","median_query_age_days","overdue_sae","unreviewed_aes","safety_training_overdue","stockout_events_90d","temp_excursions_30d","lab_tat_hours","rejected_samples_pct","critical_result_overdue","staff_vacancy_pct","training_overdue_count","turnover_90d","monitoring_actions_overdue","days_since_monitoring"}
    low_bad={"retention_pct","ip_days_stock"}
    today=date.today()
    for code,vals in current.items():
        for idx,days in enumerate((90,60,30,0)):
            frac=(idx+1)/4
            v=dict(vals)
            # Enrollment is cumulative; create plausible historical totals.
            v["enrolled_total"]=max(0,round(vals["enrolled_total"]*(0.35+0.65*frac),1))
            if code=="ETH-02":
                v["enrollment_30d"]=[14,12,10,7][idx]
                v["retention_pct"]=[97,95,93,91][idx]
                v["ip_days_stock"]=[48,35,22,10][idx]
                v["overdue_crfs"]=[3,8,12,18][idx]
                v["lab_tat_hours"]=[26,38,50,68][idx]
                v["major_deviations_30d"]=[0,1,1,2][idx]
            else:
                for k in high_bad:
                    if k in v: v[k]=round(float(vals[k])*(0.55+0.45*frac),1)
                for k in low_bad:
                    if k in v: v[k]=round(min(100,float(vals[k])+(1-frac)*(8 if k=="retention_pct" else 20)),1)
            add_metrics_by_site_code(code,str(today-timedelta(days=days)),v,create_alerts=(days==0),do_audit=False)


def get_rules(trial_id,scope="site"):
    c=conn(); rows=c.execute("SELECT * FROM risk_rules WHERE trial_id=? AND scope=? ORDER BY rule_type,domain,label",(trial_id,scope)).fetchall(); c.close()
    return [dict(r) for r in rows]


def add_metrics_by_site_code(site_code,report_date,vals,create_alerts=True,do_audit=True):
    c=conn(); site=c.execute("SELECT * FROM sites WHERE code=?",(site_code,)).fetchone()
    if not site: c.close(); raise ValueError(f"Unknown site code: {site_code}")
    prev=c.execute("SELECT * FROM metrics WHERE site_id=? AND report_date<? ORDER BY report_date DESC LIMIT 1",(site["id"],report_date)).fetchone()
    prev_data=json.loads(prev["data_json"]) if prev else None
    rules=[dict(r) for r in c.execute("SELECT * FROM risk_rules WHERE trial_id=? AND scope='site' AND enabled=1",(site["trial_id"],)).fetchall()]
    result=evaluate(vals,prev_data,rules)
    c.execute("""INSERT INTO metrics(site_id,report_date,data_json,overall_score,severity,domains_json,brief,created_at)
                 VALUES(?,?,?,?,?,?,?,?) ON CONFLICT(site_id,report_date) DO UPDATE SET
                 data_json=excluded.data_json,overall_score=excluded.overall_score,severity=excluded.severity,
                 domains_json=excluded.domains_json,brief=excluded.brief,created_at=excluded.created_at""",
              (site["id"],report_date,json.dumps(vals),result.overall_score,result.severity,json.dumps(result.domains),result.brief,now()))
    metric_id=c.execute("SELECT id FROM metrics WHERE site_id=? AND report_date=?",(site["id"],report_date)).fetchone()["id"]
    if create_alerts:
        c.execute("DELETE FROM alerts WHERE metric_id=?",(metric_id,))
        seen=set()
        for a in result.alerts:
            key=(a["domain"],a["message"])
            if key in seen: continue
            seen.add(key)
            c.execute("""INSERT INTO alerts(site_id,metric_id,report_date,domain,score,severity,message,action,created_at,updated_at)
                         VALUES(?,?,?,?,?,?,?,?,?,?)""",
                      (site["id"],metric_id,report_date,a["domain"],a["score"],a["severity"],a["message"],a["action"],now(),now()))
    c.commit(); c.close()
    MODEL_CACHE.pop(site["trial_id"],None)
    if do_audit: audit("metrics.import",f"{site_code} {report_date}: overall risk {result.overall_score}")
    return result


def _trial_histories(trial_id:int):
    c=conn(); sites=c.execute("SELECT id,code FROM sites WHERE trial_id=? ORDER BY id",(trial_id,)).fetchall(); out=[]
    for site in sites:
        rows=c.execute("SELECT report_date,data_json,overall_score,severity FROM metrics WHERE site_id=? ORDER BY report_date ASC",(site["id"],)).fetchall()
        hist=[]
        for r in rows:
            try: data=json.loads(r["data_json"])
            except Exception: data={}
            hist.append({"date":r["report_date"],"data":data,"score":float(r["overall_score"] or 0),"severity":r["severity"]})
        if hist: out.append(hist)
    sig=c.execute("SELECT COUNT(*) n,COALESCE(MAX(m.created_at),'') mx FROM metrics m JOIN sites s ON m.site_id=s.id WHERE s.trial_id=?",(trial_id,)).fetchone()
    c.close(); return out,(int(sig["n"]),sig["mx"])


def get_ai_bundle(trial_id:int,force:bool=False):
    histories,sig=_trial_histories(trial_id)
    cached=MODEL_CACHE.get(trial_id)
    if cached and cached[0]==sig and not force: return cached[1]
    bundle=train_bundle(histories)
    MODEL_CACHE[trial_id]=(sig,bundle)
    return bundle


def site_ai_prediction(site_id:int,bundle=None,save:bool=True):
    c=conn(); site=c.execute("SELECT * FROM sites WHERE id=?",(site_id,)).fetchone()
    if not site: c.close(); return None
    rows=c.execute("SELECT * FROM metrics WHERE site_id=? ORDER BY report_date DESC LIMIT 2",(site_id,)).fetchall()
    c.close()
    if not rows: return None
    current=json.loads(rows[0]["data_json"]); previous=json.loads(rows[1]["data_json"]) if len(rows)>1 else None
    bundle=bundle or get_ai_bundle(site["trial_id"])
    out=ml_predict(bundle,current,previous); out["metric_id"]=rows[0]["id"]; out["report_date"]=rows[0]["report_date"]
    if save:
        c=conn(); c.execute("""INSERT INTO ml_predictions(site_id,metric_id,report_date,p7,p14,p30,anomaly_score,risk_band,anomaly_band,model_mode,model_trained_at,drivers_json,model_note,created_at)
          VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(site_id,metric_id) DO UPDATE SET
          p7=excluded.p7,p14=excluded.p14,p30=excluded.p30,anomaly_score=excluded.anomaly_score,risk_band=excluded.risk_band,anomaly_band=excluded.anomaly_band,
          model_mode=excluded.model_mode,model_trained_at=excluded.model_trained_at,drivers_json=excluded.drivers_json,model_note=excluded.model_note,created_at=excluded.created_at""",
          (site_id,rows[0]["id"],rows[0]["report_date"],out["p7"],out["p14"],out["p30"],out["anomaly_score"],out["risk_band"],out["anomaly_band"],out["model_mode"],out["trained_at"],json.dumps(out["drivers"]),out["model_note"],now())); c.commit(); c.close()
    return out


def trial_ai_predictions(trial_id:int):
    bundle=get_ai_bundle(trial_id)
    c=conn(); sites=c.execute("SELECT * FROM sites WHERE trial_id=? ORDER BY code",(trial_id,)).fetchall(); c.close()
    out={}
    for s in sites: out[s["id"]]=site_ai_prediction(s["id"],bundle)
    return bundle,out


@app.on_event("startup")
def startup(): init_db()


def qtl_summary(trial_id,latest_rows,critical_count):
    vals=[]; enroll=0; target=0
    for s,m in latest_rows:
        if not m: continue
        d=json.loads(m["data_json"]); enroll+=float(d.get("enrollment_30d",0)); target+=float(d.get("enrollment_target_30d",0))
    scores=[float(m["overall_score"]) for _,m in latest_rows if m]
    aggregate={
      "mean_operational_risk":sum(scores)/len(scores) if scores else 0,
      "critical_warnings":critical_count,
      "recruitment_achievement_pct":(enroll/target*100) if target else 100,
    }
    rules=get_rules(trial_id,"trial")
    out=[]
    for r in rules:
        v=aggregate.get(r["metric"],0); sc=score_rule(v,r)
        out.append({**r,"value":round(v,1),"score":sc,"status":sev(sc) if sc else "Within limit"})
    return out


def trial_summary(trial_id:int,role="All roles"):
    c=conn(); trial=c.execute("SELECT * FROM trials WHERE id=?",(trial_id,)).fetchone()
    if not trial: c.close(); return None
    sites=c.execute("SELECT * FROM sites WHERE trial_id=? ORDER BY code",(trial_id,)).fetchall(); rows=[]; countries={}
    bundle=get_ai_bundle(trial_id)
    for s in sites:
        m=c.execute("SELECT * FROM metrics WHERE site_id=? ORDER BY report_date DESC LIMIT 1",(s["id"],)).fetchone()
        hist=c.execute("SELECT * FROM metrics WHERE site_id=? ORDER BY report_date DESC LIMIT 8",(s["id"],)).fetchall()
        forecasts,slope,confidence=forecast_risk(hist)
        aip=site_ai_prediction(s["id"],bundle)
        rows.append((s,m,forecasts,slope,confidence,aip))
        if m:
            countries.setdefault(s["country"],[]).append(float(m["overall_score"]))
    doms=ROLE_DOMAINS.get(role)
    where="WHERE s.trial_id=? AND a.status IN ('Open','In progress')"
    params=[trial_id]
    allalerts=c.execute(f"SELECT a.*,s.code site_code,s.name site_name FROM alerts a JOIN sites s ON a.site_id=s.id {where}",params).fetchall()
    if doms is not None: allalerts=[a for a in allalerts if a["domain"] in doms]
    open_alerts=len(allalerts); critical=sum(1 for a in allalerts if a["severity"]=="Critical")
    latest_pairs=[(s,m) for s,m,_,_,_,_ in rows]
    qtls=qtl_summary(trial_id,latest_pairs,critical)
    c.close(); scores=[m["overall_score"] for _,m,_,_,_,_ in rows if m]
    country_view=[{"country":k,"mean":round(sum(v)/len(v),1),"max":round(max(v),1),"sites":len(v),"severity":sev(sum(v)/len(v))} for k,v in countries.items()]
    return {"trial":trial,"rows":rows,"open_alerts":open_alerts,"critical":critical,"avg_score":round(sum(scores)/len(scores),1) if scores else 0,"qtls":qtls,"countries":country_view,
            "ai_model":{"mode":bundle.mode,"n_samples":bundle.n_samples,"auc":round(bundle.internal_auc,3) if bundle.internal_auc is not None else None,"note":bundle.note,"trained_at":bundle.trained_at}}


@app.get("/",response_class=HTMLResponse)
def dashboard(request:Request,trial_id:Optional[int]=None,role:str="All roles"):
    c=conn(); trials=c.execute("SELECT * FROM trials ORDER BY id").fetchall(); c.close()
    if not trials: raise HTTPException(500,"No trial")
    tid=trial_id or trials[0]["id"]; summary=trial_summary(tid,role)
    c=conn(); alerts=c.execute("""SELECT a.*,s.code site_code,s.name site_name FROM alerts a JOIN sites s ON a.site_id=s.id
      WHERE s.trial_id=? AND a.status IN ('Open','In progress') ORDER BY CASE a.severity WHEN 'Critical' THEN 1 WHEN 'High' THEN 2 WHEN 'Medium' THEN 3 ELSE 4 END,a.score DESC LIMIT 30""",(tid,)).fetchall(); c.close()
    doms=ROLE_DOMAINS.get(role); alerts=[a for a in alerts if doms is None or a["domain"] in doms][:12]
    return templates.TemplateResponse("index.html",{"request":request,"trials":trials,"summary":summary,"alerts":alerts,"role":role,"roles":ROLE_DOMAINS.keys()})


@app.get("/site/{site_id}",response_class=HTMLResponse)
def site_view(request:Request,site_id:int):
    c=conn(); site=c.execute("SELECT s.*,t.code trial_code,t.title trial_title,t.recruitment_end_date FROM sites s JOIN trials t ON s.trial_id=t.id WHERE s.id=?",(site_id,)).fetchone()
    if not site: c.close(); raise HTTPException(404)
    metrics=c.execute("SELECT * FROM metrics WHERE site_id=? ORDER BY report_date DESC LIMIT 12",(site_id,)).fetchall(); latest=metrics[0] if metrics else None
    alerts=c.execute("SELECT * FROM alerts WHERE site_id=? ORDER BY CASE status WHEN 'Open' THEN 1 WHEN 'In progress' THEN 2 ELSE 3 END,created_at DESC",(site_id,)).fetchall()
    brief_rows=c.execute("SELECT * FROM ai_briefs WHERE site_id=? ORDER BY id DESC LIMIT 5",(site_id,)).fetchall(); c.close()
    latest_data=json.loads(latest["data_json"]) if latest else {}; domains=json.loads(latest["domains_json"]) if latest else {}
    ai_prediction=site_ai_prediction(site_id) if latest else None
    ai_briefs=[]
    for b in brief_rows:
        try: result=json.loads(b["result_json"])
        except Exception: result={}
        ai_briefs.append({**dict(b),"result":result})
    forecasts,slope,confidence=forecast_risk(metrics)
    hist_data=[json.loads(m["data_json"]) for m in metrics]
    recruitment=forecast_recruitment(hist_data,float(site["enrollment_target"] or 0))
    if recruitment["days_to_target"] is not None:
        recruitment["projected_date"]=str(date.today()+timedelta(days=recruitment["days_to_target"]))
    chart=[]; chronological=list(reversed(metrics)); n=max(len(chronological)-1,1)
    for i,m in enumerate(chronological):
        chart.append({"x":round(i/n*100,1),"y":round(100-float(m["overall_score"]),1),"date":m["report_date"],"score":m["overall_score"]})
    return templates.TemplateResponse("site.html",{"request":request,"site":site,"metrics":metrics,"latest":latest,"latest_data":latest_data,"domains":domains,"alerts":alerts,"fields":FIELDS,"forecasts":forecasts,"slope":slope,"confidence":confidence,"recruitment":recruitment,"chart":chart,"ai_prediction":ai_prediction,"ai_briefs":ai_briefs,"ai_provider":provider_label(load_ai_settings(AI_SETTINGS_PATH))})


@app.post("/site/{site_id}/metrics")
def add_site_metrics(site_id:int,report_date:str=Form(...),payload:str=Form(...)):
    try: vals=json.loads(payload)
    except Exception: raise HTTPException(400,"Metrics payload must be valid JSON")
    c=conn(); s=c.execute("SELECT code FROM sites WHERE id=?",(site_id,)).fetchone(); c.close()
    if not s: raise HTTPException(404)
    vals={k:float(vals.get(k,0) or 0) for k in FIELDS}; add_metrics_by_site_code(s["code"],report_date,vals)
    return RedirectResponse(f"/site/{site_id}",303)


@app.get("/setup",response_class=HTMLResponse)
def setup_page(request:Request,trial_id:Optional[int]=None):
    c=conn(); trials=c.execute("SELECT * FROM trials ORDER BY id").fetchall(); tid=trial_id or (trials[0]["id"] if trials else None)
    trial=c.execute("SELECT * FROM trials WHERE id=?",(tid,)).fetchone() if tid else None
    sites=c.execute("SELECT * FROM sites WHERE trial_id=? ORDER BY code",(tid,)).fetchall() if tid else []; c.close()
    return templates.TemplateResponse("setup.html",{"request":request,"trials":trials,"trial":trial,"sites":sites})


@app.post("/trials/create")
def create_trial(code:str=Form(...),title:str=Form(...),phase:str=Form(""),sponsor:str=Form(""),start_date:str=Form(""),planned_end_date:str=Form(""),target_sample_size:float=Form(0),recruitment_end_date:str=Form("")):
    c=conn()
    try:
        c.execute("INSERT INTO trials(code,title,phase,sponsor,start_date,planned_end_date,target_sample_size,recruitment_end_date,created_at) VALUES(?,?,?,?,?,?,?,?,?)",(code.strip(),title.strip(),phase,sponsor,start_date,planned_end_date,target_sample_size,recruitment_end_date,now())); c.commit()
        tid=c.execute("SELECT id FROM trials WHERE code=?",(code.strip(),)).fetchone()["id"]
    except sqlite3.IntegrityError: c.close(); raise HTTPException(400,"Trial code already exists")
    c.close(); default_rules(tid); audit("trial.create",f"{code} — {title}"); return RedirectResponse(f"/setup?trial_id={tid}",303)


@app.post("/trials/{trial_id}/update")
def update_trial(trial_id:int,title:str=Form(...),phase:str=Form(""),sponsor:str=Form(""),start_date:str=Form(""),planned_end_date:str=Form(""),target_sample_size:float=Form(0),recruitment_end_date:str=Form(""),monitoring_interval_days:float=Form(90),safety_sae_hours:float=Form(24)):
    c=conn(); c.execute("""UPDATE trials SET title=?,phase=?,sponsor=?,start_date=?,planned_end_date=?,target_sample_size=?,recruitment_end_date=?,monitoring_interval_days=?,safety_sae_hours=? WHERE id=?""",
      (title,phase,sponsor,start_date,planned_end_date,target_sample_size,recruitment_end_date,monitoring_interval_days,safety_sae_hours,trial_id)); c.commit(); c.close(); audit("trial.update",f"Trial {trial_id} configuration updated"); return RedirectResponse(f"/setup?trial_id={trial_id}",303)


@app.post("/trials/{trial_id}/sites/add")
def add_site(trial_id:int,code:str=Form(...),name:str=Form(...),country:str=Form(""),region:str=Form(""),city:str=Form(""),pi_name:str=Form(""),enrollment_target:float=Form(0)):
    c=conn()
    try: c.execute("INSERT INTO sites(trial_id,code,name,country,region,city,pi_name,enrollment_target,created_at) VALUES(?,?,?,?,?,?,?,?,?)",(trial_id,code.strip(),name.strip(),country,region,city,pi_name,enrollment_target,now())); c.commit()
    except sqlite3.IntegrityError: c.close(); raise HTTPException(400,"Site code already exists in this trial")
    c.close(); audit("site.create",f"{code} — {name}"); return RedirectResponse(f"/setup?trial_id={trial_id}",303)


@app.post("/sites/{site_id}/update")
def update_site(site_id:int,code:str=Form(...),name:str=Form(...),country:str=Form(""),region:str=Form(""),city:str=Form(""),pi_name:str=Form(""),enrollment_target:float=Form(0),status:str=Form("Active")):
    c=conn(); site=c.execute("SELECT * FROM sites WHERE id=?",(site_id,)).fetchone()
    if not site:
        c.close(); raise HTTPException(404,"Site not found")
    try:
        c.execute("""UPDATE sites SET code=?,name=?,country=?,region=?,city=?,pi_name=?,enrollment_target=?,status=? WHERE id=?""",
                  (code.strip(),name.strip(),country,region,city,pi_name,enrollment_target,status,site_id)); c.commit()
    except sqlite3.IntegrityError:
        c.close(); raise HTTPException(400,"Site code already exists in this trial")
    tid=site["trial_id"]; c.close(); audit("site.update",f"Site {site_id}: {code} — {name}")
    return RedirectResponse(f"/setup?trial_id={tid}",303)


@app.post("/sites/{site_id}/delete")
def delete_site(site_id:int):
    c=conn(); site=c.execute("SELECT * FROM sites WHERE id=?",(site_id,)).fetchone()
    if not site:
        c.close(); raise HTTPException(404,"Site not found")
    tid=site["trial_id"]; code=site["code"]; name=site["name"]
    metric_ids=[r["id"] for r in c.execute("SELECT id FROM metrics WHERE site_id=?",(site_id,)).fetchall()]
    c.execute("DELETE FROM alerts WHERE site_id=?",(site_id,))
    c.execute("DELETE FROM ml_predictions WHERE site_id=?",(site_id,))
    c.execute("DELETE FROM ai_briefs WHERE site_id=?",(site_id,))
    c.execute("DELETE FROM metrics WHERE site_id=?",(site_id,))
    c.execute("DELETE FROM sites WHERE id=?",(site_id,))
    c.commit(); c.close(); audit("site.delete",f"{code} — {name}; deleted with {len(metric_ids)} metric period(s)")
    return RedirectResponse(f"/setup?trial_id={tid}",303)


@app.get("/configuration",response_class=HTMLResponse)
def configuration(request:Request,trial_id:Optional[int]=None):
    c=conn(); trials=c.execute("SELECT * FROM trials ORDER BY id").fetchall(); tid=trial_id or trials[0]["id"]
    trial=c.execute("SELECT * FROM trials WHERE id=?",(tid,)).fetchone(); rules=c.execute("SELECT * FROM risk_rules WHERE trial_id=? ORDER BY scope,rule_type,domain,label",(tid,)).fetchall(); signals=c.execute("SELECT * FROM protocol_signals WHERE trial_id=? ORDER BY adopted,page,id",(tid,)).fetchall(); c.close()
    return templates.TemplateResponse("configuration.html",{"request":request,"trials":trials,"trial":trial,"rules":rules,"signals":signals,"fields":FIELDS})


@app.post("/trials/{trial_id}/protocol")
async def upload_protocol(trial_id:int,file:UploadFile=File(...)):
    name=Path(file.filename or "protocol.pdf").name
    if not name.lower().endswith(".pdf"): raise HTTPException(400,"Protocol upload currently supports PDF files")
    path=PROTOCOL_DIR/f"trial_{trial_id}_{name}"; raw=await file.read(); path.write_bytes(raw)
    try: pages=extract_pdf_pages(path); suggestions=suggest_operational_rules(pages)
    except Exception as e: path.unlink(missing_ok=True); raise HTTPException(400,f"Could not read PDF: {e}")
    c=conn(); c.execute("UPDATE trials SET protocol_path=?,protocol_name=? WHERE id=?",(str(path),name,trial_id)); c.execute("DELETE FROM protocol_signals WHERE trial_id=? AND adopted=0",(trial_id,))
    for s in suggestions:
        c.execute("""INSERT INTO protocol_signals(trial_id,protocol_name,page,domain,metric,direction,amber,red,critical,message,snippet,created_at)
                     VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",(trial_id,name,s["page"],s["domain"],s["metric"],s["direction"],s["amber"],s["red"],s["critical"],s["message"],s["snippet"],now()))
    c.commit(); c.close(); audit("protocol.upload",f"Trial {trial_id}: {name}; {len(suggestions)} candidate operational rules"); return RedirectResponse(f"/configuration?trial_id={trial_id}",303)


@app.get("/trials/{trial_id}/protocol/view")
def view_protocol(trial_id:int):
    c=conn(); t=c.execute("SELECT protocol_path,protocol_name FROM trials WHERE id=?",(trial_id,)).fetchone(); c.close()
    if not t or not t["protocol_path"] or not Path(t["protocol_path"]).exists(): raise HTTPException(404)
    return FileResponse(t["protocol_path"],media_type="application/pdf",filename=t["protocol_name"])


@app.post("/protocol-signals/{signal_id}/adopt")
def adopt_signal(signal_id:int):
    c=conn(); s=c.execute("SELECT * FROM protocol_signals WHERE id=?",(signal_id,)).fetchone()
    if not s: c.close(); raise HTTPException(404)
    c.execute("""INSERT INTO risk_rules(trial_id,rule_type,scope,domain,metric,label,direction,amber,red,critical,unit,action,source,source_page,created_at)
                 VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
      (s["trial_id"],"KRI","site",s["domain"],s["metric"],s["message"],s["direction"],s["amber"],s["red"],s["critical"],"",s["message"],f"Protocol: {s['protocol_name']}",s["page"],now()))
    c.execute("UPDATE protocol_signals SET adopted=1 WHERE id=?",(signal_id,)); c.commit(); tid=s["trial_id"]; c.close(); audit("protocol.rule_adopt",f"Signal {signal_id} adopted as KRI"); return RedirectResponse(f"/configuration?trial_id={tid}",303)


@app.post("/rules/add")
def add_rule(trial_id:int=Form(...),rule_type:str=Form("KRI"),scope:str=Form("site"),domain:str=Form(...),metric:str=Form(...),label:str=Form(...),direction:str=Form("high_bad"),amber:float=Form(...),red:float=Form(...),critical:float=Form(...),unit:str=Form(""),action:str=Form("")):
    c=conn(); c.execute("""INSERT INTO risk_rules(trial_id,rule_type,scope,domain,metric,label,direction,amber,red,critical,unit,action,created_at)
      VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",(trial_id,rule_type,scope,domain,metric,label,direction,amber,red,critical,unit,action,now())); c.commit(); c.close(); audit("rule.create",f"{rule_type} {label}"); return RedirectResponse(f"/configuration?trial_id={trial_id}",303)


@app.post("/rules/{rule_id}/toggle")
def toggle_rule(rule_id:int):
    c=conn(); r=c.execute("SELECT trial_id,enabled FROM risk_rules WHERE id=?",(rule_id,)).fetchone()
    if not r: c.close(); raise HTTPException(404)
    c.execute("UPDATE risk_rules SET enabled=? WHERE id=?",(0 if r["enabled"] else 1,rule_id)); c.commit(); tid=r["trial_id"]; c.close(); audit("rule.toggle",f"Rule {rule_id} toggled"); return RedirectResponse(f"/configuration?trial_id={tid}",303)


@app.post("/rules/{rule_id}/delete")
def delete_rule(rule_id:int):
    c=conn(); r=c.execute("SELECT trial_id FROM risk_rules WHERE id=?",(rule_id,)).fetchone()
    if not r: c.close(); raise HTTPException(404)
    c.execute("DELETE FROM risk_rules WHERE id=?",(rule_id,)); c.commit(); tid=r["trial_id"]; c.close(); audit("rule.delete",f"Rule {rule_id} deleted"); return RedirectResponse(f"/configuration?trial_id={tid}",303)


@app.get("/import",response_class=HTMLResponse)
def import_page(request:Request):
    c=conn(); sites=c.execute("SELECT s.code,s.name,t.code trial_code FROM sites s JOIN trials t ON s.trial_id=t.id ORDER BY t.code,s.code").fetchall(); c.close()
    default_map={"site_code":"site_code","report_date":"report_date",**{k:k for k in FIELDS}}
    return templates.TemplateResponse("import.html",{"request":request,"sites":sites,"fields":FIELDS,"default_map":json.dumps(default_map,indent=2)})


def process_csv_text(text,column_map=None):
    reader=csv.DictReader(io.StringIO(text)); mp=column_map or {"site_code":"site_code","report_date":"report_date",**{k:k for k in FIELDS}}
    ok=0; errors=[]
    for i,row in enumerate(reader,start=2):
        try:
            sc=(row.get(mp.get("site_code","site_code")) or "").strip(); rd=(row.get(mp.get("report_date","report_date")) or "").strip()
            if not sc or not rd: raise ValueError("missing mapped site_code/report_date")
            vals={}
            for k in FIELDS:
                source=mp.get(k,k); vals[k]=float(row.get(source,0) or 0)
            add_metrics_by_site_code(sc,rd,vals); ok+=1
        except Exception as e: errors.append(f"Row {i}: {e}")
    return ok,errors


@app.post("/import")
async def import_csv(file:UploadFile=File(...),column_map:str=Form("")):
    raw=await file.read()
    try: text=raw.decode("utf-8-sig")
    except Exception: raise HTTPException(400,"CSV must be UTF-8")
    try: mp=json.loads(column_map) if column_map.strip() else None
    except Exception: raise HTTPException(400,"Column map must be valid JSON")
    ok,errors=process_csv_text(text,mp); audit("csv.import",f"Imported {ok} rows from {file.filename}; {len(errors)} errors")
    return RedirectResponse(f"/import?ok={ok}&errors={len(errors)}",303)


@app.post("/redcap/sync")
async def redcap_sync(api_url:str=Form(...),token:str=Form(...),column_map:str=Form("")):
    # Token is used for this request only and is never stored.
    payload={"token":token,"content":"record","action":"export","format":"csv","type":"flat","csvDelimiter":"","rawOrLabel":"raw","rawOrLabelHeaders":"raw","exportCheckboxLabel":"false","returnFormat":"json"}
    try:
        async with httpx.AsyncClient(timeout=45.0) as client:
            resp=await client.post(api_url,data=payload); resp.raise_for_status(); text=resp.text
    except Exception as e: raise HTTPException(400,f"REDCap sync failed: {e}")
    try: mp=json.loads(column_map) if column_map.strip() else None
    except Exception: raise HTTPException(400,"Column map must be valid JSON")
    ok,errors=process_csv_text(text,mp); audit("redcap.sync",f"REDCap API sync imported {ok} rows; {len(errors)} errors")
    return RedirectResponse(f"/import?ok={ok}&errors={len(errors)}&redcap=1",303)


@app.post("/alerts/{alert_id}/update")
def update_alert(alert_id:int,status:str=Form(...),owner:str=Form(""),due_date:str=Form(""),note:str=Form(""),root_cause:str=Form(""),capa_action:str=Form(""),verification_plan:str=Form(""),effectiveness_status:str=Form(""),escalated_to:str=Form("")):
    c=conn(); a=c.execute("SELECT site_id FROM alerts WHERE id=?",(alert_id,)).fetchone()
    if not a: c.close(); raise HTTPException(404)
    c.execute("""UPDATE alerts SET status=?,owner=?,due_date=?,note=?,root_cause=?,capa_action=?,verification_plan=?,effectiveness_status=?,escalated_to=?,updated_at=? WHERE id=?""",
      (status,owner,due_date,note,root_cause,capa_action,verification_plan,effectiveness_status,escalated_to,now(),alert_id)); c.commit(); c.close(); audit("alert.capa_update",f"Alert {alert_id} -> {status}; owner={owner}"); return RedirectResponse(f"/site/{a['site_id']}",303)


@app.get("/ai",response_class=HTMLResponse)
def ai_center(request:Request,trial_id:Optional[int]=None):
    c=conn(); trials=c.execute("SELECT * FROM trials ORDER BY id").fetchall(); tid=trial_id or trials[0]["id"]
    trial=c.execute("SELECT * FROM trials WHERE id=?",(tid,)).fetchone(); sites=c.execute("SELECT * FROM sites WHERE trial_id=? ORDER BY code",(tid,)).fetchall(); c.close()
    bundle,preds=trial_ai_predictions(tid)
    predictions=[{"site":s,"pred":preds.get(s["id"])} for s in sites]
    settings=load_ai_settings(AI_SETTINGS_PATH)
    safe_settings={**settings,"api_key":"********" if settings.get("api_key") else ""}
    return templates.TemplateResponse("ai.html",{"request":request,"trials":trials,"trial":trial,"bundle":bundle,"importance":feature_importance(bundle),"predictions":predictions,"settings":safe_settings,"provider_label":provider_label(settings)})


@app.post("/ai/retrain")
def ai_retrain(trial_id:int=Form(...)):
    MODEL_CACHE.pop(trial_id,None); bundle=get_ai_bundle(trial_id,force=True); audit("ai.model_retrain",f"Trial {trial_id}: mode={bundle.mode}, samples={bundle.n_samples}")
    return RedirectResponse(f"/ai?trial_id={trial_id}",303)


@app.post("/ai/settings")
def ai_settings(provider:str=Form("local"),endpoint:str=Form(""),model:str=Form(""),api_key:str=Form(""),trial_id:int=Form(...)):
    old=load_ai_settings(AI_SETTINGS_PATH)
    key=old.get("api_key","") if api_key in ("","********") else api_key
    settings={"provider":provider,"endpoint":endpoint.strip(),"model":model.strip(),"api_key":key}
    save_ai_settings(AI_SETTINGS_PATH,settings); audit("ai.settings",f"AI provider set to {provider}; model={model}")
    return RedirectResponse(f"/ai?trial_id={trial_id}",303)


@app.post("/site/{site_id}/ai-analyze")
async def site_ai_analyze(site_id:int,narrative_type:str=Form("Monitoring / operations note"),narrative:str=Form(...)):
    c=conn(); site=c.execute("SELECT s.*,t.code trial_code,t.title trial_title FROM sites s JOIN trials t ON s.trial_id=t.id WHERE s.id=?",(site_id,)).fetchone()
    if not site: c.close(); raise HTTPException(404)
    latest=c.execute("SELECT * FROM metrics WHERE site_id=? ORDER BY report_date DESC LIMIT 1",(site_id,)).fetchone()
    alerts=c.execute("SELECT domain,severity,message,action FROM alerts WHERE site_id=? AND status IN ('Open','In progress') ORDER BY score DESC LIMIT 12",(site_id,)).fetchall(); c.close()
    pred=site_ai_prediction(site_id) if latest else None
    context={"trial":site["trial_code"],"site":site["code"],"latest_metrics":json.loads(latest["data_json"]) if latest else {},"open_alerts":[dict(a) for a in alerts],"prediction":{k:v for k,v in (pred or {}).items() if k not in ("drivers",)},"drivers":(pred or {}).get("drivers",[])[:8]}
    settings=load_ai_settings(AI_SETTINGS_PATH)
    try: result=await copilot_analyze(settings,narrative,context)
    except Exception as e: raise HTTPException(400,f"AI copilot request failed: {e}")
    c=conn(); c.execute("INSERT INTO ai_briefs(trial_id,site_id,narrative_type,narrative,result_json,provider,created_at) VALUES(?,?,?,?,?,?,?)",(site["trial_id"],site_id,narrative_type,narrative,json.dumps(result),result.get("provider",settings.get("provider","local")),now())); c.commit(); c.close(); audit("ai.copilot",f"{site['code']} {narrative_type} analyzed via {result.get('provider')}")
    return RedirectResponse(f"/site/{site_id}#ai-copilot",303)


@app.get("/audit",response_class=HTMLResponse)
def audit_page(request:Request):
    c=conn(); rows=c.execute("SELECT * FROM audit ORDER BY id DESC LIMIT 300").fetchall(); c.close(); return templates.TemplateResponse("audit.html",{"request":request,"rows":rows})


@app.get("/export/alerts.csv")
def export_alerts():
    c=conn(); rows=c.execute("""SELECT t.code trial_code,s.code site_code,s.name site_name,a.report_date,a.domain,a.score,a.severity,a.status,a.owner,a.due_date,a.message,a.action,a.root_cause,a.capa_action,a.verification_plan,a.effectiveness_status,a.escalated_to,a.note
      FROM alerts a JOIN sites s ON a.site_id=s.id JOIN trials t ON s.trial_id=t.id ORDER BY a.report_date DESC,a.score DESC""").fetchall(); c.close()
    out=io.StringIO(); w=csv.writer(out); w.writerow(rows[0].keys() if rows else ["trial_code"])
    for r in rows: w.writerow(list(r))
    return StreamingResponse(iter([out.getvalue()]),media_type="text/csv",headers={"Content-Disposition":"attachment; filename=trial_ops_alerts_and_capa.csv"})
