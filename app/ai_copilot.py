from __future__ import annotations

import json, re
from pathlib import Path
from typing import Any, Dict
import httpx


def load_settings(path: Path) -> dict:
    defaults={
        "provider":"local",
        "endpoint":"http://127.0.0.1:11434",
        "model":"",
        "api_key":""
    }
    if path.exists():
        try:
            obj=json.loads(path.read_text(encoding="utf-8"))
            defaults.update({k:v for k,v in obj.items() if k in defaults})
        except Exception:
            pass
    return defaults


def save_settings(path: Path, settings: dict) -> None:
    path.write_text(json.dumps(settings,indent=2),encoding="utf-8")


def provider_label(settings: dict) -> str:
    p=settings.get("provider","local")
    if p=="ollama": return f"Ollama · {settings.get('model') or 'model not set'}"
    if p=="openai_compatible": return f"OpenAI-compatible · {settings.get('model') or 'model not set'}"
    return "Local explainable ML/NLP"


def _json_from_text(text: str) -> dict:
    text=text.strip()
    try: return json.loads(text)
    except Exception: pass
    m=re.search(r"\{.*\}",text,re.S)
    if m:
        try: return json.loads(m.group(0))
        except Exception: pass
    return {"summary":text,"root_causes":[],"recommended_actions":[],"evidence":[],"uncertainties":["Could not parse structured model output."]}


def local_analysis(narrative: str, context: dict) -> dict:
    txt=(narrative or "").lower()
    causes=[]; actions=[]; evidence=[]
    rules=[
        (("staff","vacan","turnover","nurse","coordinator"),"Staffing capacity or continuity may be contributing to operational deterioration.","Review delegation, coverage, workload and training; document a staffing recovery plan."),
        (("query","crf","data entry","database"),"Data-management backlog may be delaying data completion and query resolution.","Assign query/CRF owners, set a clearance deadline and monitor backlog age weekly."),
        (("recruit","screen","referral","eligible"),"Recruitment pathway performance may be below plan.","Review screening logs, referral sources, screen-failure reasons and site-specific recruitment actions."),
        (("missed visit","retention","lost to follow","follow-up"),"Participant follow-up or retention processes may require attention.","Review missed visits, contact attempts, visit windows and retention barriers; implement targeted follow-up."),
        (("sae","safety","adverse event"),"A safety-reporting workflow issue may require immediate operational review.","Escalate to the PI/safety lead, verify reporting timelines and document corrective action."),
        (("stock","drug","ip ","pharmacy","temperature"),"Investigational-product supply or storage processes may be at risk.","Verify inventory, resupply lead time, accountability and temperature-control actions."),
        (("lab","sample","specimen","turnaround"),"Laboratory or specimen-flow delays may be contributing to site risk.","Map specimen collection, transport, processing and critical-result communication; address bottlenecks."),
        (("deviation","protocol","noncompliance"),"Protocol adherence processes may need reinforcement.","Perform root-cause review, retrain relevant staff and verify CAPA effectiveness."),
    ]
    for terms,cause,action in rules:
        if any(t in txt for t in terms):
            causes.append(cause); actions.append(action)
    for d in context.get("drivers",[])[:5]:
        evidence.append(f"{d.get('label')}: {d.get('value')} ({d.get('direction')})")
    if not causes:
        causes=["No single operational root cause can be inferred confidently from the narrative alone."]
        actions=["Review the highest-risk indicators with the site team and document root-cause analysis before assigning CAPA."]
    return {
        "summary":"AI-assisted operational hypothesis generated from the narrative and current risk signals. Human verification is required.",
        "root_causes":causes[:5],"recommended_actions":actions[:5],"evidence":evidence,
        "uncertainties":["This is decision support, not a causal determination.","Confirm against source records, protocol, monitoring findings and site-team input."],
        "provider":"local"
    }


SYSTEM_PROMPT="""You are an AI copilot for clinical-trial OPERATIONS and risk-based quality management. You do not make clinical, participant-safety, DSMB, regulatory, protocol-waiver, or final CAPA decisions. Analyze only operational causes and actions. Use the supplied evidence and narrative; do not invent facts. Return JSON only with keys: summary (string), root_causes (array of strings), recommended_actions (array of strings), evidence (array of strings), uncertainties (array of strings). Explicitly flag urgent safety-reporting or participant-protection workflow concerns for human escalation when present."""

async def analyze(settings: dict, narrative: str, context: dict) -> dict:
    provider=settings.get("provider","local")
    if provider=="local":
        return local_analysis(narrative,context)
    user_payload={"narrative":narrative,"operational_context":context}
    prompt=json.dumps(user_payload,ensure_ascii=False)
    if provider=="ollama":
        endpoint=(settings.get("endpoint") or "http://127.0.0.1:11434").rstrip("/")+"/api/chat"
        model=settings.get("model") or "llama3.1:8b"
        payload={"model":model,"stream":False,"messages":[{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":prompt}],"options":{"temperature":0.1}}
        async with httpx.AsyncClient(timeout=120.0) as client:
            r=await client.post(endpoint,json=payload); r.raise_for_status(); obj=r.json()
        text=((obj.get("message") or {}).get("content") or "").strip()
        out=_json_from_text(text); out["provider"]="ollama"; return out
    if provider=="openai_compatible":
        base=(settings.get("endpoint") or "").rstrip("/")
        if not base: raise ValueError("OpenAI-compatible endpoint is not configured")
        endpoint=base if base.endswith("/chat/completions") else base+"/chat/completions"
        headers={"Content-Type":"application/json"}
        if settings.get("api_key"): headers["Authorization"]="Bearer "+settings["api_key"]
        payload={"model":settings.get("model") or "","temperature":0.1,"messages":[{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":prompt}]}
        async with httpx.AsyncClient(timeout=120.0) as client:
            r=await client.post(endpoint,json=payload,headers=headers); r.raise_for_status(); obj=r.json()
        text=(((obj.get("choices") or [{}])[0].get("message") or {}).get("content") or "").strip()
        out=_json_from_text(text); out["provider"]="openai_compatible"; return out
    raise ValueError(f"Unknown AI provider: {provider}")
