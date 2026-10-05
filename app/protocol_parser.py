from __future__ import annotations
import re
from pathlib import Path

try:
    import fitz
except Exception:
    fitz=None


def extract_pdf_pages(path: Path):
    if fitz is None:
        raise RuntimeError("PyMuPDF is not installed")
    doc=fitz.open(path)
    out=[]
    for i,page in enumerate(doc):
        out.append({"page":i+1,"text":page.get_text("text") or ""})
    doc.close()
    return out


def _snippet(text, start, end, radius=180):
    a=max(0,start-radius); b=min(len(text),end+radius)
    return " ".join(text[a:b].split())


def suggest_operational_rules(pages):
    """Transparent protocol mining: produces candidates only; user must verify/adopt."""
    suggestions=[]
    patterns=[
        ("Safety","overdue_sae","high_bad",0,1,2,
         re.compile(r"(?:serious adverse event|\bSAE\b).{0,180}?(?:within|no later than|report).{0,80}?(24|48|72)\s*(?:hours?|hrs?)",re.I|re.S),
         "SAE reporting timeline detected; keep overdue SAE reporting at zero and escalate immediately."),
        ("Supply","temp_excursions_30d","high_bad",0,1,2,
         re.compile(r"(?:temperature|storage).{0,160}?(?:excursion|deviation|2\s*[-–]\s*8\s*°?C|2\s+to\s+8)",re.I|re.S),
         "Investigational-product storage/temperature requirement detected."),
        ("Retention","missed_visit_pct","high_bad",5,10,20,
         re.compile(r"(?:visit window|missed visit|follow[- ]?up visit|retention)",re.I),
         "Participant visit-window/retention requirement detected."),
        ("Laboratory","lab_tat_hours","high_bad",24,48,72,
         re.compile(r"(?:laboratory|specimen|sample).{0,180}?(?:within|turnaround|processed).{0,60}?(\d{1,3})\s*(?:hours?|hrs?)",re.I|re.S),
         "Laboratory/sample timing requirement detected."),
        ("Data quality","median_query_age_days","high_bad",7,14,28,
         re.compile(r"(?:CRF|case report form|data entry|query).{0,180}?(?:within|days?).{0,60}?(\d{1,2})\s*days?",re.I|re.S),
         "Data-entry/query timeliness language detected."),
        ("Monitoring","days_since_monitoring","high_bad",60,90,120,
         re.compile(r"(?:monitoring visit|site monitoring|central monitoring|risk[- ]based monitoring)",re.I),
         "Monitoring requirement detected; verify the protocol/monitoring-plan frequency."),
        ("Protocol","major_deviations_30d","high_bad",0,1,2,
         re.compile(r"(?:protocol deviation|protocol violation|major deviation)",re.I),
         "Protocol-deviation management requirement detected."),
    ]
    for page in pages:
        text=page["text"]
        for domain,metric,direction,amber,red,critical,pat,msg in patterns:
            m=pat.search(text)
            if m:
                # If a clear hour value exists for lab, use it to shape thresholds conservatively.
                if metric=="lab_tat_hours" and m.lastindex:
                    try:
                        x=float(m.group(1)); amber=x; red=x*1.5; critical=x*2
                    except Exception: pass
                suggestions.append({
                    "page":page["page"],"domain":domain,"metric":metric,"direction":direction,
                    "amber":round(float(amber),1),"red":round(float(red),1),"critical":round(float(critical),1),
                    "message":msg,"snippet":_snippet(text,m.start(),m.end())
                })
    # Deduplicate by metric, keeping first source occurrence.
    seen=set(); ded=[]
    for s in suggestions:
        if s["metric"] in seen: continue
        seen.add(s["metric"]); ded.append(s)
    return ded
