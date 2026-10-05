# AI4CT v0.3 — AI-powered Clinical-Trial Operations Early-Warning System

AI4CT v0.3 is a local-first, human-governed clinical-trial operations and risk-based quality management (RBQM) prototype. It combines transparent KRI/QTL logic with **actual machine-learning prediction**, **unsupervised anomaly detection**, and an **AI operations copilot**.

## What is genuinely AI-powered in v0.3

### 1. Predictive machine learning
A Random Forest classifier estimates the probability that a site will enter a **high/critical operational-risk state**. The interface reports 7-, 14- and 30-day probabilities. The 30-day probability is the model output; 7- and 14-day probabilities are horizon-scaled from the 30-day estimate and should be treated as decision-support signals rather than validated event probabilities.

### 2. Anomaly detection
An Isolation Forest identifies site-period patterns that are unusual relative to the model's learned operational distribution. Each site receives an anomaly score and an interpretation (Typical / Watch / Unusual / Very unusual).

### 3. Explainable AI
The site page lists the strongest model drivers for the current prediction. These are model associations, not causal conclusions.

### 4. AI operations copilot
Users can paste a monitoring note, protocol-deviation narrative, CAPA narrative, site communication, or safety-reporting workflow note. The copilot combines that narrative with current site risk signals and returns:
- operational summary
- possible root causes
- suggested operational actions
- evidence used
- uncertainty / safeguards

The copilot works in a local explainable mode by default. It can optionally use:
- **Ollama** for a local LLM; or
- an **OpenAI-compatible chat-completions endpoint**.

No external LLM is required for the ML early-warning system.

## Important model-mode safeguard

AI4CT deliberately distinguishes two modes:

- **bootstrap-demo**: used while the installation has too little longitudinal trial history. The model is trained on synthetic operational scenarios so that the software and workflow can be evaluated. It is **not a clinically, operationally, or regulatorily validated predictive model**.
- **local-history**: activated automatically once enough longitudinal site transitions and both outcome classes are available. The model then trains on data accumulated in the local installation. This still requires formal validation before production use.

The AI Center always shows the model mode, training sample count, internal AUC (when estimable), and validation warning.

## Existing v0.2.1 functions retained

- trial and site creation/editing/deletion
- protocol PDF upload
- protocol-derived operational-rule suggestions with source pages
- configurable site KRIs and trial-level QTLs
- transparent operational risk score
- recruitment, retention, protocol, data-quality, safety, IP/supply, laboratory, staffing and monitoring domains
- rule/trend early warnings
- CAPA ownership, due dates, escalation, verification and effectiveness tracking
- CSV import
- REDCap-compatible CSV mapping
- REDCap API synchronization
- recruitment forecasting
- country/site risk views
- audit trail
- alert/CAPA export

## Install on Windows

Extract the ZIP. Open the extracted `AI4CT_v03` folder in File Explorer. Click the address bar, type:

```text
cmd
```

Press Enter, then run:

```bat
py -m venv .venv
```

```bat
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Then start AI4CT:

```bat
.\.venv\Scripts\python.exe run.py
```

Open:

```text
http://127.0.0.1:8010
```

Keep the Command Prompt window open while using the app.

If port 8010 is already occupied by an older AI4CT version, either close the old server or run `netstat -ano | findstr :8010` and stop the old process before starting v0.3.

## Upgrade from v0.2 / v0.2.1 without losing your data

1. Close the old AI4CT server.
2. Make a backup of your old `trial_ops.sqlite3`.
3. Copy that file into the v0.3 folder, replacing the demo database.
4. Copy your old `data` folder as well if you uploaded protocols.
5. Start v0.3. Database migration is automatic; the new AI tables are added without deleting existing trial/site data.

## AI Center

Open **AI center** from the navigation bar. It displays:
- current model mode
- number of training examples
- internal AUC when available
- positive-class rate
- site 7/14/30-day AI probabilities
- anomaly scores
- global feature importance
- language-model provider configuration
- manual model retraining

## Optional Ollama connection

If Ollama is installed and running locally, open **AI center** and choose `Ollama (local LLM)`. Typical endpoint:

```text
http://127.0.0.1:11434
```

Enter the installed Ollama model name. The app will send the operational narrative and structured site context to your local Ollama instance.

## Optional OpenAI-compatible API

Choose `OpenAI-compatible API`, enter the endpoint base ending in `/v1` (or a compatible base URL), model name, and API key. The key is stored locally in `ai_settings.local.json` if saved. **Do not share or commit that file.**

## Human-governance boundary

AI4CT is operational decision support. It must not autonomously:
- make participant clinical decisions
- determine causality for safety events
- waive protocol requirements
- make DSMB decisions
- make sponsor/regulatory submissions or determinations
- close CAPAs without human verification
- replace investigators, monitors, data managers, safety staff, pharmacists, laboratory staff, sponsors, ethics committees, regulators or DSMBs

Any high-risk or safety-related AI signal should trigger human review against source data and governing documents.

## Key local files

- `trial_ops.sqlite3` — trials, sites, metrics, alerts, CAPAs, AI predictions and AI analyses
- `data/` — uploaded protocol files
- `ai_settings.local.json` — optional local AI provider credentials/settings; keep private
- `app/` — source code
- `requirements.txt` — Python dependencies

## Research-development recommendation

Before prospective operational use, validate AI4CT using retrospective or prospectively collected trial-operations data. Predefine prediction horizons, outcomes, missing-data handling, calibration metrics, discrimination metrics, alert burden, false-positive rate, decision-curve/utility measures, subgroup/site performance, and human-factors endpoints. Freeze and version models used in formal evaluation.
