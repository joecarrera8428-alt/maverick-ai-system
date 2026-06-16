# Maverick — Autonomous ML Trading System

> Source code is not public. This document covers the architecture, methodology, and results of a live production system built over six months.

---

## What This Is

This project started with hardware. I spec'd the components, ordered the parts, and assembled a dedicated AI server from scratch: AMD Ryzen 7 9800X3D, RTX 5070 Ti 16GB, 64GB DDR5. Once it was built I configured Ubuntu, set up local LLM inference with Ollama, and got the base services running before a single line of trading code existed. The machine had to work before the system could.

From there I built Maverick, a full-stack ML system that gates futures trade entries through a trained machine learning model, logs every prediction with its full 53-feature vector, and accumulates live outcomes to trigger automated retraining. It runs 24/7 and I manage it remotely over SSH. The whole thing, hardware through production deployment, took six months working solo.

The most important part of this project is not what I built. It is what I found wrong with what I had already built, and what I did about it.

---

## Hardware Build

Everything below was ordered, assembled, and configured by hand. No pre-built machine, no cloud VM.

| Component | Spec |
|---|---|
| **CPU** | AMD Ryzen 7 9800X3D |
| **GPU** | RTX 5070 Ti 16GB |
| **RAM** | 64GB DDR5 |
| **OS** | Ubuntu Linux (custom configured) |
| **Services** | Ollama (local LLM inference), Open WebUI, Flask API, systemd cron pipeline |
| **Remote Access** | SSH / Tailscale (LAN + remote) |

---

## Software Stack

| Layer | Technology |
|---|---|
| **ML Training** | Python, scikit-learn, RandomForest, Logistic Regression, walk-forward validation |
| **Inference API** | Flask (REST), SQLite WAL mode, async signal logging |
| **Statistical Validation** | R (data.table, randomForest, pROC, ggplot2) |
| **Execution Strategy** | C# / NinjaScript (NinjaTrader 8), async HttpClient |
| **Data Pipeline** | pandas, numpy, joblib, gzip/CSV, cron orchestration |
| **External Features** | CBOE VIX/VIX3M daily ingest, regime classification |
| **Infrastructure** | Ubuntu Linux, systemd, SSH/SCP, MD5 integrity verification |
| **Databases** | SQLite x3 — alpha_trades.db, signal_log.db, external_features.db |

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        MAVERICK SYSTEM                          │
│                                                                 │
│  ┌──────────────┐    HTTP/JSON    ┌──────────────────────────┐  │
│  │  NinjaTrader │ ─────────────► │   Flask Inference API    │  │
│  │  8  (C#)     │ ◄───────────── │   /predict_alpha         │  │
│  │              │  conf. score   │   /outcome               │  │
│  │  4 Micro     │                │                          │  │
│  │  Futures     │                │   RandomForest Model     │  │
│  │  Instruments │                │   53 features            │  │
│  └──────────────┘                │   Adaptive thresholds    │  │
│                                  └──────────┬───────────────┘  │
│                                             │                  │
│             ┌───────────────────────────────┴──────────────┐   │
│             │                                              │   │
│      ┌──────▼──────┐              ┌───────────────────┐   │   │
│      │ alpha_       │              │ signal_log.db     │   │   │
│      │ trades.db    │              │                   │   │   │
│      │              │              │ Every model call  │   │   │
│      │ Live trade   │              │ Full feature vec  │   │   │
│      │ outcomes     │              │ Score + threshold │   │   │
│      └──────────────┘              └───────────────────┘   │   │
│                                                             │   │
│  ┌──────────────────────────────────────────────────────┐  │   │
│  │              DAILY AUTOMATED PIPELINE                │  │   │
│  │                                                      │  │   │
│  │  21:05 UTC  VIX/VIX3M ingest from CBOE              │  │   │
│  │  21:10 UTC  Forward labeler runs outcome assignment  │  │   │
│  │  21:15 UTC  R classifies regime, writes thresholds   │  │   │
│  │  05:00 UTC  Retrain gate check  (AUC >= 0.58)       │  │   │
│  │                                                      │  │   │
│  │  Full cycle: ~4 minutes. No intervention needed.    │  │   │
│  └──────────────────────────────────────────────────────┘  │   │
└─────────────────────────────────────────────────────────────────┘
```

NinjaTrader 8 runs the execution strategy on four micro futures instruments. On every qualifying bar it sends an HTTP request to the Flask inference API with a full 53-feature payload. Flask scores the request against the live RandomForest model and returns a confidence score. If the score clears the instrument's adaptive threshold, the strategy fires the entry. Every call, fired or not, gets logged to signal_log.db with the full feature vector and forward outcome for retraining.

The daily pipeline runs automatically at four scheduled times: VIX/VIX3M ingest from CBOE, forward labeler for outcome assignment, R regime classification and threshold write, and the retrain gate check at AUC >= 0.58. Full cycle runs in about four minutes with no intervention needed.

---

## How It Was Built

### The Audit

Before I wrote a line of new code I needed to know exactly what I had. The audit produced 24 confirmed problems. The model had been trained on 10,232 rows and by the time I finished auditing it, 8,207 of those rows were gone. No way to reproduce what the model had learned or validate it against the original data.

Two features were 88% populated in training and 0.012% populated at live inference. The model had been calibrating on values it almost never received in production. That kind of mismatch does not show up until you go looking for it. The labeler had also never run on one instrument at all, and excluded 50-70% of signals on the others. The model had not trained on most of what it was being asked to score.

I also ran an alternative entry strategy across 3,077 trades on 4 instruments during this period. Combined result was -$6,600 after commission. Shut it down and documented everything.

### Null Hypothesis Before Code

I needed to know if the core concept had any statistical edge before rebuilding anything. I built a 5-phase R validation pipeline where each phase produces a binary gate. Nothing moves forward until the gate passes.

Phase 0 documented the random-entry baseline: 22.6% win rate, -22.59 ticks per trade, 72.2% stop-out rate. That is what the system needed to beat.

Phase 1 tested 19 entry conditions across 6.5 years and 66,696 labeled trades. One condition showed positive EV. The zone-based conditions showed nothing different from random.

The bigger finding came from running randomForest feature importance across all 53 features. `bar_hour` ranked first at importance 1,827, three times higher than any entry condition flag. When you trade matters more than what signal fired. That single result drove the entire session-hour filtering logic in the current system. Phases 2 through 4 built the full feature matrix, compared 4 model architectures, and produced a data-driven trade spec. All decisions came from R output. Nothing was hardcoded.

### Build, Deploy, Know When to Stop

Trained models in scikit-learn with walk-forward splits: 2020-2024 train, 2025 validation, 2026 holdout.

Both models failed the AUC gate at threshold 0.52 on the first retrain attempt. One came in at 0.4940 on the holdout. Below 0.50 means the model would anti-select trades if deployed. The gate worked. No pkl was written. The two-stage model architecture was also wrong for this data. 95.5% of survived trades hit target, so Stage 2 p_win was essentially constant and the combined score collapsed to p_survive anyway. Redesigned as single-stage and deployed in proxy mode: all qualifying signals fire, all outcomes post to alpha_trades.db, model score is the only gate.

### Feature Engineering and Pipeline Hardening

Validated VIX/VIX3M ratio as a predictive feature through an R gate (AUC 0.5225, Q1-to-Q5 win rate spread 6.92pp). Built a daily cron ingest pipeline. CBOE data goes into SQLite external_features.db and Flask reads it at inference time. 1,897-row historical backfill covering 2019-2026.

Built adaptive per-instrument thresholds adjusted daily by regime classification and VIX conditions. Base 0.60, floor 0.50, ceiling 0.65. R computes and writes the threshold each session and the server reads it on every predict call.

Two production bugs worth noting. A shortFlagCount gate was filtering all trending-day bars in the live strategy but was absent from the training methodology, so the model had trained on bars the strategy was silently discarding in production. Caught through a training data audit after missing two full trending days. Separately, historical replay bars were firing 4,821 phantom Flask calls on restart and posting sim orders on historical data. Fixed with a realtime guard and cleaned the phantom entries from signal_log.

---

## Current Results

| Metric | Value |
|---|---|
| **Live trades** | 33 |
| **Win rate** | 66.7% (22/33) |
| **Active model** | Random Forest, gen=20, AUC=0.675 |
| **Training samples** | 10,232 |
| **Daily pipeline runtime** | ~4 minutes, fully automated |
| **Instruments** | MES, MNQ, M2K, MYM (micro futures) |
| **Signal logging** | Every prediction logged with full feature vector |
| **Retrain trigger** | 500 live outcomes (accumulating) |

---

## Where This Is Going

The goal is a live real-money trading platform I can run and generate income from. Not a research project, not a demo. A system that earns.

Right now the work is in an Alpha Discovery phase. I'm researching a new model architecture that operates on every market bar rather than only at zone-touch events. The current research has already identified a volatility-based filter that produces a 51.6% win rate in backtesting against a 42.9% breakeven for the target exit structure. When the model clears the out-of-sample AUC gate, real capital goes in.

The longer-term plan is a dedicated multi-GPU inference machine that can run heavier models locally and reduce latency on live predictions. The infrastructure built over these six months is the foundation. The system is still learning and so am I.

---

## What I Would Do Differently

Start with the null hypothesis. Several months of the original build went into execution infrastructure before I formally tested whether the edge existed at all. That order of operations was wrong. R runs first now, always, and nothing gets built until a gate passes.

Log everything from day one. Signal logging was added after the fact and I lost the ability to reconstruct months of inference history because of that. The labeler and signal_log.db are now core infrastructure, not something bolted on later.

Treat train/inference alignment as a first-class requirement from the start, not something you check when something looks off. Feature population rates in training vs. live inference diverged significantly and it took a full audit to surface it. That check is now gated before any model touches production.

---

## A Note on Source Code

The source is not published here. This is a live system that continues to learn and I intend to keep it that way. The architecture, methodology, and findings are documented above. Happy to walk through any part of it in detail in an interview.

---

*Built independently. Hardware spec, assembly, OS configuration, and full software stack. No team, no coursework, no tutorials. Six months.*
