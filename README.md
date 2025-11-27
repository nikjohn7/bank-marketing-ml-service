# Bank Marketing ML Service

A production-ready ML service predicting term deposit subscriptions from bank marketing campaign data.

## Quick Start

The repo includes a pre-trained model in `artifacts/`, so the API works out of the box.

**Docker (recommended):**
```bash
docker build -t bank-marketing-api .
docker run -p 8000:8000 bank-marketing-api

# API is now running at http://localhost:8000
# Try: curl http://localhost:8000/health
```

**Local setup:**
```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Run API (model already trained)
uvicorn src.app:app --host 0.0.0.0 --port 8000

# Run tests
pytest tests/ -v
```

**To retrain the model:**
```bash
# Local
python -m src.train

# Or via Docker
docker run -v $(pwd)/artifacts:/app/artifacts bank-marketing-api python -m src.train
```

---

## Part A: Model Training

### Key EDA Decisions

| Finding | Impact | Action |
|---------|--------|--------|
| `duration` correlates 0.44 with target | Data leakage (only known after call ends) | **Excluded from training** |
| 16% rows share identical demographics | Same person contacted multiple times | Created `profile_id` for grouped train/test split |
| 88.3% negative class | Severe imbalance | Used `class_weight="balanced"`, focused on F1/AUC |
| `pdays`: 82% are -1 | Raw value less interpretable | Engineered `was_contacted_before` binary feature |
| Balance has extreme outliers (100K+) | Skews scaling | Applied RobustScaler |

### Model Performance

| Metric | Value |
|--------|-------|
| ROC-AUC | 0.766 |
| F1 (default threshold 0.5) | 0.333 |
| F1 (optimized threshold 0.22) | 0.451 |

**Why RandomForest?** It handles mixed feature types well, doesn't need much tuning for an MVP, and gives us feature importance for free. Good starting point before trying more complex models.

**Why the custom threshold?** The default 0.5 threshold is quite conservative for imbalanced data. By lowering it to 0.22, we catch more potential subscribers (higher recall) at the cost of some false positives - worth it if contact costs are low.

### Future Model Improvements

If I had more time, here's what I'd explore:

- **Try XGBoost or LightGBM** - These often outperform RandomForest on tabular data, especially with proper tuning
- **Hyperparameter search** - I used sensible defaults; a proper grid search might squeeze out a few more percentage points
- **Handle class imbalance better** - Could try oversampling the minority class or adjusting sample weights more carefully
- **Feature interactions** - Things like age × balance or job × education might capture useful patterns
- **Better understand predictions** - Add feature importance plots to explain *why* the model predicts what it does

---

## Part B: Model Hosting

### API Endpoints

**`GET /health`** - Returns `{"status": "ok", "model_loaded": true}`

**`POST /predict`** - Returns prediction with probability

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age": 40, "job": "management", "marital": "married", ...}'

# Response: 
# {
#   "prediction": "no", 
#   "probability": 0.1234, 
#   "threshold_used": 0.22,
#   "model_version": "v1"
# }
```

### Design Decisions

1. **Dynamic Schema Validation**: Pydantic model generated from `schema_contract.json` - if valid categories change, API validation updates automatically
2. **Feature Engineering Bridge**: API accepts `pdays` (what the caller knows), transforms to `was_contacted_before` internally (what the model expects)
3. **Pipeline Serialization**: Preprocessing bundled with model in single artifact - avoids train/serve skew issues
4. **Optimized Threshold**: API uses the 0.22 threshold from training analysis, not the default 0.5
5. **Structured Logging**: All predictions logged with key features for debugging and monitoring

---

## Additional Questions

### Part A

**Q1: What testing would you have on training code?**

I'd focus on a few key areas:
- **Data checks**: Does the data match the expected schema? Are values in sensible ranges?
- **Pipeline sanity**: Can we load the saved model and get a valid prediction?
- **Reproducibility**: Running training twice with the same seed should give the same model
- **Performance guardrails**: Fail the build if AUC drops below some threshold (e.g., 0.75)

The current tests cover the basics - verifying artifacts exist and predictions work. In production, I'd add the performance regression checks.

**Q2: How do you store configs, data, hyperparameters, and evaluation outcomes?**

Right now I'm saving everything to `training_run.json` - timestamp, hyperparameters, metrics, threshold analysis. It's simple but works for a single-model setup.

For a team working on multiple experiments, I'd use something like MLflow or Weights & Biases. The main things I'd want tracked:
- Git commit (so I can reproduce exactly)
- Data version or checksum
- All hyperparameters
- Metrics and any plots

**Q3: Training results fluctuate without changes - what's wrong?**

Usually comes down to randomness not being controlled:
- Model uses random sampling internally (e.g., RandomForest bootstrap)
- Train/test split is random
- Data might load in different order

Fix: Set `random_state=42` everywhere (already done in my code), pin library versions, sort data after loading if needed.

**Q4: Retraining strategy?**

I'd think about two triggers:

*Scheduled*: Retrain monthly or quarterly as a baseline, since customer behavior drifts over time.

*Triggered*: Retrain if we notice performance dropping - either the model metrics (AUC, F1) or business metrics (actual conversion rate vs predicted).

The process would be: train new model → compare against current model on holdout data → if better, deploy gradually (start with 5% of traffic) → monitor → full rollout.

---

### Part B

**Q1: How would you safely promote a new model?**

I'd do it in stages:
1. First, test offline - make sure it beats the current model on a holdout set
2. Shadow deploy - run both models in parallel but only serve the old one, compare predictions
3. Canary release - send 5-10% of traffic to new model, watch metrics closely
4. Gradual rollout - if metrics look good, increase to 25%, 50%, then 100%

Key thing is having automated rollback if something goes wrong. If conversion rate drops or error rate spikes, switch back to the old model immediately.

**Q2: Schema/preprocessing differs from production model - how to handle?**

A few options depending on the situation:

If it's a breaking change (new required field), I'd version the API - keep `/v1/predict` running the old model while `/v2/predict` runs the new one. Give consumers time to migrate.

If it's just preprocessing changes, the nice thing about bundling preprocessing in the pipeline is that consumers don't need to change anything - they send raw features, we handle the transformation.

For any change, I'd communicate early, provide a migration guide, and monitor error rates during the transition.

**Q3: Observability metrics and alerting?**

I'd want to track:

| What | Why | Alert if |
|------|-----|----------|
| Response time (p95) | User experience | > 200ms for 5 min |
| Error rates (5xx) | Something's broken | > 1% |
| Prediction distribution | Detect drift | Shifts > 15% from training |
| Validation errors | Bad input data | > 5% of requests |
| Model age | Staleness | > 90 days since training |

Implementation would be structured logging → metrics collection (Prometheus) → dashboards (Grafana) → alerts to Slack.

I've added basic structured logging to the API as a starting point.

---

## Project Structure

```
├── src/
│   ├── app.py              # FastAPI application
│   ├── train.py            # Training pipeline
│   ├── config.py           # Centralized config
│   └── schema_contract.json
├── tests/
│   ├── test_api.py         # API endpoint tests
│   └── test_train.py       # Training pipeline tests
├── artifacts/              # Model + metadata
├── eda.ipynb              # Exploratory analysis
├── Dockerfile
├── requirements.txt        # Production dependencies
└── requirements-dev.txt    # Development dependencies (includes Jupyter, etc.)
```

---

## Time Breakdown

I aimed to stay within the 3-hour limit. Here's roughly how I spent my time:

| Task | Time | Notes |
|------|------|-------|
| Reading the brief | ~7 min | Understanding requirements and dataset description |
| EDA | ~1 hr 20 min | This took longer than expected - I kept finding things worth investigating (the repeat customers, duration leakage, pdays distribution). Could have been more disciplined here. |
| Data cleaning & schema | ~10 min | Straightforward once EDA decisions were made |
| Model training | ~15 min | Knew RandomForest would be my MVP choice |
| Threshold analysis | ~5 min | Quick addition - didn't want to try multiple models since this was only 40% of the exercise |
| Building the API | ~40 min | Took longer than I'd like - fiddled with the dynamic Pydantic generation |
| Writing tests | ~15 min | Focused on covering the main paths and edge cases |
| README & documentation | ~25 min | Tried to be concise while covering all the required questions |
| Second pass / tweaks | ~15 min | While writing the README I noticed small things to improve |

**Total: ~3 hours 50 minutes**

Slightly over, mainly because I went deep on the EDA. In hindsight, I could have timeboxed that more strictly given the 40/60 split guidance.