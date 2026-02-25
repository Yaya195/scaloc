# Scalability Framework (APs + RPs)

This document unifies scalability planning for:

- AP scalability (`docs/ap_scalability.md`)
- RP scalability (`docs/rp_scalability.md`)

Goal: provide one implementation and evaluation framework for **no-retrain inference scalability**.

---

## 1) Unified Objective

Train once, then continue deployment under environment drift while preserving model weights.

- Encoder/GNN parameter updates are disabled during scalability evaluation.
- Inference remains operational under AP and RP inventory changes.
- Degradation is measured, monitored, and bounded by acceptance criteria.

---

## 2) Scalability Axes

## 2.1 AP Axis

- Drift source: AP add/remove/identifier changes.
- Model interface impact: AP-ID handling in encoder input pipeline.
- No-retrain mechanism: registry + deterministic unknown handling.
- Primary stress metric: unknown-AP ratio.

## 2.2 RP Axis

- Drift source: RP add/remove/relocation and graph topology updates.
- Model interface impact: graph node/edge set and normalization artifacts.
- No-retrain mechanism: versioned graph refresh with frozen model.
- Primary stress metric: graph health under churn (size/connectivity/degree).

---

## 3) Shared Inference Contract

1. Encoder output dimensionality (`latent_dim`) is fixed.
2. GNN architecture and weights are fixed.
3. All environment adaptation occurs in data/registry/graph artifacts.
4. Every request is evaluated against one consistent artifact version bundle.

---

## 4) Unified Artifact Versioning

Each deployment bundle should include:

- `model_version`: encoder+GNN checkpoint identifier
- `ap_registry_version`: AP mapping policy snapshot
- `graph_version`: RP graph snapshot per domain
- `normalization_version`: coordinate/RSSI stats bound to graph version

Inference request logs should record all four for auditability.

---

## 5) Unified Experiment Matrix

Evaluate 2D churn conditions:

- AP churn levels: `a ∈ {0, 5, 10, 20, 30, 40, 50}%`
- RP churn levels: `r ∈ {0, 5, 10, 20, 30, 40, 50}%`

Matrix cell `(a, r)`:

1. Apply AP transformation to queries.
2. Apply RP/graph transformation to domain artifacts.
3. Run frozen-model inference.
4. Record metrics.

This yields a degradation surface rather than a single curve.

---

## 6) Metrics (Unified)

## 6.1 Localization Quality

- mean / median / p75 / p90 localization error (m)

## 6.2 AP Robustness

- unknown-AP ratio
- malformed-AP ratio
- AP coverage ratio

## 6.3 RP/Graph Robustness

- RP count
- average degree
- connected components
- isolated node ratio

## 6.4 Operational Metrics

- inference latency
- failure rate
- confidence score drift

---

## 7) Monitoring Over Time

For each deployment window `t` (daily/weekly):

- collect AP and RP churn indicators
- run fixed validation protocol
- track:
  - `error_p50(t)`, `error_p90(t)`
  - `unknown_ap_ratio(t)`
  - `rp_count(t)`, connectivity metrics

Monitor not only level drift but slope (rate of degradation).

---

## 8) Acceptance Criteria Template

Define tiered SLOs by churn band.

Example placeholders:

- Low churn band (`a<=10%`, `r<=10%`):
  - median degradation <= 5%
  - p90 degradation <= 10%
- Medium churn band (`a<=20%`, `r<=20%`):
  - median degradation <= 10%
  - p90 degradation <= 15%
- No hard inference failures in all tested cells.

Thresholds must be calibrated using deployment targets.

---

## 9) Decision Rules

- **Green**: metrics within SLOs → keep current artifacts active.
- **Yellow**: p90 drift warning → canary refreshed AP/RP artifacts.
- **Red**: sustained SLO breach → rollback artifact version and trigger maintenance workflow.

This preserves no-retrain operation while enforcing quality control.

---

## 10) Implementation Order

1. Finalize AP registry + deterministic mapping policy.
2. Finalize versioned RP graph bundle and activation/rollback flow.
3. Build unified churn scenario generator.
4. Build experiment runner for 2D matrix `(AP churn, RP churn)`.
5. Build monitoring dashboard and alert rules.

---

## 11) Related Documents

- AP-specific design: `docs/ap_scalability.md`
- RP-specific design: `docs/rp_scalability.md`
- Model overview: `docs/model_design.md`
