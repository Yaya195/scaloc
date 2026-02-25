# AP Scalability: Design, Implementation Guide, and Evaluation Protocol

## 1) Scope and Goal

This document defines **AP scalability** for the system:

- Train once on an initial AP set of size `N`.
- Deploy and continue inference while AP inventory changes (APs added/removed) **without retraining either encoder or GNN**.
- Keep GNN fixed by preserving the encoder output size (`latent_dim`).
- Quantify and monitor performance degradation under AP churn over time.

This document only covers scalability with respect to APs (not RP scalability).

---

## 2) Definitions

- **Known AP**: AP identity present in the training registry.
- **New AP**: AP identity not present in training registry.
- **AP churn event**: any add/remove/change in AP inventory in a deployment site.
- **Unknown-AP ratio**: fraction of observed AP signals in a query that belong to new APs.
- **Operational scalability**: system keeps producing stable predictions under churn with bounded degradation and no retraining.

---

## 3) Design Constraints

1. **No retraining** post-deployment for encoder and GNN.
2. **Fixed GNN interface**: input to GNN remains fixed-size latent vectors.
3. **Deterministic inference behavior** for new APs (same AP maps to same representation each time).
4. **Backward compatibility** for checkpoints and registry versions.
5. **Graceful degradation** rather than hard failure.

---

## 4) Recommended Architecture for AP Scalability

### 4.1 Frozen Model Contract

- Freeze all learned parameters:
  - AP encoder core (modulator + pooling).
  - GNN + downstream localization head.
- Keep `latent_dim` unchanged across model lifetime.

### 4.2 AP Registry Layer (new runtime component)

Introduce a persistent registry:

- `external_ap_id -> internal_index`
- versioned metadata:
  - `registry_version`
  - `created_at`, `updated_at`
  - `active/inactive` status
  - AP stats (optional): first seen timestamp, frequency, site/floor tags

The registry is a serving artifact coupled with model checkpoint metadata.

### 4.3 Handling New APs without Retraining

Use one of these inference-only options (ordered by recommendation):

1. **Deterministic OOV bucket hashing (recommended baseline)**
   - Reserve `B` OOV buckets.
   - Map unknown AP ID to one bucket deterministically.
   - No table growth required for every AP identity.

2. **Deterministic AP-specific embedding synthesis (advanced)**
   - Generate an embedding vector from AP ID + salt + stable transform.
   - Store generated vector for reproducibility.

3. **Append-only dynamic rows with deterministic init (optional)**
   - Grow table for each new AP identity.
   - Initialize row deterministically.
   - Still no learning updates.

### 4.4 Handling Removed APs

- No model update required.
- AP simply stops appearing in query fingerprints.
- Registry marks AP as inactive for auditing.

### 4.5 Why this works

- GNN depends on latent vectors, not AP cardinality.
- AP-space dynamics are absorbed by the registry + OOV/synthesis policy.
- Inference remains available and deterministic under churn.

---

## 5) Implementation Plan (Phased)

### Phase A — Registry & Mapping

1. Add registry file format (`json` or `parquet`) under model artifacts.
2. Add runtime mapper that converts raw AP IDs to internal indices.
3. Add strict validation and logging for malformed AP IDs.
4. Add deterministic seed/salt config for stable OOV behavior.

### Phase B — Inference Path Integration

1. Route all AP ID parsing through the mapper.
2. Ensure both packed and single-query encoder paths use mapped IDs.
3. Prevent out-of-range embedding access by construction.
4. Add per-query telemetry:
   - unknown-AP ratio
   - malformed AP count
   - active AP count

### Phase C — Monitoring and Guardrails

1. Add confidence annotations tied to unknown-AP ratio.
2. Add thresholds and alerts for sustained AP drift.
3. Track performance vs AP churn in dashboards/reports.

---

## 6) Evaluation Protocol: Train on N APs, Infer under AP Churn

## 6.1 Dataset Splits for AP Churn Study

Let `A_train` be APs visible during training (`|A_train| = N`).

Create evaluation scenarios by transforming inference fingerprints:

- **Additions**:
  - inject new APs not in `A_train` to reach churn levels `+p%`.
- **Removals**:
  - remove AP observations from known set to reach `-p%`.
- **Mixed churn**:
  - combine additions and removals at controlled ratios.

Suggested levels: `p ∈ {0, 5, 10, 20, 30, 40, 50}`.

## 6.2 Experimental Matrix

For each domain/site and each churn level:

1. Evaluate with no retraining.
2. Repeat across seeds (if stochastic components exist).
3. Log:
   - localization metrics (mean/median/p75/p90 error)
   - unknown-AP ratio distribution
   - failure/empty-fingerprint rate
   - inference latency overhead

## 6.3 Time-Series Degradation Monitoring

Simulate deployment timeline (e.g., weekly windows):

- Week `t`: apply cumulative AP churn profile.
- Evaluate on the same validation protocol each week.
- Track metric drift over time.

Primary degradation curve:

- `error_metric(t)` vs `unknown_ap_ratio(t)`

Secondary curves:

- `error_metric` vs AP removal ratio
- `error_metric` vs mixed churn ratio

## 6.4 Acceptance Criteria (example)

Set product-facing SLO thresholds (to be calibrated):

- At unknown-AP ratio `<= 10%`: median error degradation `<= 5%`.
- At unknown-AP ratio `<= 20%`: p90 degradation `<= 12%`.
- No hard inference failure under any tested churn level.

These are placeholders; tune per deployment quality targets.

---

## 7) Metrics and Reporting

### 7.1 Core Localization Metrics

- mean error (m)
- median error (m)
- p75 error (m)
- p90 error (m)

### 7.2 AP-Scalability Metrics

- unknown-AP ratio
- malformed-AP ratio
- AP coverage ratio (`known observed APs / total observed APs`)
- empty fingerprint rate after sanitization
- prediction confidence calibration under churn

### 7.3 Recommended Figures (publication-quality)

1. Degradation curves: error vs churn level.
2. Time-series drift: weekly median/p90 vs unknown-AP ratio.
3. Robustness envelope: worst-case vs typical-case degradation.
4. Latency impact under churn.

---

## 8) Risks and Mitigations

### Risk 1: Many new APs collapse information

- Mitigation: use multiple OOV buckets (not single UNK) and deterministic mapping.

### Risk 2: Registry inconsistency across services

- Mitigation: registry version pinning and checksum validation at load time.

### Risk 3: Silent quality decay over months

- Mitigation: continuous monitoring + alerting on unknown-AP ratio and p90 drift.

### Risk 4: Malformed AP identifiers

- Mitigation: strict parser + sanitization + telemetry.

---

## 9) Deliverables Checklist

- [ ] AP registry schema and versioning policy.
- [ ] Deterministic AP mapping policy (known/OOV/malformed).
- [ ] Inference integration for all encoder entrypoints.
- [ ] Churn evaluation harness and scenario generator.
- [ ] Monitoring dashboard/report templates.
- [ ] Operational runbook with alert thresholds and fallback actions.

---

## 10) Out of Scope (for now)

- RP scalability and graph topology churn handling.
- Online/continual learning updates.
- Federated adaptation strategies for new AP environments.

These are reserved for the next design discussion.
