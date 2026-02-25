# RP Scalability: Design, Implementation Guide, and Evaluation Protocol

## 1) Scope and Goal

This document defines **RP scalability** for the system:

- Train once, then handle RP inventory changes (RPs added/removed/shifted) at deployment.
- Keep encoder and GNN weights frozen for inference-time scalability studies.
- Quantify localization degradation as RP graph structure changes over time.

This document focuses on RP scalability only (AP scalability is in `docs/ap_scalability.md`).

---

## 2) Definitions

- **RP (Reference Point)**: graph node representing aggregated fingerprints at a location.
- **RP churn**: add/remove/update of RP nodes due to environment updates, remapping, crowdsource refresh, etc.
- **Graph refresh**: rebuilding RP graph artifacts (nodes, edges, RP fingerprints, normalization stats).
- **Operational RP scalability**: system continues valid inference under RP churn without model retraining.

---

## 3) Current System Behavior (As Implemented)

1. RP count is adaptive at preprocessing via KMeans (`adaptive_scale`, `min_rps`).
2. Graphs are variable-size; node count is not fixed by model architecture.
3. Inference re-encodes RP fingerprints from graph artifacts each run.
4. Therefore, GNN can consume different RP graph sizes **if artifacts are rebuilt and reloaded**.

Important: current behavior is primarily **offline refresh**, not full online dynamic RP registry.

---

## 4) Target RP-Scalable Design (No Retrain)

### 4.1 Frozen Model Contract

- Freeze encoder and GNN weights at deployment for this mode.
- Keep latent dimensions unchanged.
- Allow only graph artifact updates (RPs and edges).

### 4.2 RP Graph Artifact Contract

Each domain graph artifact must include:

- RP node list (`rp_id`, `x`, `y`, `ap_ids`, `rssi`)
- edge list / edge construction policy
- normalization stats (`coord_min/max`, `rssi_min/max`)
- graph version metadata:
  - `graph_version`
  - `generated_at`
  - `source_data_window`
  - `rp_count`

### 4.3 Update Modes

1. **Batch refresh (recommended first)**
   - Rebuild RP tables + graph periodically (daily/weekly/event-based).
   - Atomically swap active graph version for inference.

2. **Incremental refresh (advanced)**
   - Add/remove/update RPs in graph store between full rebuilds.
   - Recompute local edges or neighborhood windows.

### 4.4 Inference Safety Rules

- Inference always reads one consistent graph version per request.
- Query is normalized with that graph’s stored stats only.
- If domain graph version changes, request should not mix old/new stats mid-flight.

---

## 5) What “No Retrain” Means for RP Scalability

- Allowed:
  - rebuild RP graph artifacts
  - re-encode RP node features with frozen encoder
  - run inference with frozen GNN
- Not allowed:
  - gradient updates to encoder/GNN

Interpretation: model parameters are static; only graph data changes.

---

## 6) Implementation Plan

### Phase A — Versioned Graph Refresh Pipeline

1. Add graph version metadata to every domain artifact.
2. Add atomic activation pointer (`current_graph_version`).
3. Add rollback support to previous version.

### Phase B — Inference Integration

1. Load graph by active version per domain.
2. Enforce strict stats coherence (coords/RSSI) with selected graph version.
3. Add telemetry:
   - rp_count
   - edge_count
   - average node degree
   - graph drift indicators

### Phase C — Operational Controls

1. Refresh schedule policy (time/event driven).
2. Canaries on subset of domains before full rollout.
3. Auto-rollback on quality regressions.

---

## 7) Evaluation Protocol for RP Scalability

## 7.1 Baseline Setup

- Train using baseline RP graph (`G0`) per domain.
- Freeze encoder and GNN.
- Evaluate initial performance (`E0`).

## 7.2 RP Churn Scenarios

Generate inference-time graph variants from baseline domain data:

1. **RP removal**: remove `p%` of RPs (`p ∈ {5,10,20,30,40,50}`).
2. **RP addition**: add `p%` new RPs from newly available samples.
3. **RP relocation/noise**: perturb RP coordinates to simulate map drift.
4. **Mixed churn**: combine add/remove/shift.

## 7.3 Structural Controls

For each scenario, control:

- graph size (`|V|`)
- neighborhood parameter (`k`)
- degree distribution
- connectedness (detect isolated nodes/components)

## 7.4 Metrics

Primary localization metrics:

- mean, median, p75, p90 error (meters)

RP-graph robustness metrics:

- rp_count
- avg degree
- connected components
- isolated node ratio
- graph refresh latency

## 7.5 Time-Series Monitoring

Simulate periodic graph refresh windows (`t=1..T`):

- apply cumulative RP churn profile
- evaluate fixed validation set each window
- track drift curves:
  - error(t) vs rp_count(t)
  - p90(t) vs graph-connectivity metrics

## 7.6 Acceptance Criteria (example placeholders)

- Up to 10% RP removal: median degradation <= 5%
- Up to 20% mixed churn: p90 degradation <= 12%
- No catastrophic inference failure under tested scenarios

Tune thresholds by deployment quality targets.

---

## 8) Risks and Mitigations

### Risk 1: Sparse/fragmented graphs after RP removal

- Mitigation: enforce minimum RP count and connectivity checks before activation.

### Risk 2: Stats mismatch across graph versions

- Mitigation: version-lock graph + normalization stats bundle.

### Risk 3: Silent quality drift

- Mitigation: ongoing p90 + graph health monitoring with alerts and rollback.

### Risk 4: Over-aggressive RP addition causing noisy nodes

- Mitigation: RP quality filters (minimum support, RSSI consistency, temporal stability).

---

## 9) Deliverables Checklist

- [ ] Versioned graph artifact schema.
- [ ] Atomic graph activation/rollback mechanism.
- [ ] RP churn scenario generator for offline evaluation.
- [ ] Monitoring dashboards for graph health + localization drift.
- [ ] Operational playbook (refresh cadence, canary, rollback).

---

## 10) Notes for Next Discussion

Open design decisions to finalize:

1. Refresh cadence (scheduled vs event-driven).
2. Activation policy (global vs per-domain graph versions).
3. Connectivity constraints for graph acceptance.
4. Whether to support true online incremental RP updates or keep batch refresh only.
