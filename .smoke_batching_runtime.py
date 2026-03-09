import copy
from scripts.run_scalability import build_experiment, run_one_experiment, load_domains, load_queries, load_config, resolve_device, SAMPLES_DIR

model_cfg = load_config("model_config")
fl_cfg = load_config("fl_config")
train_cfg = load_config("train_config")

device = resolve_device(train_cfg["training"].get("device", "auto"))
print(f"[smoke] device={device}")

all_domains = load_domains()
if not all_domains:
    raise RuntimeError("No domains loaded")

domain_ids = sorted(all_domains.keys())[:2]
subset = {d: all_domains[d] for d in domain_ids}
print(f"[smoke] using domains={domain_ids}")

val_queries_all = load_queries(SAMPLES_DIR / "val_samples.parquet", "val")

fl_cfg_smoke = copy.deepcopy(fl_cfg)
fl_cfg_smoke["federated"]["rounds"] = 1
fl_cfg_smoke["federated"]["local_epochs"] = 1

server, clients, val_ds = build_experiment(
    domains_subset=subset,
    val_queries_all=val_queries_all,
    model_cfg=model_cfg,
    fl_cfg=fl_cfg_smoke,
    train_cfg=train_cfg,
    device=device,
    max_rps_per_domain=48,
    max_queries_per_domain=48,
    seed=fl_cfg_smoke["federated"].get("seed", 42),
)

print(f"[smoke] clients={len(clients)}, val_domains={len(val_ds)}")
if not clients:
    raise RuntimeError("No clients built in smoke test")

res = run_one_experiment(
    name="smoke_batching_runtime",
    server=server,
    clients=clients,
    val_datasets=val_ds,
    fl_cfg=fl_cfg_smoke,
    train_cfg=train_cfg,
    device=device,
    rounds_override=1,
)

print("[smoke] done")
print(f"[smoke] total_time_sec={res.get('total_time_sec')}")
print(f"[smoke] avg_loss_final={res.get('avg_loss_final')}")
print(f"[smoke] global_metrics={res.get('eval_metrics', {}).get('global', {})}")
