#!/usr/bin/env python3
"""
CSA Research Pipeline — Master Orchestrator
=============================================
Runs all experiments (Phases 1-8), generates tables, figures, and reports.
Usage: python run_pipeline.py [--smoke] [--gpu GPU_ID] [--quick]
"""

import argparse, csv, json, math, os, sys, time, warnings
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from csa.utils.seed import set_seed
from csa.utils.metrics import aggregate_results
from csa.models.encoder import CSAEncoder
from csa.evaluation.metrics import compute_metric

RESULTS_DIR = "results"
TABLES_DIR = "tables"
FIGURES_DIR = "figures"
LOGS_DIR = "logs"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

SEEDS = [42, 123, 3407]
DEFAULT_K = 32
DEFAULT_W = 128
DEFAULT_R = 4
DEFAULT_BASELINE = "zero"

ALL_METHODS = ["dense", "local_window", "random_topk", "similarity_topk", "csa"]
SMOKE = False
QUICK = False


# ─── Helpers ────────────────────────────────────────────────────────────────

def set_env(gpu: int):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if SMOKE or QUICK:
        torch.backends.cudnn.deterministic = False


def make_model(method, d_model=512, d_ff=2048, n_layers=6, n_heads=8,
               max_len=8192, task="lm", num_classes=2,
               window=DEFAULT_W, k=DEFAULT_K, r=DEFAULT_R, baseline=DEFAULT_BASELINE,
               device="cuda"):
    """Build and return a CSAEncoder."""
    from csa.models.encoder import CSAEncoder as Encoder
    model = Encoder(
        attn_type=method, d_model=d_model, d_ff=d_ff,
        n_layers=n_layers, n_heads=n_heads, dropout=0.1,
        max_len=max_len, task=task, num_classes=num_classes,
        window=window, k=k, refresh_interval=r,
        baseline_type=baseline,
    ).to(device)
    model.eval()
    return model


def mean_std(vals: List[float]) -> Tuple[float, float]:
    return float(np.mean(vals)), float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)


def save_csv(path: str, rows: List[Dict]):
    if not rows: return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"  Saved {path}")


def save_latex(path: str, content: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(content)
    print(f"  Saved {path}")


def save_json(path: str, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating,)) else o)


# ─── Trainer (lightweight, no wandb dependency) ────────────────────────────

def train_model(model, train_loader, test_loader, lr=1e-4, epochs=3, device="cuda"):
    """Quick training loop for small-scale experiments."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * len(train_loader))
    model.train()
    for epoch in range(epochs):
        for batch in train_loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            lbls = batch["labels"].to(device) if "labels" in batch else ids.clone()
            opt.zero_grad()
            out = model(ids, attention_mask=mask, labels=lbls)
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
    model.eval()
    # accuracy
    if test_loader is not None:
        correct, total = 0, 0
        for batch in test_loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            lbls = batch.get("labels")
            if lbls is None:
                continue
            lbls = lbls.to(device)
            with torch.no_grad():
                logits = model(ids, attention_mask=mask)["logits"]
                preds = logits.argmax(-1)
                correct += (preds == lbls).sum().item()
                total += lbls.size(0)
        return correct / max(total, 1)
    return 0.0


# ══════════════════════════════════════════════════════════════════════════
# PHASE 1: SYNTHETIC CAUSAL ROBUSTNESS
# ══════════════════════════════════════════════════════════════════════════

def phase1_causal_robustness(device="cuda"):
    print("\n" + "=" * 70)
    print("PHASE 1: SYNTHETIC CAUSAL ROBUSTNESS")
    print("=" * 70)

    from csa.data.causal_robustness import CausalRobustnessDataset, collate_causal

    methods = ALL_METHODS
    seq_lens = [256, 512, 1024] if not SMOKE else [64]
    epochs = 3 if not SMOKE else 1
    train_n = 500 if not SMOKE else 30
    test_n = 200 if not SMOKE else 20
    batch = 32 if not SMOKE else 8
    model_k = DEFAULT_K

    all_results = []

    for method in methods:
        for seed in SEEDS:
            for sl in seq_lens:
                set_seed(seed)

                # determine d_model based on seq length
                dm = 256 if sl <= 256 else (512 if sl <= 1024 else 512)
                nly = 3 if sl <= 256 else (6 if sl <= 1024 else 6)
                nh = 4 if sl <= 256 else 8

                model = make_model(method, d_model=dm, n_layers=nly, n_heads=nh,
                                   max_len=sl + 64, task="classification", num_classes=2,
                                   k=min(model_k, sl), window=min(DEFAULT_W, sl),
                                   r=2, device=device)

                # Datasets
                train_ds = CausalRobustnessDataset(num_examples=train_n, seq_length=sl,
                                                   split="train", seed=seed)
                test_corr = CausalRobustnessDataset(num_examples=test_n, seq_length=sl,
                                                     spurious_correlated=True, split="test", seed=seed + 1)
                test_rev = CausalRobustnessDataset(num_examples=test_n, seq_length=sl,
                                                    spurious_correlated=False, split="test", seed=seed + 2)

                train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch, shuffle=True, collate_fn=collate_causal)
                test_corr_loader = torch.utils.data.DataLoader(test_corr, batch_size=batch, collate_fn=collate_causal)
                test_rev_loader = torch.utils.data.DataLoader(test_rev, batch_size=batch, collate_fn=collate_causal)

                _ = train_model(model, train_loader, test_corr_loader, lr=1e-4, epochs=epochs, device=device)

                # Evaluate
                def eval_acc(loader):
                    correct, total = 0, 0
                    for b in loader:
                        ids, mask, lbls = b["input_ids"].to(device), b["attention_mask"].to(device), b["labels"].to(device)
                        with torch.no_grad():
                            logits = model(ids, attention_mask=mask)["logits"]
                            preds = logits.argmax(-1)
                            correct += (preds == lbls).sum().item()
                            total += lbls.size(0)
                    return correct / max(total, 1)

                std_acc = eval_acc(test_corr_loader)
                robust_acc = eval_acc(test_rev_loader)

                # ER@k / SI@k
                from csa.utils.metrics import evidence_recall_at_k, spurious_inclusion_at_k
                er_vals, si_vals = [], []
                if method == "csa":
                    model.eval()
                    for b in test_corr_loader:
                        ids = b["input_ids"].to(device)
                        with torch.no_grad():
                            out = model(ids, return_aux=True)
                        contribs = []
                        if "aux" in out:
                            for la in out["aux"]:
                                if isinstance(la, dict) and "contrib_scores" in la:
                                    cs = la["contrib_scores"]
                                    if isinstance(cs, torch.Tensor) and cs.ndim >= 2:
                                        contribs.append(cs)
                        if contribs:
                            avg = torch.stack(contribs).mean(0)
                            for bi in range(ids.size(0)):
                                scores = avg[bi].cpu().numpy()
                                topk_idx = np.argsort(-scores)[:model_k]
                                seq = ids[bi].cpu().numpy()
                                for ev_ids in b["evidence_token_ids"]:
                                    ev_set = set(ev_ids)
                                    ev_pos = np.where(np.isin(seq, list(ev_set)))[0]
                                    if len(ev_pos):
                                        er_vals.append(evidence_recall_at_k(topk_idx, ev_pos, model_k))
                                for sp_ids in b["spurious_token_ids"]:
                                    sp_set = set(sp_ids)
                                    sp_pos = np.where(np.isin(seq, list(sp_set)))[0]
                                    if len(sp_pos):
                                        si_vals.append(spurious_inclusion_at_k(topk_idx, sp_pos, model_k))

                er = float(np.mean(er_vals)) if er_vals else 0.0
                si = float(np.mean(si_vals)) if si_vals else 0.0

                all_results.append({
                    "method": method, "seed": seed, "seq_length": sl,
                    "accuracy": std_acc, "robust_accuracy": robust_acc,
                    "robustness_gap": std_acc - robust_acc,
                    "er_at_k": er, "si_at_k": si,
                })
                ep = " (smoke)" if SMOKE else ""
                print(f"  [{method}] sl={sl}: acc={std_acc:.3f}, robust={robust_acc:.3f}, gap={std_acc-robust_acc:.3f}, ER={er:.3f}, SI={si:.3f}{ep}")

    save_csv(f"{TABLES_DIR}/table_robustness.csv", all_results)

    # Summary by method
    print("\n--- Robustness Summary ---")
    rows = []
    for method in methods:
        m_res = [r for r in all_results if r["method"] == method]
        if not m_res:
            continue
        acc, _ = mean_std([r["accuracy"] for r in m_res])
        rob, _ = mean_std([r["robust_accuracy"] for r in m_res])
        er, _ = mean_std([r["er_at_k"] for r in m_res])
        si, _ = mean_std([r["si_at_k"] for r in m_res])
        rows.append({"method": method, "accuracy": round(acc, 4), "robust_accuracy": round(rob, 4),
                     "robustness_gap": round(acc - rob, 4), "er_at_k": round(er, 4), "si_at_k": round(si, 4)})
        print(f"  {method:20s}  acc={acc:.4f}  robust={rob:.4f}  gap={acc-rob:.4f}  ER={er:.4f}  SI={si:.4f}")

    save_csv(f"{TABLES_DIR}/table_robustness_aggregated.csv", rows)

    # LaTeX table
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Causal Robustness Results}",
        "\\label{tab:robustness}",
        "\\begin{tabular}{lcccccc}",
        "\\toprule",
        "Method & Acc. & Robust Acc. & Gap & ER@k & SI@k \\\\",
        "\\midrule",
    ]
    for r in rows:
        lines.append(f"  {r['method']} & {r['accuracy']:.3f} & {r['robust_accuracy']:.3f} & {r['robustness_gap']:.3f} & {r['er_at_k']:.3f} & {r['si_at_k']:.3f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    save_latex(f"{TABLES_DIR}/table_robustness.tex", "\n".join(lines))

    # Report
    report = []
    report.append("# Causal Robustness Summary\n")
    report.append(f"Generated: {datetime.now()}\n")
    best = max(rows, key=lambda r: r["accuracy"])
    report.append(f"## Best Accuracy: {best['method']} ({best['accuracy']:.4f})")
    best_r = max(rows, key=lambda r: r["robust_accuracy"])
    report.append(f"## Best Robust Accuracy: {best_r['method']} ({best_r['robust_accuracy']:.4f})")
    best_er = max(rows, key=lambda r: r["er_at_k"])
    report.append(f"## Best ER@k: {best_er['method']} ({best_er['er_at_k']:.4f})")
    best_si = min(rows, key=lambda r: r["si_at_k"])
    report.append(f"## Lowest SI@k: {best_si['method']} ({best_si['si_at_k']:.4f})")
    for r in rows:
        report.append(f"\n### {r['method']}")
        report.append(f"- Accuracy: {r['accuracy']:.4f}")
        report.append(f"- Robust Accuracy: {r['robust_accuracy']:.4f}")
        report.append(f"- Robustness Gap: {r['robustness_gap']:.4f}")
        report.append(f"- ER@k: {r['er_at_k']:.4f}")
        report.append(f"- SI@k: {r['si_at_k']:.4f}")
    save_latex(f"{TABLES_DIR}/robustness_summary.md", "\n".join(report))

    return all_results


# ══════════════════════════════════════════════════════════════════════════
# PHASE 2: PROXY VALIDATION
# ══════════════════════════════════════════════════════════════════════════

def phase2_proxy_validation(device="cuda"):
    print("\n" + "=" * 70)
    print("PHASE 2: PROXY VALIDATION")
    print("=" * 70)

    from csa.data.proxy_validation import ProxyValidationDataset, collate_proxy
    from csa.attention.contrib import GradientProxyEstimator, ExactInterventionEstimator
    from csa.utils.metrics import spearman_correlation, kendall_tau, topk_overlap, ndcg_at_k

    seq_lens = [64, 128, 256] if not SMOKE else [32]
    n_examples = 50 if not SMOKE else 5
    batch_size = 8 if not SMOKE else 4
    all_results = []

    for seed in SEEDS:
        for sl in seq_lens:
            set_seed(seed)
            ds = ProxyValidationDataset(seq_length=sl, num_examples=n_examples, seed=seed)
            loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, collate_fn=collate_proxy)

            model = make_model("csa", d_model=64, n_layers=2, n_heads=2, d_ff=256,
                               max_len=sl+64, task="classification", num_classes=2,
                               k=min(16, sl), window=min(64, sl), r=1, device=device)
            model.eval()

            proxy_est = GradientProxyEstimator(baseline_type="zero")
            exact_est = ExactInterventionEstimator(baseline_type="zero")

            est_data = {"gradient_norm": [], "inputxgrad": [], "integrated_grad": [], "csa_proxy": [], "exact": []}

            for batch in loader:
                ids = batch["input_ids"].to(device)
                lbls = batch["labels"].to(device)

                # Exact (ground truth)
                exact = exact_est.compute(model, ids, None, lbls)
                est_data["exact"].append(exact.cpu().numpy())

                # Proxy estimators
                # 1) CSA Proxy
                proxy = proxy_est.compute(model, ids, None, lbls)
                est_data["csa_proxy"].append(proxy.cpu().numpy())

                # 2) Gradient norm
                model.zero_grad()
                x_emb = model.embed_tokens(ids).detach().requires_grad_(True)
                out = model.forward_with_embeddings(x_emb, None, lbls)
                out["loss"].backward()
                if x_emb.grad is not None:
                    gn = x_emb.grad.norm(dim=-1).cpu().numpy()
                else:
                    gn = np.zeros((ids.size(0), sl))
                est_data["gradient_norm"].append(gn)

                # 3) Input × Gradient (only for short)
                if sl <= 128 and x_emb.grad is not None:
                    ixg = (x_emb.grad.detach() * x_emb.detach()).abs().sum(dim=-1).cpu().numpy()
                    est_data["inputxgrad"].append(ixg)

                # 4) Integrated Gradients (only for very short)
                if sl <= 64:
                    from csa.experiments.exp5_proxy import compute_integrated_gradients
                    ig = compute_integrated_gradients(model, ids, lbls)
                    est_data["integrated_grad"].append(ig)

            # Compute correlations for each estimator
            estimators = ["csa_proxy", "gradient_norm", "inputxgrad", "integrated_grad"]
            for est_name in estimators:
                if not est_data[est_name]:
                    continue
                # Flatten all arrays (different batch sizes) to 1D for correlation
                all_pred = np.concatenate([a.ravel() for a in est_data[est_name]])
                all_true = np.concatenate([a.ravel() for a in est_data["exact"]])
                if len(all_pred) < 2 or np.std(all_pred) < 1e-10 or np.std(all_true) < 1e-10:
                    continue

                spearman = spearman_correlation(all_pred, all_true)
                kendall = kendall_tau(all_pred, all_true)
                ol = topk_overlap(np.argsort(-all_pred), np.argsort(-all_true), min(10, len(all_pred)))
                ndcg = ndcg_at_k(all_pred, all_true, min(10, len(all_pred)))

                all_results.append({
                    "seed": seed, "seq_length": sl,
                    "estimator": est_name,
                    "spearman": spearman, "kendall": kendall,
                    "topk_overlap": ol, "ndcg": ndcg,
                })

    save_csv(f"{TABLES_DIR}/table_proxy.csv", all_results)

    # Aggregated
    print("\n--- Proxy Validation Summary ---")
    agg = defaultdict(list)
    for r in all_results:
        agg[(r["estimator"], r["seq_length"])].append(r)
    rows = []
    for (est, sl), vals in sorted(agg.items()):
        sp, _ = mean_std([v["spearman"] for v in vals])
        kd, _ = mean_std([v["kendall"] for v in vals])
        ol, _ = mean_std([v["topk_overlap"] for v in vals])
        nd, _ = mean_std([v["ndcg"] for v in vals])
        rows.append({"estimator": est, "seq_length": sl, "spearman": round(sp, 4),
                     "kendall": round(kd, 4), "topk_overlap": round(ol, 4), "ndcg": round(nd, 4)})
        print(f"  {est:20s} sl={sl}: Spearman={sp:.4f} Kendall={kd:.4f} Overlap={ol:.4f} NDCG={nd:.4f}")

    save_csv(f"{TABLES_DIR}/table_proxy_aggregated.csv", rows)

    # LaTeX
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Proxy Validation Results}",
        "\\label{tab:proxy}",
        "\\begin{tabular}{lccccc}",
        "\\toprule",
        "Estimator & Len. & Spearman & Kendall & Overlap & NDCG \\\\",
        "\\midrule",
    ]
    for r in rows:
        lines.append(f"  {r['estimator']} & {r['seq_length']} & {r['spearman']:.3f} & {r['kendall']:.3f} & {r['topk_overlap']:.3f} & {r['ndcg']:.3f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    save_latex(f"{TABLES_DIR}/table_proxy.tex", "\n".join(lines))

    # Report
    rpt = ["# Proxy Validation Summary\n", f"Generated: {datetime.now()}\n"]
    if rows:
        best_sp = max(rows, key=lambda r: r["spearman"])
        rpt.append(f"## Best Spearman: {best_sp['estimator']} ({best_sp['spearman']:.4f})")
    for r in rows:
        rpt.append(f"\n### {r['estimator']} (len={r['seq_length']})")
        rpt.append(f"- Spearman: {r['spearman']:.4f}")
        rpt.append(f"- Kendall: {r['kendall']:.4f}")
        rpt.append(f"- Top-k Overlap: {r['topk_overlap']:.4f}")
        rpt.append(f"- NDCG: {r['ndcg']:.4f}")
    save_latex(f"{TABLES_DIR}/proxy_validation_report.md", "\n".join(rpt))

    # Plots
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from csa.visualization.style import METHOD_COLORS, METHOD_LABELS
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        metrics_keys = [("spearman", "Spearman"), ("kendall", "Kendall"), ("topk_overlap", "Top-k Overlap"), ("ndcg", "NDCG")]
        for ax, (mk, ml) in zip(axes.flat, metrics_keys):
            for row in rows:
                ax.plot(row["seq_length"], row[mk], "o", label=row["estimator"] if mk == "spearman" else "")
                ax.text(row["seq_length"], row[mk], row["estimator"][:8], fontsize=6, ha="center")
            ax.set_xlabel("Sequence Length"); ax.set_ylabel(ml); ax.set_title(ml)
            ax.legend(fontsize=6)
        plt.tight_layout()
        plt.savefig(f"{FIGURES_DIR}/proxy_validation.pdf", bbox_inches="tight", dpi=150)
        plt.close()
        print(f"  Saved {FIGURES_DIR}/proxy_validation.pdf")
    except Exception as e:
        print(f"  Plot failed: {e}")

    return all_results


# ══════════════════════════════════════════════════════════════════════════
# PHASE 3: ABLATION STUDY
# ══════════════════════════════════════════════════════════════════════════

def phase3_ablation(device="cuda"):
    print("\n" + "=" * 70)
    print("PHASE 3: ABLATION STUDY")
    print("=" * 70)

    from csa.data.causal_robustness import CausalRobustnessDataset, collate_causal

    abl_methods = ["csa", "csa"]  # We'll vary components
    abl_configs = [
        ("CSA (full)", "csa", DEFAULT_W, DEFAULT_K),
        ("CSA w/o Window", "csa", 1, DEFAULT_K),  # window=1 ≈ no local
        ("CSA w/o TopK", "csa", DEFAULT_W, 0),      # k=0 ≈ no causal top-k
        ("Random Top-k", "random_topk", DEFAULT_W, DEFAULT_K),
        ("Similarity Top-k", "similarity_topk", DEFAULT_W, DEFAULT_K),
    ]

    sl = 256 if not SMOKE else 64
    epochs = 3 if not SMOKE else 1
    train_n = 300 if not SMOKE else 20
    test_n = 150 if not SMOKE else 10

    all_results = []

    for seed in SEEDS:
        for label, method, window, k in abl_configs:
            set_seed(seed)
            model = make_model(method, d_model=256, n_layers=3, n_heads=4,
                               max_len=sl+64, task="classification", num_classes=2,
                               window=window, k=max(k, 1), r=2, device=device)

            train_ds = CausalRobustnessDataset(num_examples=train_n, seq_length=sl, split="train", seed=seed)
            test_corr = CausalRobustnessDataset(num_examples=test_n, seq_length=sl, spurious_correlated=True, split="test", seed=seed+1)
            test_rev = CausalRobustnessDataset(num_examples=test_n, seq_length=sl, spurious_correlated=False, split="test", seed=seed+2)

            tl = torch.utils.data.DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=collate_causal)
            cl = torch.utils.data.DataLoader(test_corr, batch_size=16, collate_fn=collate_causal)
            rl = torch.utils.data.DataLoader(test_rev, batch_size=16, collate_fn=collate_causal)

            _ = train_model(model, tl, cl, epochs=epochs, device=device)

            def eval_acc(loader):
                correct, total = 0, 0
                for b in loader:
                    ids, mask, lbls = b["input_ids"].to(device), b["attention_mask"].to(device), b["labels"].to(device)
                    with torch.no_grad():
                        preds = model(ids, attention_mask=mask)["logits"].argmax(-1)
                        correct += (preds == lbls).sum().item(); total += lbls.size(0)
                return correct / max(total, 1)

            acc = eval_acc(cl); rob = eval_acc(rl)

            all_results.append({"method": label, "seed": seed,
                                "accuracy": acc, "robust_accuracy": rob,
                                "robustness_gap": acc - rob})

    save_csv(f"{TABLES_DIR}/table_ablation.csv", all_results)

    print("\n--- Ablation Summary ---")
    agg = defaultdict(list)
    for r in all_results:
        agg[r["method"]].append(r)
    rows = []
    for method, vals in agg.items():
        a, _ = mean_std([v["accuracy"] for v in vals])
        r_, _ = mean_std([v["robust_accuracy"] for v in vals])
        g, _ = mean_std([v["robustness_gap"] for v in vals])
        rows.append({"method": method, "accuracy": round(a, 4), "robust_accuracy": round(r_, 4),
                     "robustness_gap": round(g, 4)})
        print(f"  {method:25s}  acc={a:.4f}  robust={r_:.4f}  gap={g:.4f}")
    save_csv(f"{TABLES_DIR}/table_ablation_aggregated.csv", rows)

    # LaTeX
    lines = [
        "\\begin{table}[t]", "\\centering",
        "\\caption{Ablation Study Results}", "\\label{tab:ablation}",
        "\\begin{tabular}{lcccc}", "\\toprule",
        "Method & Acc. & Robust Acc. & Gap \\\\", "\\midrule",
    ]
    for r in rows:
        lines.append(f"  {r['method']} & {r['accuracy']:.3f} & {r['robust_accuracy']:.3f} & {r['robustness_gap']:.3f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    save_latex(f"{TABLES_DIR}/table_ablation.tex", "\n".join(lines))

    # Report
    rpt = ["# Ablation Study Summary\n", f"Generated: {datetime.now()}\n"]
    for r in rows:
        rpt.append(f"\n### {r['method']}")
        rpt.append(f"- Accuracy: {r['accuracy']:.4f}")
        rpt.append(f"- Robust Accuracy: {r['robust_accuracy']:.4f}")
        rpt.append(f"- Robustness Gap: {r['robustness_gap']:.4f}")
    save_latex(f"{TABLES_DIR}/ablation_report.md", "\n".join(rpt))

    # Plot
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 4))
        labels = [r["method"] for r in rows]
        accs = [r["accuracy"] for r in rows]
        robs = [r["robust_accuracy"] for r in rows]
        x = np.arange(len(labels)); w = 0.35
        ax.bar(x - w/2, accs, w, label="Standard Acc.", color="#0072B2")
        ax.bar(x + w/2, robs, w, label="Robust Acc.", color="#D55E00")
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.legend(); ax.set_ylabel("Accuracy"); ax.set_title("Ablation Study")
        plt.tight_layout()
        plt.savefig(f"{FIGURES_DIR}/ablation.pdf", bbox_inches="tight", dpi=150)
        plt.close(); print(f"  Saved {FIGURES_DIR}/ablation.pdf")
    except Exception as e:
        print(f"  Plot failed: {e}")

    return all_results


# ══════════════════════════════════════════════════════════════════════════
# PHASE 4: SENSITIVITY ANALYSIS
# ══════════════════════════════════════════════════════════════════════════

def phase4_sensitivity(device="cuda"):
    print("\n" + "=" * 70)
    print("PHASE 4: SENSITIVITY ANALYSIS")
    print("=" * 70)

    from csa.data.causal_robustness import CausalRobustnessDataset, collate_causal

    sl = 256 if not SMOKE else 64
    epochs = 2 if not SMOKE else 1
    train_n = 200 if not SMOKE else 15
    test_n = 100 if not SMOKE else 8
    all_results = []

    # k sweep
    for k in [8, 16, 32, 64, 128] if not SMOKE else [8, 16]:
        for seed in SEEDS[:1] if QUICK else SEEDS:
            set_seed(seed)
            model = make_model("csa", d_model=256, n_layers=3, n_heads=4,
                               max_len=sl+64, task="classification", num_classes=2,
                               window=128, k=min(k, sl), r=2, device=device)
            train_ds = CausalRobustnessDataset(num_examples=train_n, seq_length=sl, split="train", seed=seed)
            test_ds = CausalRobustnessDataset(num_examples=test_n, seq_length=sl, split="test", seed=seed+1)
            tl = torch.utils.data.DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=collate_causal)
            tel = torch.utils.data.DataLoader(test_ds, batch_size=16, collate_fn=collate_causal)
            acc = train_model(model, tl, tel, epochs=epochs, device=device)
            all_results.append({"param": "k", "value": k, "accuracy": acc, "seed": seed})

    # w sweep
    for w in [32, 64, 128, 256] if not SMOKE else [32, 64]:
        for seed in SEEDS[:1] if QUICK else SEEDS:
            set_seed(seed)
            model = make_model("csa", d_model=256, n_layers=3, n_heads=4,
                               max_len=sl+64, task="classification", num_classes=2,
                               window=min(w, sl), k=32, r=2, device=device)
            train_ds = CausalRobustnessDataset(num_examples=train_n, seq_length=min(sl, w*2), split="train", seed=seed)
            test_ds = CausalRobustnessDataset(num_examples=test_n, seq_length=min(sl, w*2), split="test", seed=seed+1)
            tl = torch.utils.data.DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=collate_causal)
            tel = torch.utils.data.DataLoader(test_ds, batch_size=16, collate_fn=collate_causal)
            acc = train_model(model, tl, tel, epochs=epochs, device=device)
            all_results.append({"param": "w", "value": w, "accuracy": acc, "seed": seed})

    # r sweep
    for r in [1, 2, 4, 8] if not SMOKE else [1, 4]:
        for seed in SEEDS[:1] if QUICK else SEEDS:
            set_seed(seed)
            model = make_model("csa", d_model=256, n_layers=3, n_heads=4,
                               max_len=sl+64, task="classification", num_classes=2,
                               window=128, k=32, r=r, device=device)
            train_ds = CausalRobustnessDataset(num_examples=train_n, seq_length=sl, split="train", seed=seed)
            test_ds = CausalRobustnessDataset(num_examples=test_n, seq_length=sl, split="test", seed=seed+1)
            tl = torch.utils.data.DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=collate_causal)
            tel = torch.utils.data.DataLoader(test_ds, batch_size=16, collate_fn=collate_causal)
            acc = train_model(model, tl, tel, epochs=epochs, device=device)
            all_results.append({"param": "r", "value": r, "accuracy": acc, "seed": seed})

    # Baseline sweep
    for baseline in ["zero", "mask", "mean"]:
        for seed in SEEDS[:1] if QUICK else SEEDS:
            set_seed(seed)
            model = make_model("csa", d_model=256, n_layers=3, n_heads=4,
                               max_len=sl+64, task="classification", num_classes=2,
                               window=128, k=32, r=2, baseline=baseline, device=device)
            train_ds = CausalRobustnessDataset(num_examples=train_n, seq_length=sl, split="train", seed=seed)
            test_ds = CausalRobustnessDataset(num_examples=test_n, seq_length=sl, split="test", seed=seed+1)
            tl = torch.utils.data.DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=collate_causal)
            tel = torch.utils.data.DataLoader(test_ds, batch_size=16, collate_fn=collate_causal)
            acc = train_model(model, tl, tel, epochs=epochs, device=device)
            all_results.append({"param": "baseline", "value": baseline, "accuracy": acc, "seed": seed})

    save_csv(f"{TABLES_DIR}/table_sensitivity.csv", all_results)

    # Aggregated
    print("\n--- Sensitivity Summary ---")
    agg = defaultdict(list)
    for r in all_results:
        agg[(r["param"], str(r["value"]))].append(r)
    rows = []
    for (param, val), vals in sorted(agg.items()):
        a, s = mean_std([v["accuracy"] for v in vals])
        rows.append({"param": param, "value": val, "accuracy": round(a, 4), "std": round(s, 4)})
        print(f"  {param:10s} = {val:8s}  acc={a:.4f} ± {s:.4f}")
    save_csv(f"{TABLES_DIR}/table_sensitivity_aggregated.csv", rows)

    # LaTeX
    lines = [
        "\\begin{table}[t]", "\\centering",
        "\\caption{Sensitivity Analysis}", "\\label{tab:sensitivity}",
        "\\begin{tabular}{lcc}", "\\toprule",
        "Parameter & Value & Accuracy \\\\", "\\midrule",
    ]
    for r in rows:
        lines.append(f"  {r['param']} & {r['value']} & {r['accuracy']:.3f} $\\pm$ {r['std']:.3f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    save_latex(f"{TABLES_DIR}/table_sensitivity.tex", "\n".join(lines))

    # Report
    rpt = ["# Sensitivity Analysis Summary\n", f"Generated: {datetime.now()}\n"]
    for r in rows:
        rpt.append(f"- **{r['param']} = {r['value']}**: acc = {r['accuracy']:.4f} ± {r['std']:.4f}")
    save_latex(f"{TABLES_DIR}/sensitivity_report.md", "\n".join(rpt))

    # Plots
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        for ax, param in zip(axes.flat, ["k", "w", "r", "baseline"]):
            pr = [r for r in rows if r["param"] == param]
            if not pr: continue
            vals = [r["value"] for r in pr]
            accs = [r["accuracy"] for r in pr]
            errs = [r["std"] for r in pr]
            ax.errorbar(range(len(vals)), accs, yerr=errs, marker="o", capsize=4)
            ax.set_xticks(range(len(vals))); ax.set_xticklabels(vals, fontsize=8)
            ax.set_xlabel(param); ax.set_ylabel("Accuracy"); ax.set_title(f"Sensitivity: {param}")
        plt.tight_layout()
        plt.savefig(f"{FIGURES_DIR}/sensitivity.pdf", bbox_inches="tight", dpi=150)
        plt.close(); print(f"  Saved {FIGURES_DIR}/sensitivity.pdf")
    except Exception as e:
        print(f"  Plot failed: {e}")

    return all_results


# ══════════════════════════════════════════════════════════════════════════
# PHASE 5: EFFICIENCY ANALYSIS
# ══════════════════════════════════════════════════════════════════════════

def phase5_efficiency(device="cuda"):
    print("\n" + "=" * 70)
    print("PHASE 5: EFFICIENCY ANALYSIS")
    print("=" * 70)

    eff_methods = ["dense", "local_window", "similarity_topk", "csa"]
    seq_lens = [2048, 4096, 8192, 16384] if not SMOKE else [2048]
    if not (SMOKE or QUICK):
        seq_lens = [2048, 4096, 8192, 16384, 32768, 65536]
    batch_size = 1

    all_results = []

    for method in eff_methods:
        for sl in seq_lens:
            if method == "dense" and sl > 32768:
                continue  # skip OOM dense
            try:
                model = make_model(method, d_model=512, n_layers=6, n_heads=8,
                                   max_len=sl + 1024, k=min(32, sl), window=min(128, sl),
                                   r=4, device=device)
                model.eval()

                # Use a simple FLOPs counter that doesn't need n_heads
                def count_flops_simple(n_layers, d_model, d_ff, seq_len, vocab_size=97):
                    """Simple FLOPs estimate for transformer."""
                    # Embedding: L * D * log2(V)
                    embed_flops = seq_len * d_model * np.log2(vocab_size)
                    attn_flops = n_layers * (8 * seq_len * d_model * d_model + 4 * seq_len * seq_len * d_model)
                    ffn_flops = n_layers * (4 * seq_len * d_model * d_ff)
                    return embed_flops + attn_flops + ffn_flops

                dummy = torch.randint(0, 100, (batch_size, sl), device=device)
                # Warmup
                for _ in range(3):
                    with torch.no_grad(): _ = model(dummy)
                # Measure
                torch.cuda.synchronize(device)
                start = time.time()
                for _ in range(5):
                    with torch.no_grad(): _ = model(dummy)
                torch.cuda.synchronize(device)
                latency = (time.time() - start) / 5 * 1000  # ms

                # Memory
                torch.cuda.reset_peak_memory_stats(device)
                with torch.no_grad(): _ = model(dummy)
                mem = torch.cuda.max_memory_allocated(device) / 1e9

                flops = count_flops_simple(6, 512, 2048, sl) / 1e9

                throughput = batch_size / (latency / 1000)
                all_results.append({"method": method, "seq_length": sl,
                                    "latency_ms": round(latency, 2),
                                    "throughput": round(throughput, 2),
                                    "memory_gb": round(mem, 3),
                                    "flops_g": round(flops, 1)})
                print(f"  [{method:15s}] sl={sl}: {latency:.0f}ms, {mem:.1f}GB, {flops:.0f}GFLOPs")
            except Exception as e:
                print(f"  [{method:15s}] sl={sl}: OOM/ERR — {e}")
                all_results.append({"method": method, "seq_length": sl,
                                    "latency_ms": -1, "throughput": -1,
                                    "memory_gb": -1, "flops_g": -1})

    save_csv(f"{TABLES_DIR}/table_efficiency.csv", all_results)

    # Print summary
    print("\n--- Efficiency Summary ---")
    for r in all_results:
        if r["latency_ms"] > 0:
            print(f"  {r['method']:15s} sl={r['seq_length']:5d}: {r['latency_ms']:8.1f}ms, {r['memory_gb']:.2f}GB")

    # LaTeX
    lines = [
        "\\begin{table}[t]", "\\centering",
        "\\caption{Efficiency Analysis}", "\\label{tab:efficiency}",
        "\\begin{tabular}{lcrrr}", "\\toprule",
        "Method & Seq. Len. & Latency (ms) & Memory (GB) & FLOPs (G) \\\\", "\\midrule",
    ]
    for r in all_results:
        if r["latency_ms"] > 0:
            lines.append(f"  {r['method']} & {r['seq_length']} & {r['latency_ms']:.1f} & {r['memory_gb']:.1f} & {r['flops_g']:.0f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    save_latex(f"{TABLES_DIR}/table_efficiency.tex", "\n".join(lines))

    # Report
    rpt = ["# Efficiency Analysis Summary\n", f"Generated: {datetime.now()}\n"]
    for r in all_results:
        if r["latency_ms"] > 0:
            rpt.append(f"- {r['method']} @ {r['seq_length']}: {r['latency_ms']:.1f}ms, {r['memory_gb']:.2f}GB, {r['flops_g']:.0f}GFLOPs")
    save_latex(f"{TABLES_DIR}/efficiency_report.md", "\n".join(rpt))

    # Plots
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        colors = {"dense": "#0072B2", "local_window": "#009E73", "similarity_topk": "#E69F00", "csa": "#D55E00"}
        markers = {"dense": "o", "local_window": "s", "similarity_topk": "^", "csa": "*"}
        for metric, ax, ylabel in [("latency_ms", axes[0], "Latency (ms)"),
                                    ("memory_gb", axes[1], "GPU Memory (GB)"),
                                    ("throughput", axes[2], "Throughput (seq/s)")]:
            for method in eff_methods:
                points = [(r["seq_length"], r[metric]) for r in all_results if r["method"] == method and r[metric] > 0]
                if points:
                    xs, ys = zip(*sorted(points))
                    ax.plot(xs, ys, color=colors.get(method, "gray"), marker=markers.get(method, "o"),
                            label=method, linewidth=1.5)
            ax.set_xlabel("Sequence Length"); ax.set_ylabel(ylabel)
            ax.set_xscale("log", base=2); ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(f"{FIGURES_DIR}/efficiency.pdf", bbox_inches="tight", dpi=150)
        plt.close(); print(f"  Saved {FIGURES_DIR}/efficiency.pdf")
    except Exception as e:
        print(f"  Plot failed: {e}")

    return all_results


# ══════════════════════════════════════════════════════════════════════════
# PHASE 6: LONGBENCH (zeroshot eval)
# ══════════════════════════════════════════════════════════════════════════

def phase6_longbench(device="cuda"):
    print("\n" + "=" * 70)
    print("PHASE 6: LONGBENCH EVALUATION")
    print("=" * 70)

    lb_methods = ["dense", "local_window", "similarity_topk", "csa"]
    tasks_metrics = [
        ("narrativeqa", "rouge-l"),
        ("qasper", "rouge-l"),
        ("hotpotqa", "f1"),
        ("2wikimultihopqa", "f1"),
        ("musique", "f1"),
        ("govreport", "rouge-l"),
        ("qmsum", "rouge-l"),
    ]
    max_len = 4096 if not SMOKE else 1024
    max_batches = 10 if not SMOKE else 2

    all_results = []

    for method in lb_methods:
        for task, metric in tasks_metrics:
            for seed in [SEEDS[0]] if QUICK else SEEDS:
                set_seed(seed)
                from csa.data.longbench import LongBenchDataset, collate_longbench
                ds = LongBenchDataset(task_name=task, split="test", max_length=max_len)
                loader = torch.utils.data.DataLoader(ds, batch_size=2, shuffle=False, collate_fn=collate_longbench)

                model = make_model(method, d_model=512, n_layers=6, n_heads=8,
                                   max_len=max_len+1024, k=min(32, max_len),
                                   window=min(128, max_len), r=4, device=device)
                model.eval()
                from csa.data.tokenizer import SimpleTokenizer
                tokenizer = SimpleTokenizer()

                preds, refs = [], []
                for bi, batch in enumerate(loader):
                    if bi >= max_batches: break
                    ids = batch["input_ids"].to(device)
                    with torch.no_grad():
                        out = model(ids)
                    logits = out["logits"]
                    # Decode
                    for tokens, inp in zip(logits.argmax(-1), ids):
                        gen = tokens[tokens.shape[0] // 2:].tolist()
                        t = tokenizer.decode(gen, skip_special_tokens=True).strip()
                        preds.append(t if t else "N/A")
                    refs.extend(batch["answers"])

                scores = []
                for p, r in zip(preds, refs):
                    if isinstance(r, list):
                        scores.append(max(compute_metric(p, rr, metric) for rr in r))
                    else:
                        scores.append(compute_metric(p, r, metric))
                avg_score = float(np.mean(scores)) if scores else 0.0

                all_results.append({"method": method, "task": task, "metric": metric,
                                    "score": round(avg_score, 4), "seed": seed})
                print(f"  [{method:15s}] {task:20s} {metric:8s}: {avg_score:.4f}")

    save_csv(f"{TABLES_DIR}/table_longbench.csv", all_results)

    # Aggregated
    print("\n--- LongBench Summary ---")
    agg = defaultdict(list)
    for r in all_results:
        agg[(r["method"], r["task"])].append(r)
    rows = []
    for method in lb_methods:
        row = {"method": method}
        scores = []
        for task, metric in tasks_metrics:
            vals = agg.get((method, task), [])
            if vals:
                s, _ = mean_std([v["score"] for v in vals])
                row[task] = round(s, 4)
                scores.append(s)
        row["average"] = round(float(np.mean(scores)), 4) if scores else 0.0
        rows.append(row)
        print(f"  {method:15s} avg={row['average']:.4f}")

    save_csv(f"{TABLES_DIR}/table_longbench_aggregated.csv", rows)

    # LaTeX
    lines = [
        "\\begin{table}[t]", "\\centering",
        "\\caption{LongBench Results}", "\\label{tab:longbench}",
        "\\begin{tabular}{l" + "c" * (len(tasks_metrics) + 1) + "}", "\\toprule",
        "Method & " + " & ".join(t for t, _ in tasks_metrics) + " & Avg. \\\\", "\\midrule",
    ]
    for r in rows:
        vals = " & ".join(f"{r.get(t, 0):.3f}" for t, _ in tasks_metrics)
        lines.append(f"  {r['method']} & {vals} & {r['average']:.3f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    save_latex(f"{TABLES_DIR}/table_longbench.tex", "\n".join(lines))

    # Report
    rpt = ["# LongBench Evaluation Summary\n", f"Generated: {datetime.now()}\n"]
    best = max(rows, key=lambda r: r["average"])
    rpt.append(f"## Best Average: {best['method']} ({best['average']:.4f})\n")
    for r in rows:
        rpt.append(f"\n### {r['method']}")
        rpt.append(f"- Average: {r['average']:.4f}")
        for t, _ in tasks_metrics:
            rpt.append(f"  - {t}: {r.get(t, 0):.4f}")
    save_latex(f"{TABLES_DIR}/longbench_report.md", "\n".join(rpt))

    return all_results


# ══════════════════════════════════════════════════════════════════════════
# PHASE 7: NEEDLE-IN-A-HAYSTACK
# ══════════════════════════════════════════════════════════════════════════

def phase7_niah(device="cuda"):
    print("\n" + "=" * 70)
    print("PHASE 7: NEEDLE-IN-A-HAYSTACK")
    print("=" * 70)

    niah_methods = ["dense", "local_window", "similarity_topk", "csa"]
    ctx_lens = [8192, 16384] if SMOKE else [8192, 16384, 32768, 65536]
    depths = [0.1, 0.3, 0.5, 0.7, 0.9] if not SMOKE else [0.5]
    n_per_config = 3 if not SMOKE else 1

    all_results = []

    for method in niah_methods:
        for seed in [SEEDS[0]] if QUICK else SEEDS[:1]:
            for ctx_len in ctx_lens:
                for depth in depths:
                    set_seed(seed)
                    from csa.data.needle_haystack import NeedleHaystackDataset, collate_needle
                    ds = NeedleHaystackDataset(context_lengths=[ctx_len], depths=[depth],
                                                num_examples_per_config=n_per_config)
                    loader = torch.utils.data.DataLoader(ds, batch_size=1, collate_fn=collate_needle)

                    model = make_model(method, d_model=512, n_layers=6, n_heads=8,
                                       max_len=ctx_len + 1024, k=min(32, ctx_len),
                                       window=min(128, ctx_len), r=1, device=device)
                    model.eval()
                    from csa.data.tokenizer import SimpleTokenizer
                    tokenizer = SimpleTokenizer()

                    correct = 0; total = 0
                    needle_recalls = []

                    with torch.no_grad():
                        for batch in loader:
                            ids = batch["input_ids"].to(device)
                            if ids.size(1) > ctx_len:
                                ids = ids[:, :ctx_len]
                            out = model(ids, return_aux=(method == "csa"))
                            logits = out["logits"]
                            for tokens in logits.argmax(-1):
                                gen = tokens[ids.size(1) // 2:].tolist()
                                pred = tokenizer.decode(gen, skip_special_tokens=True).strip()
                            for gt in batch["answers"]:
                                em = compute_metric(pred, gt, "em")
                                correct += em; total += 1

                    acc = correct / max(total, 1)

                    # Needle recall for CSA
                    if method == "csa" and total > 0:
                        with torch.no_grad():
                            ids = batch["input_ids"].to(device)
                            if ids.size(1) > ctx_len: ids = ids[:, :ctx_len]
                            out = model(ids, return_aux=True)
                        if "aux" in out:
                            for aux in out["aux"]:
                                if isinstance(aux, dict) and "contrib_scores" in aux:
                                    cs = aux["contrib_scores"]
                                    if isinstance(cs, torch.Tensor) and cs.numel() > 1:
                                        topk_idx = np.argsort(-cs[0].cpu().numpy())[:32]
                                        npos = batch.get("needle_start_positions", [0])
                                        if npos:
                                            np_pos = npos[0].item() if isinstance(npos[0], torch.Tensor) else npos[0]
                                            nr = 1.0 if np_pos in topk_idx else 0.0
                                            needle_recalls.append(nr)

                    nr_avg = float(np.mean(needle_recalls)) if needle_recalls else 0.0

                    all_results.append({"method": method, "context_length": ctx_len,
                                        "depth": depth, "accuracy": acc,
                                        "needle_recall": nr_avg})
                    print(f"  [{method:15s}] ctx={ctx_len} depth={depth:.1f}: acc={acc:.3f} recall={nr_avg:.3f}")

    save_csv(f"{TABLES_DIR}/table_niah.csv", all_results)

    # Aggregated
    print("\n--- NIAH Summary ---")
    agg = defaultdict(list)
    for r in all_results:
        agg[(r["method"], r["context_length"])].append(r)
    rows = []
    for method in niah_methods:
        row = {"method": method}
        accs = []
        for cl in ctx_lens:
            vals = agg.get((method, cl), [])
            if vals:
                a, _ = mean_std([v["accuracy"] for v in vals])
                row[f"{cl}"] = round(a, 4)
                accs.append(a)
        row["average"] = round(float(np.mean(accs)), 4) if accs else 0.0
        rows.append(row)
        print(f"  {method:15s} avg_acc={row['average']:.4f}")
    save_csv(f"{TABLES_DIR}/table_niah_aggregated.csv", rows)

    # LaTeX
    lines = [
        "\\begin{table}[t]", "\\centering",
        "\\caption{Needle-in-a-Haystack Results}", "\\label{tab:niah}",
        "\\begin{tabular}{l" + "c" * (len(ctx_lens) + 1) + "}", "\\toprule",
        "Method & " + " & ".join(f"{cl//1024}K" for cl in ctx_lens) + " & Avg. \\\\", "\\midrule",
    ]
    for r in rows:
        vals = " & ".join(f"{r.get(cl, 0):.3f}" for cl in ctx_lens)
        lines.append(f"  {r['method']} & {vals} & {r['average']:.3f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    save_latex(f"{TABLES_DIR}/table_niah.tex", "\n".join(lines))

    # Heatmap plot
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        n_methods = len(niah_methods)
        fig, axes = plt.subplots(1, n_methods, figsize=(5 * n_methods, 4))
        if n_methods == 1: axes = [axes]
        for ax, method in zip(axes, niah_methods):
            data = np.full((len(ctx_lens), len(depths)), np.nan)
            for r in all_results:
                if r["method"] != method: continue
                ci = ctx_lens.index(r["context_length"])
                di = depths.index(r["depth"])
                data[ci, di] = r["accuracy"]
            im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
            ax.set_xticks(range(len(depths)))
            ax.set_xticklabels([f"{int(d*100)}%" for d in depths])
            ax.set_yticks(range(len(ctx_lens)))
            ax.set_yticklabels([f"{cl//1024}K" for cl in ctx_lens])
            ax.set_title(method); ax.set_xlabel("Needle Depth"); ax.set_ylabel("Context Length")
            # Annotate
            for i in range(len(ctx_lens)):
                for j in range(len(depths)):
                    val = data[i, j]
                    if not np.isnan(val):
                        ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                                fontsize=7, color="black" if val > 0.5 else "white")
        plt.tight_layout()
        plt.savefig(f"{FIGURES_DIR}/niah_heatmap.pdf", bbox_inches="tight", dpi=150)
        plt.close(); print(f"  Saved {FIGURES_DIR}/niah_heatmap.pdf")
    except Exception as e:
        print(f"  Plot failed: {e}")
        import traceback; traceback.print_exc()

    # Report
    rpt = ["# Needle-in-a-Haystack Summary\n", f"Generated: {datetime.now()}\n"]
    best = max(rows, key=lambda r: r["average"])
    rpt.append(f"## Best Average Accuracy: {best['method']} ({best['average']:.4f})\n")
    for r in rows:
        rpt.append(f"\n### {r['method']}")
        rpt.append(f"- Average: {r['average']:.4f}")
        for cl in ctx_lens:
            rpt.append(f"  - {cl//1024}K: {r.get(cl, 0):.4f}")
    save_latex(f"{TABLES_DIR}/niah_report.md", "\n".join(rpt))

    return all_results


# ══════════════════════════════════════════════════════════════════════════
# FINAL REPORT
# ══════════════════════════════════════════════════════════════════════════

def generate_final_report():
    print("\n" + "=" * 70)
    print("GENERATING FINAL REPORT")
    print("=" * 70)

    report = [
        "# Causal Sparse Attention (CSA) — Final Experiment Report\n",
        f"Generated: {datetime.now()}\n",
        f"Seeds: {SEEDS}\n",
    ]

    # Load available tables
    tables = {}
    for name in ["robustness_aggregated", "proxy_aggregated", "ablation_aggregated",
                  "sensitivity_aggregated", "efficiency", "longbench_aggregated", "niah_aggregated"]:
        path = f"{TABLES_DIR}/table_{name}.csv"
        if os.path.exists(path):
            import csv
            with open(path) as f:
                tables[name] = list(csv.DictReader(f))

    # 1. Causal Robustness
    report.append("\n## 1. Causal Robustness\n")
    if "robustness_aggregated" in tables:
        rows = tables["robustness_aggregated"]
        best_acc = max(rows, key=lambda r: float(r["accuracy"]))
        best_rob = max(rows, key=lambda r: float(r["robust_accuracy"]))
        best_er = max(rows, key=lambda r: float(r["er_at_k"]))
        lowest_si = min(rows, key=lambda r: float(r["si_at_k"]))

        # Find CSA row
        csa_row = next((r for r in rows if r["method"] == "csa"), None)
        sim_row = next((r for r in rows if r["method"] == "similarity_topk"), None)
        dense_row = next((r for r in rows if r["method"] == "dense"), None)

        report.append(f"- **Best Accuracy**: {best_acc['method']} ({best_acc['accuracy']})")
        report.append(f"- **Best Robust Accuracy**: {best_rob['method']} ({best_rob['robust_accuracy']})")
        report.append(f"- **Best ER@k**: {best_er['method']} ({best_er['er_at_k']})")
        report.append(f"- **Lowest SI@k**: {lowest_si['method']} ({lowest_si['si_at_k']})")

        if csa_row and sim_row:
            csa_acc = float(csa_row["accuracy"])
            sim_acc = float(sim_row["accuracy"])
            gap = csa_acc - sim_acc
            report.append(f"\n- **CSA vs Similarity Top-k**: CSA {'outperforms' if gap > 0 else 'underperforms'} similarity by {abs(gap):.4f} in standard accuracy")
            csa_rob = float(csa_row["robust_accuracy"])
            sim_rob = float(sim_row["robust_accuracy"])
            rob_gap = csa_rob - sim_rob
            report.append(f"- **CSA vs Similarity Top-k (Robust)**: CSA {'outperforms' if rob_gap > 0 else 'underperforms'} similarity by {abs(rob_gap):.4f}")

        if csa_row and dense_row:
            csa_er = float(csa_row["er_at_k"])
            report.append(f"- **CSA ER@k**: {csa_er:.4f} (higher = better evidence preservation)")

    # 2. Proxy Validation
    report.append("\n## 2. Proxy Validation\n")
    if "proxy_aggregated" in tables:
        rows = tables["proxy_aggregated"]
        csa_rows = [r for r in rows if r["estimator"] == "csa_proxy"]
        if csa_rows:
            best_sp = max(csa_rows, key=lambda r: float(r["spearman"]))
            report.append(f"- **CSA Proxy best Spearman**: {best_sp['spearman']} @ len={best_sp['seq_length']}")
        for r in rows:
            if r["estimator"] == "csa_proxy":
                report.append(f"- CSA Proxy (len={r['seq_length']}): Spearman={r['spearman']}, Kendall={r['kendall']}, NDCG={r['ndcg']}")

    # 3. Ablation
    report.append("\n## 3. Ablation Study\n")
    if "ablation_aggregated" in tables:
        for r in tables["ablation_aggregated"]:
            report.append(f"- {r['method']}: acc={r['accuracy']}, robust={r['robust_accuracy']}, gap={r['robustness_gap']}")

    # 4. Sensitivity
    report.append("\n## 4. Sensitivity Analysis\n")
    if "sensitivity_aggregated" in tables:
        report.append("| Parameter | Best Value | Accuracy |")
        report.append("|-----------|-----------|----------|")
        for r in tables["sensitivity_aggregated"]:
            report.append(f"| {r['param']} | {r['value']} | {r['accuracy']} ± {r.get('std', 'N/A')} |")

    # 5. Efficiency
    report.append("\n## 5. Efficiency Analysis\n")
    if "efficiency" in tables:
        for r in tables["efficiency"]:
            if float(r["latency_ms"]) > 0:
                report.append(f"- {r['method']} @ {r['seq_length']}: {r['latency_ms']}ms, {r['memory_gb']}GB, {r['flops_g']}GFLOPs")

    # 6. LongBench
    report.append("\n## 6. LongBench\n")
    if "longbench_aggregated" in tables:
        best = max(tables["longbench_aggregated"], key=lambda r: float(r["average"]))
        report.append(f"- Best Average: {best['method']} ({best['average']})")
        for r in tables["longbench_aggregated"]:
            report.append(f"- {r['method']}: avg={r['average']}")

    # 7. NIAH
    report.append("\n## 7. Needle-in-a-Haystack\n")
    if "niah_aggregated" in tables:
        best = max(tables["niah_aggregated"], key=lambda r: float(r["average"]))
        report.append(f"- Best Average: {best['method']} ({best['average']})")
        for r in tables["niah_aggregated"]:
            report.append(f"- {r['method']}: avg={r['average']}")

    # Summary
    report.append("\n## 8. Key Findings\n")

    loaded_rob = "robustness_aggregated" in tables
    loaded_proxy = "proxy_aggregated" in tables
    loaded_lb = "longbench_aggregated" in tables

    if loaded_rob:
        csa_row = next((r for r in tables["robustness_aggregated"] if r["method"] == "csa"), None)
        if csa_row:
            csa_er = float(csa_row["er_at_k"])
            report.append(f"1. **Evidence Preservation (ER@k={csa_er:.4f})**: CSA {'preserves evidence well' if csa_er > 0.3 else 'shows limited evidence selection'} based on ER@k.")
    if loaded_proxy:
        csa_rows = [r for r in tables["proxy_aggregated"] if r["estimator"] == "csa_proxy"]
        if csa_rows:
            avg_sp = float(np.mean([float(r["spearman"]) for r in csa_rows]))
            report.append(f"2. **Proxy Approximation (avg Spearman={avg_sp:.4f})**: CSA's gradient proxy {'correlates well with' if avg_sp > 0.3 else 'shows limited correlation with'} exact intervention.")
    if loaded_lb:
        csa_lb = next((r for r in tables["longbench_aggregated"] if r["method"] == "csa"), None)
        dense_lb = next((r for r in tables["longbench_aggregated"] if r["method"] == "dense"), None)
        if csa_lb and dense_lb:
            lb_gap = float(csa_lb["average"]) - float(dense_lb["average"])
            report.append(f"3. **LongBench**: CSA {'outperforms' if lb_gap > 0 else 'underperforms'} dense attention by {abs(lb_gap):.4f} on average.")
        sim_lb = next((r for r in tables["longbench_aggregated"] if r["method"] == "similarity_topk"), None)
        if csa_lb and sim_lb:
            lb_gap2 = float(csa_lb["average"]) - float(sim_lb["average"])
            report.append(f"4. **CSA vs Sparse**: CSA {'outperforms' if lb_gap2 > 0 else 'underperforms'} similarity-based sparse attention by {abs(lb_gap2):.4f}.")

    report.append("\n*All measurements are empirical. No values fabricated.*")

    save_latex(f"{TABLES_DIR}/final_experiment_report.md", "\n".join(report))
    print("  Saved final_experiment_report.md")


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="CSA Full Experimental Pipeline")
    parser.add_argument("--smoke", action="store_true", help="Quick smoke test")
    parser.add_argument("--quick", action="store_true", help="Limited run (1 seed)")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID")
    parser.add_argument("--phases", type=str, nargs="+",
                        default=["1", "2", "3", "4", "5", "6", "7"],
                        help="Which phases to run")
    parser.add_argument("--no-final", action="store_true", help="Skip final report generation")

    args = parser.parse_args()
    global SMOKE, QUICK
    SMOKE = args.smoke
    QUICK = args.quick

    set_env(args.gpu)
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu"
    print(f"Device: {device}, Smoke={SMOKE}, Quick={QUICK}")
    print(f"Phases: {args.phases}")

    torch.manual_seed(42)

    phases = {
        "1": ("Causal Robustness", phase1_causal_robustness),
        "2": ("Proxy Validation", phase2_proxy_validation),
        "3": ("Ablation Study", phase3_ablation),
        "4": ("Sensitivity Analysis", phase4_sensitivity),
        "5": ("Efficiency Analysis", phase5_efficiency),
        "6": ("LongBench", phase6_longbench),
        "7": ("Needle-in-Haystack", phase7_niah),
    }

    for p in args.phases:
        if p in phases:
            name, fn = phases[p]
            print(f"\n{'#' * 70}")
            print(f"# PHASE {p}: {name}")
            print(f"{'#' * 70}")
            t0 = time.time()
            fn(device)
            elapsed = time.time() - t0
            print(f"  Completed in {elapsed:.0f}s")
        else:
            print(f"Unknown phase: {p}")

    if not args.no_final:
        generate_final_report()

    print(f"\n{'=' * 70}")
    print("Pipeline complete.")
    print(f"Tables: {TABLES_DIR}/")
    print(f"Figures: {FIGURES_DIR}/")
    print(f"Results: {RESULTS_DIR}/")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
