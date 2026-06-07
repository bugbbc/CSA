#!/usr/bin/env python3
"""
Robust unit test: CSA routing vs similarity routing vs gated.
Trains on synthetic data with evidence/spurious/confounders,
measures ER@k and SI@k for all methods.
"""

import torch, sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
torch.manual_seed(42)
np.random.seed(42)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

from csa.models.encoder import CSAEncoder
from csa.data.causal_robustness import CausalRobustnessDataset, collate_causal
from csa.utils.metrics import evidence_recall_at_k, spurious_inclusion_at_k

# ─── Synthetic data with known causal structure ────────────────────────
# Control: small seq, strong evidence, clear spurious
sl = 64
train_n = 200
test_n = 50
epochs = 5
dm = 128
nly = 2
nh = 2
W = 16  # window smaller than seq so routing matters
K = 8

# Use a seed where data is clean
ds_seed = 123

train_ds = CausalRobustnessDataset(num_examples=train_n, seq_length=sl, split='train', seed=ds_seed)
test_ds = CausalRobustnessDataset(num_examples=test_n, seq_length=sl, split='test', seed=ds_seed+1)
tl = torch.utils.data.DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=collate_causal)
tel = torch.utils.data.DataLoader(test_ds, batch_size=16, collate_fn=collate_causal)

# ─── Train models ──────────────────────────────────────────────────────
results = {}
for method in ['dense', 'similarity_topk', 'gated_sparse', 'csa']:
    print(f"\n{'='*60}")
    print(f"Training {method}...")
    print('='*60)
    m = CSAEncoder(vocab_size=97, d_model=dm, d_ff=dm*4, n_layers=nly, n_heads=nh,
                   dropout=0.1, max_len=sl+16, task='classification', num_classes=2,
                   attn_type=method, window=W, k=K, refresh_interval=2,
                   baseline_type='zero', pad_token_id=0).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs*len(tl))
    m.train()
    for ep in range(epochs):
        for batch in tl:
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            lbls = batch['labels'].to(device)
            opt.zero_grad()
            out = m(ids, attention_mask=mask, labels=lbls)
            out['loss'].backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step(); sched.step()
    m.eval()

    # Accuracy
    correct, total = 0, 0
    for batch in tel:
        ids = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        lbls = batch['labels'].to(device)
        with torch.no_grad():
            preds = m(ids, attention_mask=mask)['logits'].argmax(-1)
            correct += (preds == lbls).sum().item(); total += lbls.size(0)
    acc = correct / max(total, 1)
    print(f"Accuracy: {acc:.4f}")

    # ER@k / SI@k using contribution scores (for CSA) or routing mask (for others)
    er_vals, si_vals = [], []
    model_k = min(K, sl)

    for batch in tel:
        ids = batch['input_ids'].to(device)
        with torch.no_grad():
            out = m(ids, return_aux=(method == 'csa'))

        if method == 'csa':
            # Use contribution scores
            contribs = []
            if 'aux' in out:
                for la in out['aux']:
                    if isinstance(la, dict) and 'contrib_scores' in la:
                        cs = la['contrib_scores']
                        if isinstance(cs, torch.Tensor) and cs.ndim >= 2:
                            contribs.append(cs)
            if contribs:
                avg = torch.stack(contribs).mean(0)
                for bi in range(ids.size(0)):
                    scores = avg[bi].cpu().numpy()
                    tk = np.argsort(-scores)[:model_k]
                    seq = ids[bi].cpu().numpy()
                    for ev_ids in batch['evidence_token_ids']:
                        ev_set = set(ev_ids)
                        ev_pos = np.where(np.isin(seq, list(ev_set)))[0]
                        if len(ev_pos):
                            er_vals.append(evidence_recall_at_k(tk, ev_pos, model_k))
                    for sp_ids in batch['spurious_token_ids']:
                        sp_set = set(sp_ids)
                        sp_pos = np.where(np.isin(seq, list(sp_set)))[0]
                        if len(sp_pos):
                            si_vals.append(spurious_inclusion_at_k(tk, sp_pos, model_k))
        elif method in ('similarity_topk', 'gated_sparse'):
            # Use routing mask for fair baseline ER@k/SI@k
            mm = m.layers[0].self_attn
            if hasattr(mm, 'routing'):
                # Manual compute
                x = m.embed(ids)
                q = mm.q_proj(x).view(ids.size(0), -1, nh, dm//nh).transpose(1,2)
                k = mm.k_proj(x).view(ids.size(0), -1, nh, dm//nh).transpose(1,2)
                scores = torch.matmul(q, k.transpose(-2, -1)) / ((dm//nh)**0.5)
                mask_routing = mm.routing.compute_mask(scores)
                selected = mask_routing[0,0].any(dim=0).cpu().numpy()
                sel_idx = np.where(selected)[0]
                for bi in range(ids.size(0)):
                    seq = ids[bi].cpu().numpy()
                    for ev_ids in batch['evidence_token_ids']:
                        ev_set = set(ev_ids)
                        ev_pos = np.where(np.isin(seq, list(ev_set)))[0]
                        if len(ev_pos):
                            er_vals.append(evidence_recall_at_k(sel_idx, ev_pos, model_k))
                    for sp_ids in batch['spurious_token_ids']:
                        sp_set = set(sp_ids)
                        sp_pos = np.where(np.isin(seq, list(sp_set)))[0]
                        if len(sp_pos):
                            si_vals.append(spurious_inclusion_at_k(sel_idx, sp_pos, model_k))

    er = float(np.mean(er_vals)) if er_vals else 0.0
    si = float(np.mean(si_vals)) if si_vals else 0.0
    results[method] = {'accuracy': acc, 'er_at_k': er, 'si_at_k': si}
    print(f"ER@k: {er:.4f}, SI@k: {si:.4f}")

print(f"\n{'='*60}")
print(f"FINAL RESULTS")
print(f"{'='*60}")
print(f"{'Method':20s} {'Accuracy':12s} {'ER@k':12s} {'SI@k':12s}")
print('-'*56)
for method, res in results.items():
    print(f"{method:20s} {res['accuracy']:12.4f} {res['er_at_k']:12.4f} {res['si_at_k']:12.4f}")

# Verify claims
if 'csa' in results:
    csa_er = results['csa']['er_at_k']
    sim_er = results.get('similarity_topk', {}).get('er_at_k', 0)
    if csa_er > sim_er + 0.1:
        print(f"\n✓ CLAIM SUPPORTED: CSA ER@k ({csa_er:.4f}) > Similarity ER@k ({sim_er:.4f})")
        print("  CSA preserves evidence better than similarity-based routing.")
    else:
        print(f"\n✗ CLAIM NOT SUPPORTED: CSA ER@k ({csa_er:.4f}) ≈ Similarity ER@k ({sim_er:.4f})")
    csa_si = results['csa']['si_at_k']
    gs_si = results.get('gated_sparse', {}).get('si_at_k', 0)
    if csa_si < gs_si - 0.05:
        print(f"✓ CSA SI@k ({csa_si:.4f}) < Gated Sparse SI@k ({gs_si:.4f}) — less spurious inclusion")
    else:
        print(f"→ CSA SI@k ({csa_si:.4f}) ≈ Gated Sparse SI@k ({gs_si:.4f})")
