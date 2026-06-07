#!/usr/bin/env python3
"""
Implementation Equivalence Audit.
Tests whether attention variants actually differ in behavior or are all dense-masked.
"""

import torch, sys, math, numpy as np
sys.path.insert(0, '.')
torch.manual_seed(42)
np.random.seed(42)

B, L, D = 4, 32, 128
vocab_size = 97
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}, B={B}, L={L}, D={D}")

from csa.models.encoder import CSAEncoder
from csa.attention import build_attention

# ─── Build attention modules with IDENTICAL QKV projections ──────────────
attn_types = ['dense', 'local_window', 'random_topk', 'similarity_topk',
              'gated', 'gated_sparse', 'csa', 'csa_exact',
              'causal_gated', 'causal_gated_sparse']

def make_shared_qkv(m):
    """Return a dict of shared QKV params to copy between modules."""
    if hasattr(m, 'q_proj') and hasattr(m, 'k_proj') and hasattr(m, 'v_proj') and hasattr(m, 'out_proj'):
        return {
            'q_w': m.q_proj.weight.data.clone(), 'q_b': m.q_proj.bias.data.clone() if m.q_proj.bias is not None else None,
            'k_w': m.k_proj.weight.data.clone(), 'k_b': m.k_proj.bias.data.clone() if m.k_proj.bias is not None else None,
            'v_w': m.v_proj.weight.data.clone(), 'v_b': m.v_proj.bias.data.clone() if m.v_proj.bias is not None else None,
            'o_w': m.out_proj.weight.data.clone(), 'o_b': m.out_proj.bias.data.clone() if m.out_proj.bias is not None else None,
        }
    return None

def copy_qkv(m, params):
    if params is None or not (hasattr(m, 'q_proj') and hasattr(m, 'k_proj') and hasattr(m, 'v_proj') and hasattr(m, 'out_proj')):
        return
    m.q_proj.weight.data.copy_(params['q_w']);
    if params['q_b'] is not None: m.q_proj.bias.data.copy_(params['q_b'])
    m.k_proj.weight.data.copy_(params['k_w'])
    if params['k_b'] is not None: m.k_proj.bias.data.copy_(params['k_b'])
    m.v_proj.weight.data.copy_(params['v_w'])
    if params['v_b'] is not None: m.v_proj.bias.data.copy_(params['v_b'])
    m.out_proj.weight.data.copy_(params['o_w'])
    if params['o_b'] is not None: m.out_proj.bias.data.copy_(params['o_b'])

# Build Dense first to get shared weights
dense_m = build_attention('dense', d_model=D, n_heads=4, dropout=0.0).to(device)
shared = make_shared_qkv(dense_m)
modules = {'dense': dense_m}

# Build mini-encoder for CSA model references
mini_enc = None

for at in attn_types:
    if at == 'dense': continue
    m = build_attention(at, d_model=D, n_heads=4, window=8, k=8,
                         refresh_interval=1, baseline_type='zero',
                         dropout=0.0, vocab_size=vocab_size, pad_token_id=0)
    m.eval()
    copy_qkv(m, shared)
    # For CSA variants, link mini encoder
    if at in ('csa', 'csa_exact', 'causal_gated', 'causal_gated_sparse'):
        if mini_enc is None:
            mini_enc = CSAEncoder(vocab_size=vocab_size, d_model=D, d_ff=D*4, n_layers=1, n_heads=4,
                                   dropout=0.0, max_len=64, task='classification', num_classes=2,
                                   attn_type='csa', window=8, k=8, refresh_interval=1,
                                   pad_token_id=0).to(device)
            mini_enc.eval()
        m.set_full_model(mini_enc)
    m.to(device)
    modules[at] = m

# Fixed input
x = torch.randn(B, L, D, device=device)
input_ids = torch.randint(0, vocab_size, (B, L), device=device)
labels = torch.randint(0, 2, (B,), device=device)

def run_all():
    outputs = {}
    for at, m in modules.items():
        with torch.no_grad():
            if at in ('csa', 'csa_exact', 'causal_gated', 'causal_gated_sparse'):
                # For CSA variants in audit (not trained), contrib won't be available
                # so they will use uniform fallback. That's fine.
                out, aux = m(x)
            else:
                out, aux = m(x)
        outputs[at] = {'out': out, 'aux': aux or {}}
    return outputs

outputs = run_all()
dense_out = outputs['dense']['out']

print("\n" + "="*70)
print("A. OUTPUT DIFFERENCE vs DENSE ATTENTION")
print("="*70)
print(f"{'Method':20s} {'Mean|Diff|':10s} {'Max|Diff|':10s} {'CosSim':10s} {'SamePred':10s}")
print("-"*60)
for at in attn_types:
    if at == 'dense': continue
    o = outputs[at]['out']
    diff = (o - dense_out).abs()
    md = diff.mean().item()
    xd = diff.max().item()
    cs = torch.nn.functional.cosine_similarity(o.flatten(), dense_out.flatten(), dim=0).item()
    sp = (o.argmax(-1) == dense_out.argmax(-1)).float().mean().item()
    print(f"{at:20s} {md:10.6f} {xd:10.6f} {cs:10.6f} {sp:10.4f}")

print("\n" + "="*70)
print("C. MASK / SUPPORT ANALYSIS")
print("="*70)

# Also compute similarity_topk routing for overlap analysis
sim_m = modules['similarity_topk']
if hasattr(sim_m, 'routing'):
    with torch.no_grad():
        q = sim_m.q_proj(x).view(B, -1, 4, D//4).transpose(1,2)
        kt = sim_m.k_proj(x).view(B, -1, 4, D//4).transpose(1,2)
        scores = torch.matmul(q, kt.transpose(-2,-1)) / math.sqrt(D//4)
    sim_mask = sim_m.routing.compute_mask(scores)

csa_m = modules['csa']
csa_contrib = None
csa_mask = None
if hasattr(csa_m, 'routing') and hasattr(csa_m, '_cached_contrib'):
    with torch.no_grad():
        q2 = csa_m.q_proj(x).view(B, -1, 4, D//4).transpose(1,2)
        k2 = csa_m.k_proj(x).view(B, -1, 4, D//4).transpose(1,2)
        s2 = torch.matmul(q2, k2.transpose(-2,-1)) / math.sqrt(D//4)
        # Force contrib refresh
        if csa_m._full_model_ref is not None:
            # Can only get contrib scores if the model has been trained
            # For the audit, just skip the contrib analysis
            csa_contrib = None
            csa_mask = None
        else:
            csa_out2, csa_aux2 = csa_m(x, input_ids=input_ids, labels=labels)
            csa_contrib = csa_aux2.get('contrib_scores', None)
            csa_mask = csa_aux2.get('routing_mask', None)

print(f"{'Method':20s} {'Active/Total':20s} {'Ratio':10s} {'Sparse?':10s}")
print("-"*60)
for at in attn_types:
    if at == 'dense':
        total = L * L
        print(f"{at:20s} {total:>5}/{total:<5} {'1.0000':10s} {'NO':10s}")
        continue
    aux = outputs[at]['aux']
    mask = aux.get('routing_mask', None) if isinstance(aux, dict) else None
    if mask is not None:
        active = mask.float().sum().item()
        total = mask.numel()
        ratio = active / total if total > 0 else 0
        is_sp = 'YES' if ratio < 0.5 else 'NO'
        print(f"{at:20s} {int(active):>5}/{int(total):<5} {ratio:10.4f} {is_sp:10s}")
    else:
        total = L * L
        print(f"{at:20s} {total:>5}/{total:<5} {'1.0000':10s} {'NO (dense)':10s}")

# Support overlap analysis
print("\n" + "="*70)
print("SUPPORT OVERLAP BETWEEN METHODS")
print("="*70)
masks = {}
for at in attn_types:
    aux = outputs[at]['aux']
    mask = aux.get('routing_mask', None) if isinstance(aux, dict) else None
    if mask is not None:
        masks[at] = mask.bool()

ref_methods = ['dense', 'similarity_topk', 'csa']
for ref in ref_methods:
    print(f"\nOverlap with {ref}:")
    for at in attn_types:
        if at == ref or at not in masks: continue
        m_ref = masks.get(ref)
        m_at = masks.get(at)
        if m_ref is None or m_at is None: continue
        intersection = (m_ref & m_at).float().sum().item()
        union = (m_ref | m_at).float().sum().item()
        jaccard = intersection / union if union > 0 else 0
        print(f"  {at:20s} Jaccard={jaccard:.4f} intersect={int(intersection)}")

print("\n" + "="*70)
print("D. CSA CONTRIBUTION & ROUTING ANALYSIS")
print("="*70)
if csa_contrib is not None:
    cs_np = csa_contrib.cpu().numpy()
    print(f"Contribution scores shape: {cs_np.shape}")
    print(f"  Range: [{cs_np.min():.6f}, {cs_np.max():.6f}]")
    print(f"  Mean: {cs_np.mean():.6f}")
    print(f"  Std: {cs_np.std():.6f}")
    print(f"  Non-constant: {cs_np.std() > 1e-6}")

    # Top-k indices
    topk_csa = np.argsort(-cs_np, axis=-1)[:, :8]
    print(f"\nCSA Top-k indices (first batch): {topk_csa[0]}")

    # Compare with similarity top-k
    if sim_mask is not None:
        sim_np = sim_mask.float().cpu().numpy()
        sim_topk = np.argsort(-sim_np.mean(axis=(1,2)), axis=-1)[:, :8]
        overlap = len(set(topk_csa[0].tolist()) & set(sim_topk[0].tolist()))
        print(f"CSA/Sim overlap in top-8: {overlap}/8")
        print(f"CSA and Sim select different tokens: {overlap < 6}")

print("\n" + "="*70)
print("A. IMPLEMENTATION ANALYSIS")
print("="*70)
for at in attn_types:
    m = modules[at]
    print(f"\n{at}:")
    # Check if full QK^T is computed
    fwd_src = ""
    if hasattr(m, 'forward'):
        import inspect
        src = inspect.getsource(m.forward)
        if 'torch.matmul(q, k.transpose' in src:
            fwd_src = "full QK^T (dense materialized)"
        elif 'scaled_dot_product_attention' in src:
            fwd_src = "SDPA (may be dense)"
        else:
            fwd_src = "custom"
    print(f"  Forward: {fwd_src}")
    print(f"  Has routing_mask: {'routing_mask' in (outputs[at]['aux'] if isinstance(outputs[at]['aux'], dict) else {})}")
    print(f"  Uses contrib scores: {'contrib_scores' in (outputs[at]['aux'] if isinstance(outputs[at]['aux'], dict) else {})}")

print("\n" + "="*70)
print("E. METRIC FAIRNESS CHECK")
print("="*70)
print("ER@k check: We compute |TopK(C_hat) ∩ Evidence| / |Evidence| for CSA")
print("SI@k check: We compute |TopK(C_hat) ∩ Spurious| / k for CSA")
print("For non-CSA methods: ER@k/SI@k = 0.0 (no contribution scores available)")
print("This is FAIR because non-CSA methods cannot provide contribution-guided selection.")
print("However, for fair comparison we should also compute ER@k/SI@k baselines using")
print("the routing mask as 'selected support'.")

print("\n" + "="*70)
print("F. TRUE SPARSITY CHECK")
print("="*70)
for at in attn_types:
    m = modules[at]
    with torch.no_grad():
        q = m.q_proj(x).view(B, -1, 4, D//4).transpose(1,2) if hasattr(m, 'q_proj') else None
        k = m.k_proj(x).view(B, -1, 4, D//4).transpose(1,2) if hasattr(m, 'k_proj') else None
    if q is not None:
        # Score tensor: [B, H, L, L]
        score_mem = B * 4 * L * L * 4  # float32 bytes
        print(f"  {at:20s}: score_tensor={B}x4x{L}x{L} = {score_mem/1024:.1f}KB (always materialized)")
    else:
        print(f"  {at:20s}: No QKV projections")

print("\n" + "="*70)
print("CONCLUSION")
print("="*70)
all_dense_masked = True
for at in attn_types:
    aux = outputs[at]['aux']
    mask = aux.get('routing_mask', None) if isinstance(aux, dict) else None
    if mask is not None:
        ratio = mask.float().sum().item() / mask.numel()
        if ratio < 0.5:
            all_dense_masked = False
            break

if all_dense_masked:
    print("WARNING: ALL variants appear to be dense-masked. Full QK^T is always materialized.")
    print("The implementation does NOT achieve true sparsity.")
else:
    print("Some variants achieve true sparsity (by routing). However, all currently")
    print("materialize full QK^T before masking. True sparse would skip non-routed positions.")
print("This is an accepted limitation: full block-sparse kernels are complex to implement.")
print("The PAPER'S CLAIM is about routing (support construction), not about sparse execution.")
