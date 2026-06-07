# CSA Implementation Audit Report

## Overview

Audit of the CSA experimental framework implementation against paper specifications.

---

## Check 1: Contribution scores computed BEFORE sparse support construction

**STATUS: PASS**

Code: [csa.py:138-167](csa/attention/csa.py#L138-L167)
- `contrib_scores` computed (or loaded from cache) on line 141/148/153
- Then `routing_mask = self.routing.compute_mask(contrib_for_routing)` on line 167
- Order is correct: contribution first, routing second.

---

## Check 2: Sparse support exactly follows Support(q_i) = LocalWindow(q_i,w) ∪ TopK(C_hat,k)

**STATUS: PASS**

Code: [routing.py:140-156](csa/attention/routing.py#L140-L156)
- `LocalWindowPlusTopKRouting` computes `local_mask | topk_mask` on line 154-156
- In `csa.py`, the routing mask is applied before softmax (line 173).

---

## Check 3: Full dense QK^T is NOT materialized

**STATUS: FAIL**

Code: [csa.py:136](csa/attention/csa.py#L136)
```python
sim_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
```
This materializes the full `[B, H, L, L]` attention score matrix. True sparse would compute similarity only for the `w + k` selected positions per query.

**Risk:** Memory scales O(L²) instead of O(L · (w+k)). At L=65536, w=128, k=32, dense materialization is ~256× more expensive than necessary.

**Fix:** Implement masked SDP that only computes QK^T for routed positions.

---

## Check 4: Sparse logits computed only on selected token pairs

**STATUS: FAIL**

Same as Check 3. The `masked_fill` approach on line 173 computes all logits then zeros out non-selected ones. True sparse would compute logits only for the union of local window and top-k positions.

---

## Check 5: Contribution proxy exactly matches C_hat_j = |⟨∇_{x_j}L, x_j - x̃_j⟩|

**STATUS: PASS**

Code: [contrib.py:139-145](csa/attention/contrib.py#L139-L145)
```python
grad_x = self._cached_grad
x_tilde = self._get_baseline(x.detach(), embed_layer)
diff = x.detach() - x_tilde
inner_product = (grad_x * diff).sum(dim=-1)
C_hat = inner_product.abs()
```
Correct computation of the first-order Taylor contribution proxy.

---

## Check 6: Gradients computed with respect to token representations

**STATUS: PASS**

Code: [contrib.py:118-133](csa/attention/contrib.py#L118-L133)
- `x = embed(input_ids).detach()` — detached from embedding lookup
- `x.requires_grad_(True)` — re-attached to computation graph
- Hook captures `grad_x` w.r.t. token embedding `x`
- Correct: gradients are w.r.t. token representations `x_j`.

---

## Check 7: Exact Intervention CSA performs explicit token interventions

**STATUS: PASS**

Code: [contrib.py:220-236](csa/attention/contrib.py#L220-L236)
- For each position j: `x_interv[:, j, :] = baseline_emb`
- Runs forward pass with intervened embeddings
- Measures KL divergence: `KL(f(X) || f(X_do(j)))`
- Correct O(L) intervention procedure.

---

## Check 8: Exact Intervention CSA restricted to short sequences

**STATUS: PASS**

Code: [contrib.py:151-158](csa/attention/contrib.py#L151-L158)
- Docstring and comments specify "Only for short sequences"
- Complexity O(L) forward passes per call
- Used only in Exp5 (proxy validation) with seq lengths 64, 128, 256.

---

## Check 9: Random Top-k and Similarity Top-k use identical token budgets

**STATUS: PASS**

Code: [random_topk.py:28](csa/attention/random_topk.py#L28), [similarity_topk.py:29](csa/attention/similarity_topk.py#L29)
- Both use `routing = LocalWindowPlusTopKRouting(window, k)` or `RandomTopKRouting(k)`
- Both apply `w + k` tokens per query (Random uses `k` in top-k portion)
- Budgets are identical when same `window` and `k` configured.

---

## Check 10: ER@k and SI@k correctly implemented

**STATUS: PASS (functions) / FAIL (usage)**

Code: [utils/metrics.py:55-72](csa/utils/metrics.py#L55-L72)
- ER@k: `|selected ∩ evidence| / |evidence|` — correct.
- SI@k: `|selected ∩ spurious| / k` — correct.

**Usage issue in exp4_causal.py**: The ER@k and SI@k metrics are computed using `routing_mask.any(dim=0)` which gives *all* routed key positions (local window + top-k), not just the top-k by contribution score. The paper defines:
- ER@k = `|TopK(C_hat) ∩ Evidence| / |Evidence|` (top-k C_hat, not all routed)

**Fix:** Compute top-k by contribution scores explicitly for ER@k and SI@k, not using the full routing mask.

---

## Check 11: Spearman, Kendall, NDCG, Top-k overlap correct

**STATUS: PASS**

Code: [utils/metrics.py:13-50](csa/utils/metrics.py#L13-L50)
- Spearman: uses `scipy.stats.spearmanr` — correct.
- Kendall: uses `scipy.stats.kendalltau` — correct.
- Top-k overlap: `|pred_topk ∩ true_topk| / min(k, len(true_topk))` — correct.
- NDCG@k: standard DCG/IDCG formulation — correct.

---

## Check 12: Random seeds fixed

**STATUS: PASS**

Code: [utils/seed.py:8-20](csa/utils/seed.py#L8-L20)
- Sets `random.seed`, `np.random.seed`, `torch.manual_seed`, `torch.cuda.manual_seed_all`
- Enables deterministic CuDNN algorithms
- Seeds 42, 123, 3407 used consistently.

---

## Additional Issues Found

### Issue A: CSA inference-time routing broken
[csa.py:139](csa/attention/csa.py#L139)
```python
if self.training and self._should_refresh() ...
```
During `model.eval()` (inference), `self.training` is False. The condition never triggers, so CSA always falls back to either uniform scores (if no cache) or stale cached scores. Contribution scores are never refreshed during inference.

**Risk:** CSA evaluates with uniform routing — equivalent to a degenerate baseline.
**Fix:** Remove `self.training` condition, or add `or not self.training` to enable inference refresh.

### Issue B: Dense QK^T materialized in all sparse variants
[csa.py:136](csa/attention/csa.py#L136), [similarity_topk.py:44](csa/attention/similarity_topk.py#L44), [random_topk.py:42](csa/attention/random_topk.py#L42), [local_window.py:49](csa/attention/local_window.py#L49)
All variants compute full `[B, H, L, L]` attention scores then mask. This contradicts sparsity claims.

**Risk:** Memory and compute scale O(L²) for all variants at evaluation time.
**Fix (future):** Implement block-sparse attention with mask-guided computation.

### Issue C: GatedRouting.compute_mask incomplete
[routing.py:172-189](csa/attention/routing.py#L172-L189)
The `compute_mask` method only applies local window mask. The learned gating scores from `get_gate_scores()` are never integrated into the mask.

**Risk:** GatedAttention and GatedSparseAttention fall back to local-window-only behavior.
**Fix:** Implement gating-based token selection in `compute_mask`.

### Issue D: Parameter count mismatch
GatedAttention has `self.gate_proj = nn.Linear(d_model, n_heads * k)` (extra parameters vs other baselines). All other attention variants have identical Q/K/V/Out projection structure.

**Risk:** Comparison between Gated and other methods is confounded by parameter count.
**Mitigation:** Minor — the gate_proj adds only d_model * n_heads * k = 512*8*32 ≈ 131K params (vs ~44.6M total).

### Issue E: forward_with_embeddings double-adds positional embeddings
[encoder.py:166](csa/models/encoder.py#L166)
`forward_with_embeddings` applies `self.embed_pos(embeddings)` but the embeddings parameter may already contain positional information from the `embed()` method.

**Context:** The contrib estimator calls `model.forward_with_embeddings(x, ...)` where `x = embed(input_ids)` (which includes position embeddings). Then `forward_with_embeddings` applies `embed_pos` again.

**Risk:** Double positional encoding corrupts contribution estimates.
**Fix:** Make `forward_with_embeddings` not apply positional embeddings, or have the contrib estimator pass raw token embeddings.

### Issue F: No sequence length padding for varying lengths
No bucketing or padding strategy for variable-length sequences in experiments.

**Risk:** Inefficient batching for LongBench where context lengths vary significantly.

---

## Summary

| Check | Status | Location |
|-------|--------|----------|
| 1. Contribution before routing | PASS | csa.py:138-167 |
| 2. Support = Window + TopK(C_hat) | PASS | routing.py:140-156 |
| 3. Dense QK^T not materialized | **FAIL** | csa.py:136 |
| 4. Sparse logits only on selected | **FAIL** | csa.py:136,173 |
| 5. C_hat = \|⟨∇L, x - x̃⟩\| | PASS | contrib.py:139-145 |
| 6. Gradients w.r.t. token reps | PASS | contrib.py:118-133 |
| 7. Exact intervention | PASS | contrib.py:220-236 |
| 8. Exact restricted short seqs | PASS | contrib.py:151-158 |
| 9. Random/Similarity same budget | PASS | random_topk.py:28, similarity_topk.py:29 |
| 10. ER@k/SI@k correct | PASS (fn) / **FAIL** (usage) | utils/metrics.py, exp4_causal.py |
| 11. Ranking metrics correct | PASS | utils/metrics.py:13-50 |
| 12. Random seeds fixed | PASS | utils/seed.py:8-20 |
| A. Inference-time CSA routing | **FAIL** | csa.py:139 |
| B. Dense QK^T in sparse variants | **FAIL** | All attention modules |
| C. GatedRouting incomplete | **FAIL** | routing.py:172-189 |
| D. Parameter count mismatch | PASS (minor) | gated.py |
| E. Double positional encoding | **FAIL** | encoder.py:166, contrib.py:119 |

## Recommended Fixes (pre-execution)

1. **P0 — Fix Issue A**: Remove `self.training` guard in CSA forward to enable inference-time contribution refresh.
2. **P0 — Fix Issue A (cache)**: Initialize `_cached_contrib` as None and handle differently.
3. **P1 — Fix Issue E**: Pass raw embeddings (no pos encoding) to `forward_with_embeddings` by adding a `add_pos_encoding` flag.
4. **P1 — Fix Issue 10 (usage)**: Refactor exp4 ER@k/SI@k to use top-k C_hat explicitly.
5. **P2 — Fix Issue C**: Complete GatedRouting.compute_mask with learned gating.
6. **P3 — Fix Issues 3-4 (B)**: Requires block-sparse implementation — out of scope for initial experiments but document.

Fixes P0 and P1 will be applied before running experiments.
