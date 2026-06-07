#!/usr/bin/env python3
"""
Emergency Validity Audit — All 6 phases.
"""
import torch, sys, os, numpy as np, csv, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
torch.manual_seed(42); np.random.seed(42)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
TABLES='tables'; REPORTS='reports'
os.makedirs(TABLES,exist_ok=True); os.makedirs(REPORTS,exist_ok=True)
print(f"Device: {device}")

from csa.data.causal_robustness import CausalRobustnessDataset, collate_causal
from csa.data.tokenizer import SimpleTokenizer
from csa.models.encoder import CSAEncoder
from csa.attention import build_attention
from csa.utils.metrics import evidence_recall_at_k, spurious_inclusion_at_k
tokenizer=SimpleTokenizer()

# ═══ PHASE 1: DATASET VALIDITY ═══════════════════════════════════════════
print("\n"+"="*60+"\nPHASE 1: DATASET VALIDITY\n"+"="*60)
ds_train=CausalRobustnessDataset(20,64,3,True,"train",42)
ds_robust=CausalRobustnessDataset(20,64,3,False,"test",42)
evA_ids=[tokenizer.encode(w)[0] for w in ["network","layer","gradient","backprop","activation"]]
evB_ids=[tokenizer.encode(w)[0] for w in ["database","query","index","schema","transaction"]]

for label in [0,1]:
    for split,ds in [("train",ds_train),("robust",ds_robust)]:
        examples=[e for e in ds.examples if e['label']==label]
        if examples:
            ex=examples[0]
            text=tokenizer.decode(ex['input_ids'].tolist())
            print(f"\n  Label {label} ({split}): {text[:120]}...")
            print(f"  Evidence: {ex['evidence_words']} Spurious: {ex['spurious_word']}")

# Evidence-only accuracy
ev_correct=0
for ex in ds_train.examples[:100]:
    hasA=bool(set(evA_ids)&set(ex['evidence_token_ids']))
    hasB=bool(set(evB_ids)&set(ex['evidence_token_ids']))
    pred=0 if hasA and not hasB else (1 if hasB and not hasA else -1)
    if pred==ex['label']:ev_correct+=1
print(f"\nEvidence-only accuracy: {ev_correct}/100 ({ev_correct}%)")

# Spurious correlation check
corr_train=sum(1 for e in ds_train.examples if (e['label']==0 and e['spurious_token_ids'][0]==53)or(e['label']==1 and e['spurious_token_ids'][0]==54))
corr_rob=sum(1 for e in ds_robust.examples if (e['label']==0 and e['spurious_token_ids'][0]==54)or(e['label']==1 and e['spurious_token_ids'][0]==53))
print(f"Spurious correlated: train={corr_train}/{len(ds_train)} robust_reversed={corr_rob}/{len(ds_robust)}")

# ═══ PHASE 2: ER@k METRIC AUDIT ═════════════════════════════════════════
print("\n"+"="*60+"\nPHASE 2: ER@k METRIC AUDIT\n"+"="*60)

# Build and train a small model on the causal robustness data
sl=64; dm=64; nh=2; nly=2; K=8; W=16
train_ds=CausalRobustnessDataset(200,sl,3,True,"train",42)
test_ds=CausalRobustnessDataset(50,sl,3,True,"test",43)

rows_er=[]
methods_er=['dense','local_window','similarity_topk','gated','csa']
models={}

for method in methods_er:
    import inspect
    sig=inspect.signature(CSAEncoder.__init__)
    params=list(sig.parameters.keys())
    m=CSAEncoder(vocab_size=97,d_model=dm,d_ff=dm*4,n_layers=nly,n_heads=nh,
                 dropout=0.1,max_len=sl+16,task='classification',num_classes=2,
                 attn_type=method,window=W,k=K,refresh_interval=2,baseline_type='zero',
                 pad_token_id=0).to(device)
    opt=torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=0.01)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=3*len(torch.utils.data.DataLoader(train_ds,16,True,collate_fn=collate_causal)))
    m.train()
    for _ in range(3):
        for b in torch.utils.data.DataLoader(train_ds,16,True,collate_fn=collate_causal):
            ids,b_labels=b['input_ids'].to(device),b['labels'].to(device)
            opt.zero_grad()
            o=m(ids,labels=b_labels)
            o['loss'].backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step(); sched.step()
    m.eval()
    models[method]=m

    # Compute ER@k using routing mask for ALL methods (including CSA's routing, not contrib)
    tel=torch.utils.data.DataLoader(test_ds,16,collate_fn=collate_causal)
    er_all,si_all=[],[]
    for b in tel:
        ids=b['input_ids'].to(device)
        mm=m.layers[0].self_attn
        if hasattr(mm,'routing'):
            x=m.embed(ids)
            if hasattr(mm,'q_proj'):
                q=mm.q_proj(x).view(ids.size(0),-1,nh,dm//nh).transpose(1,2)
                k=mm.k_proj(x).view(ids.size(0),-1,nh,dm//nh).transpose(1,2)
                sim=torch.matmul(q,k.transpose(-2,-1))/math.sqrt(dm//nh)
            else:
                sim=torch.randn(ids.size(0),nh,ids.size(1),ids.size(1),device=device)
            rmask=mm.routing.compute_mask(sim)
            selected=rmask[0,0].any(dim=0).cpu().numpy()
            sel_idx=np.where(selected)[0]
            for bi in range(ids.size(0)):
                seq=ids[bi].cpu().numpy()
                ev_pos=np.where(np.isin(seq,evA_ids+evB_ids))[0]
                sp_pos=np.where(np.isin(seq,[53,54]))[0]
                if len(ev_pos):er_all.append(evidence_recall_at_k(sel_idx,ev_pos,len(sel_idx)))
                if len(sp_pos):si_all.append(spurious_inclusion_at_k(sel_idx,sp_pos,len(sel_idx)))
        else:
            # Dense: all positions selected
            for bi in range(ids.size(0)):
                seq=ids[bi].cpu().numpy()
                ev_pos=np.where(np.isin(seq,evA_ids+evB_ids))[0]
                sp_pos=np.where(np.isin(seq,[53,54]))[0]
                sel_idx=np.arange(ids.size(1))
                if len(ev_pos):er_all.append(evidence_recall_at_k(sel_idx,ev_pos,len(sel_idx)))
                if len(sp_pos):si_all.append(spurious_inclusion_at_k(sel_idx,sp_pos,len(sel_idx)))

    er_val=float(np.mean(er_all)) if er_all else 0.0
    si_val=float(np.mean(si_all)) if si_all else 0.0
    print(f"{method:20s} ER@k={er_val:.4f} SI@k={si_val:.4f} (using routing mask)")
    rows_er.append({'method':method,'er_at_k':round(er_val,4),'si_at_k':round(si_val,4)})

with open(f'{TABLES}/support_overlap_debug.csv','w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=rows_er[0].keys()); w.writeheader(); w.writerows(rows_er)

# ═══ PHASE 3: SUPPORT SELECTION COMPARISON ════════════════════════════
print("\n"+"="*60+"\nPHASE 3: SUPPORT SELECTION\n"+"="*60)

# Fixed input on trained models
x_fixed=torch.randn(4,sl,dm).to(device)
support_data=[]
methods_comp=['dense','local_window','random_topk','similarity_topk','gated','csa']

# Use the trained models, intercept routing
import math
for method in methods_comp:
    if method not in models: continue
    m=models[method]
    mm=m.layers[0].self_attn
    if hasattr(mm,'routing'):
        with torch.no_grad():
            q=mm.q_proj(x_fixed).view(4,-1,nh,dm//nh).transpose(1,2) if hasattr(mm,'q_proj') else None
            k=mm.k_proj(x_fixed).view(4,-1,nh,dm//nh).transpose(1,2) if hasattr(mm,'k_proj') else None
            if q is not None:
                sim=torch.matmul(q,k.transpose(-2,-1))/math.sqrt(dm//nh)
            else:
                sim=torch.randn(4,nh,sl,sl,device=device)
            rmask=mm.routing.compute_mask(sim)
            active=rmask[0,0,0].cpu().numpy()
            sel=np.where(active)[0]
            support_data.append({'method':method,'active':len(sel),'indices':sel.tolist()})
    else:
        support_data.append({'method':method,'active':sl,'indices':list(range(sl))})

for d in support_data:
    print(f"{d['method']:20s} active={d['active']:3d} indices={d['indices'][:10]}...")

# Overlap matrix
print("\nSupport Overlap Matrix (Jaccard):")
overlap_mat={}
for d1 in support_data:
    s1=set(d1['indices'])
    for d2 in support_data:
        inter=len(s1 & set(d2['indices'])); union=len(s1 | set(d2['indices']))
        jac=inter/union if union else 0
        overlap_mat[(d1['method'],d2['method'])]=jac

# Print matrix
methods_order=[d['method'] for d in support_data]
print("                 "+"  ".join(f"{m:15s}" for m in methods_order))
for m1 in methods_order:
    row=""
    for m2 in methods_order:
        row+=f"  {overlap_mat.get((m1,m2),0):.3f}          "[:16]
    print(f"{m1:15s}{row}")

# Save matrix
with open(f'{TABLES}/support_overlap_matrix.csv','w',newline='') as f:
    w=csv.writer(f); w.writerow(['']+methods_order)
    for m1 in methods_order:
        w.writerow([m1]+[overlap_mat.get((m1,m2),0)for m2 in methods_order])

# ═══ PHASE 4: OUTPUT DIFFERENCE ════════════════════════════════════════
print("\n"+"="*60+"\nPHASE 4: OUTPUT DIFFERENCE\n"+"="*60)

# Build fresh models with shared weights
ref_attn=build_attention('dense',d_model=dm,n_heads=nh,dropout=0.0).to(device)
shared_wts={
    'q':ref_attn.q_proj.weight.data.clone(),'k':ref_attn.k_proj.weight.data.clone(),
    'v':ref_attn.v_proj.weight.data.clone(),'o':ref_attn.out_proj.weight.data.clone()
}
dense_out, _ = ref_attn(x_fixed)
print(f"{'Method':20s} {'Mean|Diff|':10s} {'Max|Diff|':10s} {'CosSim':10s} {'SamePred':10s}")
print("-"*60)

for method in methods_comp:
    if method=='dense':continue
    m=build_attention(method,d_model=dm,n_heads=nh,window=W,k=K,dropout=0.0).to(device)
    if hasattr(m,'q_proj'):
        m.q_proj.weight.data.copy_(shared_wts['q']); m.q_proj.bias.data.zero_()
        m.k_proj.weight.data.copy_(shared_wts['k']); m.k_proj.bias.data.zero_()
        m.v_proj.weight.data.copy_(shared_wts['v']); m.v_proj.bias.data.zero_()
        m.out_proj.weight.data.copy_(shared_wts['o']); m.out_proj.bias.data.zero_()
    with torch.no_grad():
        out,_=m(x_fixed)
    diff=(out-dense_out).abs()
    md=diff.mean().item(); xd=diff.max().item()
    cs=torch.nn.functional.cosine_similarity(out.flatten(),dense_out.flatten(),dim=0).item()
    sp=(out.argmax(-1)==dense_out.argmax(-1)).float().mean().item()
    print(f"{method:20s} {md:10.6f} {xd:10.6f} {cs:10.6f} {sp:10.4f}")

# ═══ PHASE 5: ADVERSARIAL SANITY ═══════════════════════════════════════
print("\n"+"="*60+"\nPHASE 5: ADVERSARIAL SANITY TEST\n"+"="*60)

# Create a dataset where similarity favors spurious but evidence is causal
# We'll directly construct input IDs where:
# - evidence tokens are at fixed positions with low QK similarity
# - spurious tokens have high QK similarity but are reversed in robust test
class AdversarialDataset(torch.utils.data.Dataset):
    def __init__(self,n=100,sl=32,spurious_correlated=True):
        self.examples=[]
        rng=np.random.RandomState(42)
        for i in range(n):
            label=rng.randint(0,2)
            # Fixed evidence at position 2 (low similarity)
            ev_word="network" if label==0 else "database"
            # Spurious at position 3 (high similarity to query)
            sp_word="sunny" if (label==0)==spurious_correlated else "rainy"
            ev_id=tokenizer.encode(ev_word)[0]
            sp_id=tokenizer.encode(sp_word)[0]
            noise_ids=[tokenizer.encode(w)[0] for w in ["the","a","is","that","it"]]
            seq=[rng.choice(noise_ids) for _ in range(sl)]
            seq[2]=ev_id  # evidence at pos 2
            seq[3]=sp_id  # spurious at pos 3
            self.examples.append({
                'input_ids':torch.tensor(seq),'label':label,
                'evidence_token_ids':[ev_id],'spurious_token_ids':[sp_id],
                'evidence_positions':[2],'spurious_positions':[3],
            })
    def __len__(self):return len(self.examples)
    def __getitem__(self,i):return self.examples[i]

def collate_adv(batch):
    return {
        'input_ids':torch.stack([b['input_ids'] for b in batch]),
        'labels':torch.tensor([b['label'] for b in batch]),
        'evidence_token_ids':[b['evidence_token_ids'] for b in batch],
        'spurious_token_ids':[b['spurious_token_ids'] for b in batch],
    }

adv_train=AdversarialDataset(200,32,True)
adv_test=AdversarialDataset(50,32,True)
adv_robust=AdversarialDataset(50,32,False)

adv_methods=['dense','similarity_topk','gated_sparse','causal_gated_sparse','csa']
adv_results=[]
for method in adv_methods:
    m=CSAEncoder(vocab_size=97,d_model=64,d_ff=256,n_layers=2,n_heads=2,
                 dropout=0.1,max_len=48,task='classification',num_classes=2,
                 attn_type=method,window=8,k=8,refresh_interval=2,baseline_type='zero',
                 pad_token_id=0).to(device)
    tl=torch.utils.data.DataLoader(adv_train,16,True,collate_fn=collate_adv)
    opt=torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=0.01)
    m.train()
    for _ in range(10):
        for b in tl:
            ids,b_labels=b['input_ids'].to(device),b['labels'].to(device)
            opt.zero_grad(); o=m(ids,labels=b_labels); o['loss'].backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
    m.eval()

    def eval_loader(loader):
        correct=0; total=0
        for b in loader:
            ids,b_labels=b['input_ids'].to(device),b['labels'].to(device)
            with torch.no_grad():preds=m(ids)['logits'].argmax(-1)
            correct+=(preds==b_labels).sum().item(); total+=b_labels.size(0)
        return correct/max(total,1)

    iid_acc=eval_loader(torch.utils.data.DataLoader(adv_test,16,collate_fn=collate_adv))
    rob_acc=eval_loader(torch.utils.data.DataLoader(adv_robust,16,collate_fn=collate_adv))
    adv_results.append({'method':method,'iid_accuracy':round(iid_acc,4),'robust_accuracy':round(rob_acc,4)})
    print(f"{method:20s} iid={iid_acc:.4f} robust={rob_acc:.4f}")

with open(f'{TABLES}/routing_vs_weighting_sanity.csv','w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=adv_results[0].keys()); w.writeheader(); w.writerows(adv_results)

# ═══ PHASE 6: COMPLEXITY ═══════════════════════════════════════════════
print("\n"+"="*60+"\nPHASE 6: COMPLEXITY AUDIT\n"+"="*60)

# Check if full QK^T is materialized
import inspect
for method in methods_comp:
    m=build_attention(method,d_model=dm,n_heads=nh,window=W,k=K,dropout=0.0)
    src=inspect.getsource(m.forward)
    has_qkt='torch.matmul(q, k.transpose' in src or 'torch.matmul(q, k' in src
    has_sdpa='scaled_dot_product_attention' in src
    print(f"{method:20s} full_QK^T={has_qkt} uses_SDPA={has_sdpa}")

# Active edges at various lengths
for Ltest in [32,64,128]:
    print(f"\nL={Ltest}:")
    for method in methods_comp:
        m=build_attention(method,d_model=dm,n_heads=nh,window=min(W*2,Ltest),k=min(K,Ltest),dropout=0.0).to(device)
        xt=torch.randn(1,Ltest,dm).to(device)
        with torch.no_grad():
            _,aux=m(xt)
        mask=aux.get('routing_mask',None)
        if mask is not None:
            active=int(mask.float().sum().item())
            total=int(mask.numel())
        else:
            active=total=int(Ltest*Ltest)
        print(f"  {method:20s} active={active:6d}/{total:6d} ({active/total:.2%})")

print("\nEmergency audit complete.")
