import os
import math
import random
import pickle
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict

SEED = 666
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DATA_DIR = "/data/coding/line3/dataset_b/rec_data/"
OUT_DIR = "./model_save_vb17/rec_model/worker_1/round_2"
os.makedirs(OUT_DIR, exist_ok=True)

train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
item_df = pd.read_csv(os.path.join(DATA_DIR, "item.csv"))

def parse_seq(s):
    if pd.isna(s): return []
    return re.findall(r'i\d+', str(s))

train_df['seq_list'] = train_df['item_seq_dedup'].apply(parse_seq)
test_df['seq_list'] = test_df['item_seq_dedup'].apply(parse_seq)

valid_iids = sorted(item_df['iid'].astype(str).unique())
item2idx = {'<pad>': 0}
for i, iid in enumerate(valid_iids, 1):
    item2idx[iid] = i
idx2item = {v: k for k, v in item2idx.items()}
NUM_ITEMS = len(item2idx)

def map_seq(seq):
    return [item2idx.get(x, 0) for x in seq]

train_df['seq_idx'] = train_df['seq_list'].apply(map_seq)
train_df['target_idx'] = train_df['target_iid'].astype(str).apply(lambda x: item2idx.get(x, 0)).astype(int)
test_df['seq_idx'] = test_df['seq_list'].apply(map_seq)

def _reproduce_val_split(train_df: pd.DataFrame) -> set:
    unique_uids = sorted(train_df["uid"].unique())
    rng = np.random.RandomState(666)
    shuffled_uids = rng.permutation(unique_uids)
    n_train = math.ceil(len(shuffled_uids) * 0.8)
    return set(shuffled_uids[n_train:])

val_uids = _reproduce_val_split(train_df)
train_mask = ~train_df['uid'].isin(val_uids)
train_split_df = train_df[train_mask].copy()
val_split_df = train_df[~train_mask].copy()

MAX_LEN = 50
class RecDataset(Dataset):
    def __init__(self, df, is_test=False):
        self.uids = df['uid'].astype(str).tolist()
        self.seqs = df['seq_idx'].tolist()
        self.targets = df['target_idx'].tolist() if not is_test else None
        self.is_test = is_test
    def __len__(self): return len(self.uids)
    def __getitem__(self, idx):
        uid = self.uids[idx]
        seq = self.seqs[idx]
        if len(seq) > MAX_LEN:
            seq = seq[-MAX_LEN:]
        else:
            seq = seq + [0] * (MAX_LEN - len(seq))
        seq_tensor = torch.tensor(seq, dtype=torch.long)
        if self.is_test:
            return seq_tensor, uid
        return seq_tensor, torch.tensor(self.targets[idx], dtype=torch.long), uid

class SASRec(nn.Module):
    def __init__(self, num_items, max_len, d_model=128, n_heads=4, n_layers=2, dropout=0.3):
        super().__init__()
        self.item_emb = nn.Embedding(num_items, d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.emb_dropout = nn.Dropout(dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_model*4, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.dropout = nn.Dropout(dropout)
        self.num_items = num_items
    def forward(self, seq):
        mask = (seq == 0)
        pos_ids = torch.arange(seq.size(1), device=seq.device).unsqueeze(0).expand_as(seq)
        x = self.item_emb(seq) + self.pos_emb(pos_ids)
        x = self.emb_dropout(x)
        sz = seq.size(1)
        causal_mask = torch.triu(torch.ones(sz, sz, device=seq.device), diagonal=1).bool()
        out = self.encoder(x, mask=causal_mask, src_key_padding_mask=mask)
        lengths = (~mask).sum(dim=1) - 1
        lengths = torch.clamp(lengths, min=0)
        batch_idx = torch.arange(seq.size(0), device=seq.device)
        seq_rep = out[batch_idx, lengths]
        seq_rep = self.dropout(seq_rep)
        logits = torch.matmul(seq_rep, self.item_emb.weight.t())
        return logits

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SASRec(NUM_ITEMS, MAX_LEN).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150)

train_ds = RecDataset(train_split_df)
val_ds = RecDataset(val_split_df)
train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0)

def compute_ndcg10(pred_indices, targets):
    ndcg = 0.0
    count = 0
    for i in range(len(targets)):
        t = targets[i]
        if t == 0: continue
        preds = pred_indices[i]
        hit_idx = np.where(preds == t)[0]
        if len(hit_idx) > 0:
            rank = hit_idx[0] + 1
            dcg = 1.0 / math.log2(rank + 1)
            idcg = 1.0 / math.log2(2)
            ndcg += dcg / idcg
        count += 1
    return ndcg / max(count, 1)

best_ndcg = -1.0
patience = 40
patience_cnt = 0
max_epochs = 150

for epoch in range(1, max_epochs + 1):
    model.train()
    total_loss = 0.0
    for seq, target, _ in train_loader:
        seq, target = seq.to(device), target.to(device)
        optimizer.zero_grad()
        logits = model(seq)
        loss = criterion(logits, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * seq.size(0)
    scheduler.step()
    avg_loss = total_loss / len(train_ds)

    if epoch % 10 == 0:
        model.eval()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for seq, target, _ in val_loader:
                seq, target = seq.to(device), target.to(device)
                logits = model(seq)
                _, top10 = torch.topk(logits, 10, dim=1)
                all_preds.append(top10.cpu().numpy())
                all_targets.append(target.cpu().numpy())
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        ndcg = compute_ndcg10(all_preds, all_targets)
        print(f"Epoch {epoch}, Loss: {avg_loss:.4f}, Val NDCG@10: {ndcg:.4f}")

        if ndcg > best_ndcg:
            best_ndcg = ndcg
            patience_cnt = 0
            torch.save(model.state_dict(), os.path.join(OUT_DIR, "best_model.pth"))
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

with open(os.path.join(OUT_DIR, "item2idx.pkl"), "wb") as f: pickle.dump(item2idx, f)
with open(os.path.join(OUT_DIR, "idx2item.pkl"), "wb") as f: pickle.dump(idx2item, f)

def run_tta_inference(df, out_prefix, is_test=False):
    ds = RecDataset(df, is_test=is_test)
    loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=0)
    model.load_state_dict(torch.load(os.path.join(OUT_DIR, "best_model.pth"), map_location=device))
    model.train()

    tta_rounds = 30
    uid_probs = defaultdict(lambda: np.zeros(NUM_ITEMS))
    uid_counts = defaultdict(int)

    with torch.no_grad():
        for _ in range(tta_rounds):
            for batch in loader:
                if is_test:
                    seq, uids = batch
                else:
                    seq, _, uids = batch
                seq = seq.to(device)
                logits = model(seq)
                probs = F.softmax(logits, dim=1).cpu().numpy()
                for i in range(len(uids)):
                    uid_str = str(uids[i])
                    uid_probs[uid_str] += probs[i]
                    uid_counts[uid_str] += 1

    for uid in uid_probs:
        uid_probs[uid] /= uid_counts[uid]

    top100_rows = []
    softmax_rows = []

    sorted_uids = sorted(uid_probs.keys())
    for uid in sorted_uids:
        probs = uid_probs[uid]
        top100_idx = np.argsort(probs)[::-1][:100]
        valid_iids = []
        valid_scores = []
        for idx in top100_idx:
            if idx == 0: continue
            iid = idx2item[idx]
            if iid in valid_iids: continue
            valid_iids.append(iid)
            valid_scores.append(probs[idx])
            if len(valid_iids) == 100: break

        pred_str = ",".join(valid_iids)
        score_str = ",".join([f"{iid}:{score:.4f}" for iid, score in zip(valid_iids, valid_scores)])

        top100_rows.append({"uid": uid, "prediction": pred_str})
        softmax_rows.append({"uid": uid, "item_scores": score_str})

    pd.DataFrame(top100_rows).to_csv(os.path.join(OUT_DIR, f"{out_prefix}-top100.csv"), index=False)
    pd.DataFrame(softmax_rows).to_csv(os.path.join(OUT_DIR, f"{out_prefix}_softmax_item.csv"), index=False)

    if is_test:
        top10_rows = []
        for r in top100_rows:
            preds = r["prediction"].split(",")
            top10 = preds[:10]
            top10_rows.append({"uid": r["uid"], "prediction": ",".join(top10)})
        pd.DataFrame(top10_rows).to_csv(os.path.join(OUT_DIR, "B2.csv"), index=False)

print("Running Validation TTA...")
run_tta_inference(val_split_df, "validation", is_test=False)
print("Running Test TTA...")
run_tta_inference(test_df, "B2", is_test=True)
print("Done.")