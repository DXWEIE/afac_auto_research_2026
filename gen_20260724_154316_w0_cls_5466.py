
import warnings
warnings.filterwarnings("ignore")

# 可选：仅忽略特定高频警告，保留真正的错误提示
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*sparse tensor.*")

import os
os.environ["PYTHONWARNINGS"] = "ignore"


import os
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import add_self_loops, coalesce, to_undirected
from scipy.sparse import csr_matrix
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子保证可复现
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark = True

# ================= 1. 数据加载与预处理 =================
data_dir = '/data/coding/line3/dataset_b/cls_data/'
raw = np.load(os.path.join(data_dir, "B1.npz"))

adj = csr_matrix((raw["adj_data"], raw["adj_indices"], raw["adj_indptr"]), shape=tuple(raw["adj_shape"]))
features = csr_matrix((raw["attr_data"], raw["attr_indices"], raw["attr_indptr"]), shape=tuple(raw["attr_shape"]))

# 构建 edge_index：强制对称归一化 + 自环 + 去重
adj_coo = adj.tocoo()
edge_index = torch.tensor(np.vstack([adj_coo.row, adj_coo.col]), dtype=torch.long)
edge_index = to_undirected(edge_index, num_nodes=adj.shape[0])
edge_index, _ = add_self_loops(edge_index, num_nodes=adj.shape[0])
edge_index = coalesce(edge_index)

# 特征预处理：仅 Z-score 标准化，关闭 PCA 保真
x_dense = features.toarray()
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x_dense)
x = torch.tensor(x_scaled, dtype=torch.float32)

y = torch.tensor(raw["labels"], dtype=torch.long)
train_idx = raw["train_idx"]
test_idx = raw["test_idx"]

data = Data(x=x, edge_index=edge_index, y=y)

# ================= 2. 划分训练集/验证集 =================
def _reproduce_val_split(train_idx: np.ndarray) -> tuple:
    rng = np.random.RandomState(666)
    shuffled_idx = rng.permutation(train_idx)
    n_actual_train = math.ceil(len(shuffled_idx) * 0.8)
    actual_train_idx = shuffled_idx[:n_actual_train]
    val_idx = shuffled_idx[n_actual_train:]
    return actual_train_idx, val_idx

actual_train_idx, val_idx = _reproduce_val_split(train_idx)

# 设备与张量转换
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = data.to(device)
train_idx_t = torch.tensor(actual_train_idx, dtype=torch.long, device=device)
val_idx_t = torch.tensor(val_idx, dtype=torch.long, device=device)
test_idx_t = torch.tensor(test_idx, dtype=torch.long, device=device)

# ================= 3. 模型定义 =================
class GATv2JKModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4, dropout=0.25):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        dim = hidden_channels * heads
        
        # 3层 GATv2，统一输出维度 dim
        self.convs.append(GATv2Conv(in_channels, hidden_channels, heads=heads, dropout=dropout, concat=True))
        self.norms.append(nn.LayerNorm(dim))
        self.convs.append(GATv2Conv(dim, hidden_channels, heads=heads, dropout=dropout, concat=True))
        self.norms.append(nn.LayerNorm(dim))
        self.convs.append(GATv2Conv(dim, hidden_channels, heads=heads, dropout=dropout, concat=True))
        self.norms.append(nn.LayerNorm(dim))
        
        self.classifier = nn.Linear(dim, out_channels)

    def forward(self, x, edge_index):
        xs = []
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x = conv(x, edge_index)
            x = norm(x)
            if i < len(self.convs) - 1:
                x = F.gelu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            xs.append(x)
        # JKNet (Max) 聚合多跳特征
        out = torch.stack(xs, dim=0).max(dim=0)[0]
        return self.classifier(out)

# 降低 hidden_channels 至 64 配合 heads=4 保证显存安全 (<20GB)
model = GATv2JKModel(in_channels=x.shape[1], hidden_channels=64, out_channels=8, heads=4, dropout=0.25).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=6e-4)

# 10轮 Warmup + 余弦退火
epochs = 400
warmup_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=10)
cosine_scheduler = CosineAnnealingLR(optimizer, T_max=epochs-10, eta_min=1e-6)
scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[10])

criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

# ================= 4. 训练配置 =================
save_path = './model_save_vb17/cls_model/worker_0/round_3'
os.makedirs(save_path, exist_ok=True)

def save_softmax_csv(idx, probs, filename, path):
    cols = ['test_idx'] + [f'class_{i}' for i in range(8)]
    df = pd.DataFrame(probs, columns=cols[1:])
    df.insert(0, 'test_idx', idx.astype(int))
    df.to_csv(os.path.join(path, filename), index=False, float_format='%.4f')

best_val_acc = 0.0
patience = 40
wait = 0

# 清理潜在显存碎片
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# ================= 5. 训练循环 =================
for epoch in range(1, epochs + 1):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[train_idx_t], data.y[train_idx_t])
    loss.backward()
    optimizer.step()
    scheduler.step()

    if epoch % 10 == 0 or epoch == 1:
        model.eval()
        with torch.no_grad():
            out_eval = model(data.x, data.edge_index)
            val_pred = out_eval[val_idx_t].argmax(dim=1)
            val_acc = (val_pred == data.y[val_idx_t]).float().mean().item()

        print(f"Epoch {epoch:03d} | Val ACC: {val_acc:.4f} | Loss: {loss.item():.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            wait = 0
            torch.save(model.state_dict(), os.path.join(save_path, 'best_model.pth'))
            print(f"  -> New best model saved! Val ACC: {val_acc:.4f}")

            # 严格索引对齐保存验证集 Softmax
            val_probs = F.softmax(out_eval[val_idx_t], dim=1).cpu().numpy()
            save_softmax_csv(val_idx, val_probs, 'validation-softmax.csv', save_path)

            # 严格索引对齐保存测试集 Softmax (Eval模式)
            test_probs = F.softmax(out_eval[test_idx_t], dim=1).cpu().numpy()
            save_softmax_csv(test_idx, test_probs, 'B1-softmax.csv', save_path)

            # 保存 B1.csv
            test_pred = test_probs.argmax(axis=1)
            df_b1 = pd.DataFrame({'test_idx': test_idx.astype(int), 'label': test_pred})
            df_b1.to_csv(os.path.join(save_path, 'B1.csv'), index=False)
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

# ================= 6. 测试集 TTA 增强 =================
print("Starting TTA for test set...")
model.load_state_dict(torch.load(os.path.join(save_path, 'best_model.pth'), map_location=device))
model.train()  # 开启 Dropout
tta_runs = 50
tta_probs = np.zeros((len(test_idx), 8))

with torch.no_grad():
    for _ in range(tta_runs):
        out = model(data.x, data.edge_index)
        probs = F.softmax(out[test_idx_t], dim=1).cpu().numpy()
        tta_probs += probs
tta_probs /= tta_runs

# 覆盖最终输出文件
save_softmax_csv(test_idx, tta_probs, 'B1-softmax.csv', save_path)
final_pred = tta_probs.argmax(axis=1)
df_b1 = pd.DataFrame({'test_idx': test_idx.astype(int), 'label': final_pred})
df_b1.to_csv(os.path.join(save_path, 'B1.csv'), index=False)
print("TTA completed. Final B1.csv and B1-softmax.csv saved successfully.")
