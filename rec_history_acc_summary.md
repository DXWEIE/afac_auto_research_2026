
## 第1轮 | Worker-2 | NDCG@10: 0.1611
### 训练日志摘要
```
poch 210, Loss: 3.0891, Val NDCG@10: 0.1437
Epoch 220, Loss: 3.0921, Val NDCG@10: 0.1432
Epoch 230, Loss: 3.0849, Val NDCG@10: 0.1428
Epoch 240, Loss: 3.0692, Val NDCG@10: 0.1416
Epoch 250, Loss: 3.0550, Val NDCG@10: 0.1415
Epoch 260, Loss: 3.0424, Val NDCG@10: 0.1414
Epoch 270, Loss: 3.0056, Val NDCG@10: 0.1393
Epoch 280, Loss: 2.9738, Val NDCG@10: 0.1387
Epoch 290, Loss: 2.9266, Val NDCG@10: 0.1379
Epoch 300, Loss: 2.8703, Val NDCG@10: 0.1351
Running Validation TTA...
Running Test TTA...
Done.
```
### 结果总结
{
  "best_model_dir": "./model_save_vb17/rec_model/worker_2/round_0/best_model.pth",
  "best_ndcg": 0.1611,
  "model_and_params": "架构：SASRec（单向因果Transformer）；划分：UID随机80/20抽样；增强：无，仅定长截断与Padding；超参：lr=1e-3, wd=1e-4, dropout=0.2, dim=128, heads=4, layers=2, max_len=50, bs=512；特征：纯物品ID序列，无用户特征拼接/交叉；调度：CosineAnnealingLR(T_max=200)；损失正则：CE Loss(ignore=0)+梯度裁剪(max_norm=1.0)+权重衰减。",
  "training_diagnosis": "Loss持续下降但Val NDCG在Epoch 30达峰后持续下滑，呈现典型严重过拟合。早停机制因patience(40)与评估间隔(10)不匹配未能触发，导致模型在性能衰退期无效训练。Cosine调度T_max=200与总轮次300错位，后期LR过低可能加剧优化停滞。Train/Val趋势严重背离，泛化能力弱，验证集低指标预示线上效果将大幅衰减。",
  "conclusion_and_next": "评价：差。严重过拟合且峰值偏低。下轮需修复早停逻辑，引入Warmup与Label Smoothing，增加序列随机Mask增强，调大Dropout或缩减容量，并引入负采样策略提升泛化力。"
}

## 第2轮 | Worker-1 | NDCG@10: 0.0348
### 训练日志摘要
```
Epoch 010 | Loss: 2.7968 | Val NDCG@10: 0.0348
  -> Best model saved. NDCG: 0.0348
Epoch 020 | Loss: 1.4540 | Val NDCG@10: 0.0319
Epoch 030 | Loss: 1.0142 | Val NDCG@10: 0.0289
Epoch 040 | Loss: 0.7776 | Val NDCG@10: 0.0263
Epoch 050 | Loss: 0.6324 | Val NDCG@10: 0.0256
Early stopping at epoch 50
Running TTA for Test set...
Running TTA for Validation set...
Done. All files saved.
```
### 结果总结
{
  "best_model_dir": "./model_save_vb17/rec_model/worker_1/round_0/best_model.pth",
  "best_ndcg": 0.0348,
  "model_and_params": "架构：SASRec（单向Transformer，因果掩码）；验证集划分：基于UID的随机80/20划分；数据增强：训练期无，仅推理期启用30轮TTA(Dropout)；超参：lr=1e-3, weight_decay=1e-4, dropout=0.2, hidden/item_dim=128, MAX_SEQ_LEN=50, batch_size=256；用户特征：仅使用Item序列ID，无额外特征拼接/交叉；调度器：CosineAnnealingLR；损失与正则：CrossEntropyLoss(ignore_index=0)+梯度裁剪(max_norm=1.0)+权重衰减。",
  "training_diagnosis": "训练Loss持续下降(2.79→0.63)，但Val NDCG@10自Epoch10峰值0.0348后持续下滑至0.0256并触发早停。呈现典型严重过拟合：模型快速记忆训练数据但泛化能力急剧恶化。初始学习率偏高且正则不足，导致后期指标震荡下滑。极低NDCG表明排序能力弱，验证集与线上指标存在显著落差隐患，TTA难以弥补根本性过拟合。",
  "conclusion_and_next": "评价：差。严重过拟合且指标极低。下轮需降低学习率至1e-4，增强正则（提升dropout/weight_decay），引入序列随机Mask/裁剪增强，并优化早停阈值，优先解决泛化瓶颈以提升排序质量。"
}

## 第3轮 | Worker-2 | NDCG@10: 0.1528
### 训练日志摘要
```
poch 110, Loss: 5.5090, Val NDCG@10: 0.1512
Epoch 120, Loss: 5.4636, Val NDCG@10: 0.1519
Epoch 130, Loss: 5.4438, Val NDCG@10: 0.1513
Epoch 140, Loss: 5.4255, Val NDCG@10: 0.1513
Epoch 150, Loss: 5.3938, Val NDCG@10: 0.1518
Epoch 160, Loss: 5.3889, Val NDCG@10: 0.1523
Epoch 170, Loss: 5.3760, Val NDCG@10: 0.1528
Epoch 180, Loss: 5.3721, Val NDCG@10: 0.1522
Epoch 190, Loss: 5.3770, Val NDCG@10: 0.1524
Epoch 200, Loss: 5.3697, Val NDCG@10: 0.1523
Running Validation TTA...
Running Test TTA...
Done.
```
### 结果总结
{
  "best_model_dir": "./model_save_vb17/rec_model/worker_2/round_1/best_model.pth",
  "best_ndcg": 0.1528,
  "model_and_params": "架构为SASRec（单向因果Transformer自注意力）；验证集采用基于uid的随机80/20划分；训练期无显式增强，仅做序列定长截断/填充(MAX_LEN=30)，推理期启用30轮MC-Dropout TTA；核心超参：lr=5e-4, weight_decay=1e-4, dropout=0.3, d_model=64, n_heads=4, n_layers=2, batch_size=256；无用户侧特征，仅依赖item序列ID；调度器为CosineAnnealingLR(T_max=200)；损失采用CrossEntropyLoss(ignore_index=0)，配合梯度裁剪(max_norm=1.0)与权重衰减正则。",
  "training_diagnosis": "Loss从7.24平稳降至5.37，Val NDCG@10稳步升至Epoch 170峰值0.1528后进入平台期微幅震荡，未触发早停。整体收敛健康，但后期验证指标停滞而训练Loss仍降，提示轻微过拟合与模型容量瓶颈。余弦LR衰减合理无震荡。隐患在于验证集绝对指标偏低，若线上存在分布偏移或长尾物品，泛化落差风险较高。",
  "conclusion_and_next": "结论：良。收敛稳定但指标偏低且后期遇瓶颈。下轮建议：扩大d_model与层数，引入用户/物品侧特征，尝试Label Smoothing或InfoNCE损失，优化负采样策略以突破性能天花板。"
}

## 第4轮 | Worker-1 | NDCG@10: 0.1662
### 训练日志摘要
```
17
Epoch 60, Loss: 5.4738, Val NDCG@10: 0.1633
Epoch 70, Loss: 5.3538, Val NDCG@10: 0.1658
Epoch 80, Loss: 5.2640, Val NDCG@10: 0.1651
Epoch 90, Loss: 5.1826, Val NDCG@10: 0.1662
Epoch 100, Loss: 5.1320, Val NDCG@10: 0.1653
Epoch 110, Loss: 5.0888, Val NDCG@10: 0.1652
Epoch 120, Loss: 5.0602, Val NDCG@10: 0.1648
Epoch 130, Loss: 5.0443, Val NDCG@10: 0.1654
Epoch 140, Loss: 5.0403, Val NDCG@10: 0.1649
Epoch 150, Loss: 5.0363, Val NDCG@10: 0.1651
Running Validation TTA...
Running Test TTA...
Done.
```
### 结果总结
{
  "best_model_dir": "./model_save_vb17/rec_model/worker_1/round_1/best_model.pth",
  "best_ndcg": 0.1662,
  "model_and_params": "架构：SASRec（单向Transformer+因果掩码）；验证集划分：基于固定随机种子的80/20用户随机抽样；数据增强：无显式增强，仅固定MAX_LEN=30截断/零填充；关键超参：lr=5e-4, weight_decay=5e-4, dropout=0.3, hidden_dim=128, item_dim=128, MAX_SEQ_LEN=30, batch_size=256；用户特征：无额外特征，仅依赖序列ID与位置编码；调度器：CosineAnnealingLR(T_max=150)；损失与正则：CrossEntropyLoss(ignore_index=0)配合AdamW优化器及梯度裁剪(max_norm=1.0)。",
  "training_diagnosis": "Loss持续平稳下降至5.03，Val NDCG@10在Epoch 90达峰值0.1662后进入平台期并微幅震荡，未触发早停。Loss与指标趋势表明模型已充分学习但后期存在轻微过拟合或表征瓶颈。学习率余弦衰减合理但后期收益递减。验证集指标绝对值偏低，且未记录训练集指标，存在线上泛化落差隐患，TTA推理可部分缓解但无法根本提升表征能力。",
  "conclusion_and_next": "良。模型收敛平稳但指标偏低且后期陷入平台期。下轮建议：引入负采样或InfoNCE损失替代CE，增加序列随机掩码/打乱增强，扩大hidden_dim至256并调整学习率策略，尝试融合用户画像特征以突破瓶颈。"
}

## 第5轮 | Worker-2 | NDCG@10: 0.1647
### 训练日志摘要
```
poch 110, Loss: 5.7790, Val NDCG@10: 0.1596
Epoch 120, Loss: 5.7357, Val NDCG@10: 0.1610
Epoch 130, Loss: 5.7129, Val NDCG@10: 0.1623
Epoch 140, Loss: 5.6872, Val NDCG@10: 0.1625
Epoch 150, Loss: 5.6656, Val NDCG@10: 0.1635
Epoch 160, Loss: 5.6561, Val NDCG@10: 0.1640
Epoch 170, Loss: 5.6543, Val NDCG@10: 0.1641
Epoch 180, Loss: 5.6571, Val NDCG@10: 0.1647
Epoch 190, Loss: 5.6476, Val NDCG@10: 0.1640
Epoch 200, Loss: 5.6542, Val NDCG@10: 0.1643
Running Validation TTA...
Running Test TTA...
Done.
```
### 结果总结
{
  "best_model_dir": "./model_save_vb17/rec_model/worker_2/round_2/best_model.pth",
  "best_ndcg": 0.1647,
  "model_and_params": "1.序列模型架构：SASRec（单向因果Transformer自注意力）；2.用户验证集划分策略：基于UID的随机80/20抽样；3.序列数据增强手段：训练期无显式增强，仅做定长截断/填充(MAX_LEN=40)，推理期采用30轮MC-Dropout TTA；4.关键超参：lr=3e-4, weight_decay=5e-4, dropout=0.4, d_model=128, n_heads=4, n_layers=2, MAX_SEQ_LEN=40, batch_size=256；5.用户特征处理逻辑：无，纯Item ID序列建模；6.学习率调度器类型：CosineAnnealingLR(T_max=200)；7.损失函数与正则策略：CrossEntropyLoss(ignore_index=0)，配合梯度裁剪(max_norm=1.0)与AdamW权重衰减。",
  "training_diagnosis": "Loss从14.97平稳降至5.65，Val NDCG@10从0.0811稳步升至Epoch 180峰值0.1647后微幅震荡，未触发早停。整体收敛健康，无严重过拟合或欠拟合，后期学习率衰减合理但增益放缓。Train/Val趋势一致，但绝对NDCG偏低提示表征瓶颈。隐患：随机UID划分未考虑时序漂移，TTA平滑可能掩盖真实泛化能力，存在线下线上指标落差风险。",
  "conclusion_and_next": "评级：良。收敛稳定但指标偏低。下轮建议：改用时序划分验证集对齐线上分布；提升d_model/层数或引入Label Smoothing；融合用户/物品侧特征并优化负采样策略。"
}

## 第6轮 | Worker-1 | NDCG@10: 0.1698
### 训练日志摘要
```
40
Epoch 60, Loss: 5.8214, Val NDCG@10: 0.1584
Epoch 70, Loss: 5.7207, Val NDCG@10: 0.1617
Epoch 80, Loss: 5.6553, Val NDCG@10: 0.1650
Epoch 90, Loss: 5.5916, Val NDCG@10: 0.1672
Epoch 100, Loss: 5.5530, Val NDCG@10: 0.1680
Epoch 110, Loss: 5.5151, Val NDCG@10: 0.1683
Epoch 120, Loss: 5.4898, Val NDCG@10: 0.1698
Epoch 130, Loss: 5.4844, Val NDCG@10: 0.1694
Epoch 140, Loss: 5.4737, Val NDCG@10: 0.1697
Epoch 150, Loss: 5.4693, Val NDCG@10: 0.1695
Running Validation TTA...
Running Test TTA...
Done.
```
### 结果总结
{
  "best_model_dir": "./model_save_vb17/rec_model/worker_1/round_2/best_model.pth",
  "best_ndcg": 0.1698,
  "model_and_params": "1.架构：SASRec单向Transformer；2.验证划分：固定Seed随机抽样20%UID；3.序列增强：无，仅MAX_LEN=50截断/零填充；4.超参：lr=3e-4, wd=1e-3, dropout=0.3, hidden/item_dim=128, bs=256, 无label_smoothing；5.用户特征：无，纯序列ID+位置编码；6.调度器：CosineAnnealingLR(T_max=150)；7.损失正则：CE Loss(ignore=0)+梯度裁剪(1.0)+AdamW权重衰减。",
  "training_diagnosis": "Loss平稳降至5.47，NDCG@10于120轮达峰0.1698后进入平台期，未触发早停。整体收敛健康，无过拟合或LR震荡。但指标绝对值偏低，提示模型欠拟合。TTA利用train模式Dropout做概率集成，虽平滑预测但引入方差。随机划分验证集与线上时序分布不一致，叠加固定截断与无用户特征，线上指标存在显著落差隐患。",
  "conclusion_and_next": "结论：良。收敛稳定但表征力不足。下轮需：增大hidden_dim与层数；尝试动态序列长度；引入用户侧特征交叉；验证集改为时间切分；对比确定性推理与TTA效果。"
}

## 第7轮 | Worker-0 | NDCG@10: 0.0477
### 训练日志摘要
```
Loading data...
Start training...
Epoch 010 | Loss: 5.3819 | Val NDCG@10: 0.0477
Epoch 020 | Loss: 4.5027 | Val NDCG@10: 0.0466
Epoch 030 | Loss: 3.9759 | Val NDCG@10: 0.0424
Epoch 040 | Loss: 3.6263 | Val NDCG@10: 0.0407
Epoch 050 | Loss: 3.3738 | Val NDCG@10: 0.0402
Early stopping at epoch 50
Loading best model for TTA...
Running TTA for Test set...
Running TTA for Validation set...
All outputs saved successfully.
```
### 结果总结
{
  "best_model_dir": "./model_save_vb17/rec_model/worker_0/round_0/best_model.pth",
  "best_ndcg": 0.0477,
  "model_and_params": "架构：SASRec（单向Transformer+因果掩码）；验证划分：固定Seed随机抽样20%UID；序列增强：无，仅定长截断与前向Padding；超参：LR=1e-3, wd=1e-4, dropout=0.2, dim=64, max_len=50, bs=256；用户特征：无，纯序列ID+位置编码；调度器：CosineAnnealingLR；损失正则：CE Loss(ignore=0)+AdamW权重衰减+梯度裁剪。",
  "training_diagnosis": "Loss持续下降但Val NDCG@10自Epoch10峰值0.0477后单调递减至0.0402并触发早停，呈典型过拟合。初始LR偏高且无Warmup致指标快速见顶后衰退，Dropout与TTA未能有效抑制。验证集指标绝对值极低，提示数据稀疏或表征弱，存在线上指标大幅低于验证集的隐患。",
  "conclusion_and_next": "差。严重过拟合且指标偏低。下轮需：引入LR Warmup与更低初始LR，增强正则（调大dropout/加label smoothing），引入序列级数据增强（随机Mask/打乱），并排查数据稀疏性与负采样策略。"
}

## 第8轮 | Worker-2 | NDCG@10: 0.1661
### 训练日志摘要
```
poch 110, Loss: 5.6256, Val NDCG@10: 0.1625
Epoch 120, Loss: 5.5852, Val NDCG@10: 0.1637
Epoch 130, Loss: 5.5714, Val NDCG@10: 0.1645
Epoch 140, Loss: 5.5301, Val NDCG@10: 0.1651
Epoch 150, Loss: 5.5157, Val NDCG@10: 0.1661
Epoch 160, Loss: 5.5103, Val NDCG@10: 0.1660
Epoch 170, Loss: 5.4981, Val NDCG@10: 0.1657
Epoch 180, Loss: 5.4907, Val NDCG@10: 0.1657
Epoch 190, Loss: 5.5037, Val NDCG@10: 0.1660
Epoch 200, Loss: 5.4931, Val NDCG@10: 0.1659
Running Validation TTA...
Running Test TTA...
Done.
```
### 结果总结
{
  "best_model_dir": "./model_save_vb17/rec_model/worker_2/round_3/best_model.pth",
  "best_ndcg": 0.1661,
  "model_and_params": "架构：SASRec(单向因果Transformer)；验证集划分：固定种子随机用户抽样(80/20)；数据增强：训练期无，推理期采用Dropout TTA(30轮)；超参：lr=3e-4, weight_decay=1e-4, dropout=0.3, d_model=128, n_heads=4, n_layers=2, MAX_LEN=20, batch_size=512；用户特征：8维类别特征经双层MLP映射后与序列表征相加；调度器：CosineAnnealingLR(T_max=200)；损失与正则：CrossEntropyLoss(ignore_index=0)+梯度裁剪(max_norm=1.0)+权重衰减。",
  "training_diagnosis": "Loss持续平稳下降至5.49，Val NDCG@10于Epoch 150达峰值0.1661后进入平台期并微幅震荡，未触发早停。训练后期Loss仍降但指标停滞，呈现轻微过拟合/表征饱和迹象。学习率衰减合理，无剧烈震荡。绝对指标偏低，提示序列截断(MAX_LEN=20)可能丢失长程依赖，且交叉熵损失在推荐场景下区分度有限，验证集与线上指标可能存在因负样本分布差异导致的落差隐患。",
  "conclusion_and_next": "结论：良。模型收敛稳定但绝对性能偏低。下轮建议：1.放宽MAX_LEN或引入动态截断；2.替换为BPR/InfoNCE损失或引入难负样本采样；3.微调dropout与学习率预热策略，提升泛化与排序区分度。"
}

## 第9轮 | Worker-0 | NDCG@10: 0.1327
### 训练日志摘要
```
33
Epoch 60, Loss: 6.9437, Val NDCG@10: 0.1176
Epoch 70, Loss: 6.8567, Val NDCG@10: 0.1224
Epoch 80, Loss: 6.8172, Val NDCG@10: 0.1244
Epoch 90, Loss: 6.7702, Val NDCG@10: 0.1273
Epoch 100, Loss: 6.7572, Val NDCG@10: 0.1293
Epoch 110, Loss: 6.7254, Val NDCG@10: 0.1314
Epoch 120, Loss: 6.7296, Val NDCG@10: 0.1315
Epoch 130, Loss: 6.7251, Val NDCG@10: 0.1325
Epoch 140, Loss: 6.7214, Val NDCG@10: 0.1324
Epoch 150, Loss: 6.7240, Val NDCG@10: 0.1327
Running Validation TTA...
Running Test TTA...
Done.
```
### 结果总结
{
  "best_model_dir": "./model_save_vb17/rec_model/worker_0/round_1/best_model.pth",
  "best_ndcg": 0.1327,
  "model_and_params": "架构：SASRec（基于Transformer Encoder的单向因果自注意力序列模型）；验证集划分：基于UID的随机80/20划分（固定种子666）；数据增强：训练期无增强仅做定长截断/填充(MAX_LEN=20)，推理期采用30轮Dropout开启的TTA；关键超参：lr=2e-4, weight_decay=5e-4, dropout=0.4, label_smoothing=0.1, d_model=128, n_heads=4, n_layers=2, MAX_SEQ_LEN=20, batch_size=256；用户特征：无显式特征，仅依赖Item ID序列；学习率调度：CosineAnnealingLR(T_max=150)；损失与正则：CrossEntropyLoss(label_smoothing=0.1, ignore_pad)配合AdamW及梯度裁剪(max_norm=1.0)。",
  "training_diagnosis": "Loss与Val NDCG均单调改善，无震荡或早停，学习率衰减节奏合理。但Val NDCG绝对值偏低且后期增速放缓，结合纯ID序列输入，判断为欠拟合/容量瓶颈而非过拟合。训练Loss持续下降但验证指标提升有限，存在特征表达不足导致的泛化天花板。此外，推理期TTA虽提升离线分数，但会引入分布偏移与延迟，需警惕线上指标落差。",
  "conclusion_and_next": "结论：良。收敛稳定但性能遇瓶颈。下轮方向：引入用户/物品侧特征增强表征；扩大d_model与序列长度；替换CE为BPR/InfoNCE损失；移除TTA，改用训练期随机Mask或负采样提升泛化。"
}

## 第10轮 | Worker-2 | NDCG@10: 0.1544
### 训练日志摘要
```
poch 110, Loss: 6.0708, Val NDCG@10: 0.1480
Epoch 120, Loss: 6.0357, Val NDCG@10: 0.1508
Epoch 130, Loss: 5.9981, Val NDCG@10: 0.1506
Epoch 140, Loss: 5.9557, Val NDCG@10: 0.1515
Epoch 150, Loss: 5.9562, Val NDCG@10: 0.1530
Epoch 160, Loss: 5.9401, Val NDCG@10: 0.1539
Epoch 170, Loss: 5.9256, Val NDCG@10: 0.1544
Epoch 180, Loss: 5.9160, Val NDCG@10: 0.1540
Epoch 190, Loss: 5.9091, Val NDCG@10: 0.1544
Epoch 200, Loss: 5.9085, Val NDCG@10: 0.1541
Running Validation TTA...
Running Test TTA...
Done.
```
### 结果总结
{
  "best_model_dir": "./model_save_vb17/rec_model/worker_2/round_4/best_model.pth",
  "best_ndcg": 0.1544,
  "model_and_params": "架构：SASRec（单向Causal Transformer）；验证划分：固定Seed 666随机打乱UID取后20%；序列增强：无显式增强，仅固定MAX_LEN=50截断/零填充；关键超参：lr=2e-4, weight_decay=5e-4, dropout=0.3, hidden/item_dim=128, n_heads=4, n_layers=2, batch_size=512, 无label_smoothing；用户特征：无独立特征，仅依赖序列ID与位置编码；调度器：CosineAnnealingLR(T_max=200)；损失与正则：CrossEntropyLoss(ignore_index=0)结合AdamW权重衰减与梯度裁剪(max_norm=1.0)。",
  "training_diagnosis": "Loss从23.38平稳降至5.91，Val NDCG@10升至0.1544后于170-200epoch进入平台期微幅震荡，未触发早停。整体收敛良好，但后期Loss微降而指标停滞，提示轻微过拟合或模型容量触及瓶颈。学习率余弦衰减与梯度裁剪保障训练稳定，但绝对指标偏低，验证集与线上高分场景可能存在落差；TTA阶段强制model.train()利用Dropout做概率平均，虽能平滑输出但可能引入推理方差。",
  "conclusion_and_next": "训练平稳收敛但指标偏低（良）。下轮建议：提升d_model与层数增强表征；引入负采样或对比学习优化排序；尝试动态序列长度与数据增强；调整学习率预热与早停阈值以突破性能瓶颈。"
}
