
## 第1轮 | Worker-0 | ACC: 0.3709
### 训练日志摘要
```
l saved! Val Acc: 0.3709
Epoch 280 | Loss: 1.7964 | Val Acc: 0.3644
Epoch 290 | Loss: 1.7827 | Val Acc: 0.3619
Epoch 300 | Loss: 1.7948 | Val Acc: 0.3619
Training and inference completed successfully.
```
### 结果总结
{
  "best_model_dir": "./model_save_vb17/cls_model/worker_0/round_0/best_model.pth",
  "best_acc": 0.3709,
  "model_and_params": "3层GATv2图注意力网络，前两层4头注意力(concat=True)接LayerNorm与ELU，第三层单头输出。关键超参：lr=1e-3，weight_decay=5e-4，hidden_dim=256，heads=4，dropout=0.4，epochs=300。核心操作：全图批量训练、余弦退火调度、交叉熵损失、80/20划分、测试集50次TTA推理，无残差/跳跃连接或DropEdge等机制。",
  "training_diagnosis": "训练Loss持续平稳下降至1.79，验证集Acc从0.26升至0.3709(Epoch 270)后进入平台期微幅震荡，未触发早停。整体呈欠拟合特征，绝对精度偏低(8分类仅37%)，表明模型容量或图信息利用不足；0.4的高Dropout可能过度抑制学习，学习率衰减合理但后期收益饱和，Train-Val差距显示轻微过拟合倾向。",
  "conclusion_and_next": "结论：良。模型具备基础学习能力但精度偏低，存在表征瓶颈。下一步：降低Dropout至0.2，引入残差连接或JK机制，尝试学习率Warmup，并优化图拓扑或特征工程。"
}

## 第2轮 | Worker-1 | ACC: 0.4722
### 训练日志摘要
```
665 | Loss: 1.1075
Epoch 290 | Val ACC: 0.4436 | Loss: 1.0520
Epoch 300 | Val ACC: 0.4624 | Loss: 1.0409
Starting TTA for test set...
TTA completed. Final B1.csv and B1-softmax.csv saved successfully.
```
### 结果总结
{
  "best_model_dir": "./model_save_vb17/cls_model/worker_1/round_0",
  "best_acc": 0.4722,
  "model_and_params": "2层GATv2图注意力网络，首层采用4头注意力+BatchNorm+ReLU+Dropout(0.5)，次层单头输出；无残差/跳跃连接/Jumping Knowledge/DropEdge等机制。关键超参：lr=0.001，weight_decay=5e-4，hidden_channels=256，全图训练。核心操作：添加自环与边去重，交叉熵损失，Adam优化，验证集早停(patience=40)，测试集采用50次Dropout开启的TTA概率平均增强。",
  "training_diagnosis": "Loss从3.15平稳降至1.04，Val ACC在Epoch 190达峰值0.4722后进入平台期震荡。训练Loss持续下降而验证指标停滞，表明出现轻微过拟合或模型表征瓶颈。学习率无衰减导致后期优化收益递减，因patience较大未触发早停。整体收敛稳定，无剧烈震荡。",
  "conclusion_and_next": "效果一般（良）。验证集准确率停滞于47.2%，存在轻微过拟合。下轮建议：引入学习率衰减策略，降低Dropout或增加权重衰减，尝试引入残差连接/特征标准化，并探索更深层GNN架构以突破瓶颈。"
}

## 第3轮 | Worker-2 | ACC: 0.1348
### 训练日志摘要
```
Loss: 1.8637, Val Acc: 0.1299
Epoch 300, Loss: 1.8608, Val Acc: 0.1348
  -> New best model saved. Val Acc: 0.1348
Training finished. Loading best model for TTA...
Final TTA results saved successfully.
```
### 结果总结
{
  "best_model_dir": "./model_save_vb17/cls_model/worker_2/round_0/best_model.pth",
  "best_acc": 0.1348,
  "model_and_params": "2层GATv2图注意力网络，首层4头注意力(concat=True)接ELU激活，次层单头输出。无归一化层、残差或跳跃连接。核心超参：lr=0.005，weight_decay=5e-4，hidden=256，dropout=0.5，全图训练300轮。采用交叉熵损失，Adam优化，验证集80/20划分，最终使用50轮Dropout TTA进行预测概率平滑。",
  "training_diagnosis": "Loss持续下降（2.74→1.86）但Val Acc长期在0.04~0.13震荡，最终0.1348仅略高于随机基线。Train-Val表现严重脱节，属典型欠拟合。高Dropout(0.5)与无归一化/残差设计导致梯度传播受阻，模型未能有效捕获图结构特征，学习率尚可但架构表达能力不足。",
  "conclusion_and_next": "结论：差。模型仅略优于随机猜测，特征提取失败。下轮建议：降低Dropout至0.2~0.3，引入LayerNorm或残差连接稳定训练；尝试GCN/GraphSAGE基线对比；检查节点特征质量与图连通性，必要时调整学习率或增加隐藏层维度。"
}

## 第4轮 | Worker-1 | ACC: 0.5008
### 训练日志摘要
```
902 | Loss: 0.5992
Epoch 390 | Val ACC: 0.4910 | Loss: 0.5928
Epoch 400 | Val ACC: 0.4902 | Loss: 0.5983
Starting TTA for test set...
TTA completed. Final B1.csv and B1-softmax.csv saved successfully.
```
### 结果总结
{
  "best_model_dir": "./model_save_vb17/cls_model/worker_1/round_1/best_model.pth",
  "best_acc": 0.5008,
  "model_and_params": "GATv2图注意力网络(3层，4头注意力，LayerNorm归一化，JKNet Max跳跃连接聚合多跳特征，Dropout=0.2)。关键超参：lr=5e-4，weight_decay=1e-4，hidden=256，全图训练。核心操作：Z-score+PCA(95%)特征降维，Label Smoothing(0.1)交叉熵，余弦退火调度，Early Stopping，50次Dropout TTA推理。",
  "training_diagnosis": "训练Loss持续平稳下降至0.59，但Val ACC在Epoch 110达0.5008后陷入平台期微幅震荡。Loss与指标明显背离，表明模型出现轻度过拟合与泛化瓶颈。余弦退火前期收敛有效，后期未能跳出局部最优。因评估间隔较长未触发早停。整体处于收敛停滞状态。",
  "conclusion_and_next": "效果中等。模型泛化遇瓶颈，验证集精度停滞。下轮建议：增强正则化(如DropEdge/增大Weight Decay)、引入学习率Warmup、优化图拓扑或特征表示，尝试调整JK聚合策略以突破精度瓶颈。"
}

## 第5轮 | Worker-2 | ACC: 0.4085
### 训练日志摘要
```
036 | Loss: 0.5382
Epoch 390 | Val ACC: 0.4011 | Loss: 0.5359
Epoch 400 | Val ACC: 0.4003 | Loss: 0.5375
Starting TTA for test set...
TTA completed. Final B1.csv and B1-softmax.csv saved successfully.
```
### 结果总结
{
  "best_model_dir": "./model_save_vb17/cls_model/worker_2/round_1",
  "best_acc": 0.4085,
  "model_and_params": "GATv2结合Jumping Knowledge(JK-Concat)架构，3层图注意力层，每层后接LayerNorm与Dropout(0.2)，前两层4头注意力(concat=True)，末层单头。核心超参：lr=1e-3，weight_decay=1e-4，hidden=256，heads=4，CosineAnnealingLR(T_max=400)。预处理含StandardScaler与PCA(95%方差)，损失函数为Label Smoothing(0.1)交叉熵，含早停(patience=50)及50次Dropout TTA增强。",
  "training_diagnosis": "训练Loss持续下降至0.53，但验证集ACC在Epoch 60达峰值0.4085后陷入0.39~0.405平台期震荡，未随Loss下降而提升。表明模型在60轮后出现明显过拟合与泛化瓶颈，早停机制未触发或日志未截断。学习率余弦衰减未能有效打破性能停滞，呈现典型的训练集持续优化、验证集提前饱和特征。",
  "conclusion_and_next": "结论：中等。模型具备基础学习能力但泛化遇阻。下轮建议：增强正则化(如DropEdge/调大weight_decay)或简化JK结构；引入学习率Warmup或早停；检查图连通性与特征质量，尝试调整PCA维度或引入边特征。"
}

## 第6轮 | Worker-0 | ACC: 0.4436
### 训练日志摘要
```
412 | Loss: 0.1441
Epoch 390 | Val ACC: 0.4412 | Loss: 0.1322
Epoch 400 | Val ACC: 0.4412 | Loss: 0.1475
Starting TTA for test set...
TTA completed. Final B1.csv and B1-softmax.csv saved successfully.
```
### 结果总结
{
  "best_model_dir": "./model_save_vb17/cls_model/worker_0/round_1",
  "best_acc": 0.4436,
  "model_and_params": "GATv2架构结合Jumping Knowledge(Max聚合)与残差连接，内置LayerNorm归一化、DropPath(0.05)及4头注意力。关键超参：lr=5e-4(Cosine退火)，hidden=256，3层GAT，dropout=0.2。核心操作：StandardScaler+PCA(95%)特征降噪，CrossEntropyLoss，EarlyStopping(patience=50)，测试集50次Dropout TTA。",
  "training_diagnosis": "训练Loss持续稳定下降至0.15，但验证集ACC在Epoch 110达峰值0.4436后进入平台期并小幅震荡，后续未突破。Loss与Val ACC走势背离明显，表明模型在110轮后已严重过拟合。EarlyStopping因指标未持续恶化未触发，但泛化已停滞。后期学习率过低可能限制优化。",
  "conclusion_and_next": "状态良。前期收敛快但中后期严重过拟合，泛化遇瓶颈。下轮需强化正则化（增大Dropout/DropPath或引入DropEdge），优化早停策略与学习率调度，并探索图数据增强或特征交互以突破精度瓶颈。"
}

## 第7轮 | Worker-1 | ACC: 0.4518
### 训练日志摘要
```
 Val ACC: 0.4453 | Loss: 0.3664 | LR: 1.69e-06
Epoch 400 | Val ACC: 0.4461 | Loss: 0.3540 | LR: 1.01e-06
Starting TTA for test set...
TTA completed. Final B1.csv and B1-softmax.csv saved successfully.
```
### 结果总结
{
  "best_model_dir": "./model_save_vb17/cls_model/worker_1/round_2",
  "best_acc": 0.4518,
  "model_and_params": "3层GATv2图注意力网络，采用LayerNorm归一化、残差连接与JK-Sum多跳特征聚合，4头注意力机制。关键超参：隐藏维度256，初始学习率3e-4（Cosine衰减+10%Warmup），Dropout 0.15，训练400轮。核心操作：特征Z-score标准化，Focal Loss(gamma=2.0)优化难分样本，AdamW优化器，Early Stopping(patience=50)，测试阶段采用Dropout TTA概率平均。",
  "training_diagnosis": "训练过程平稳，验证集准确率在250轮达峰值0.4518后进入平台期，Loss持续下降但Val ACC未同步提升，呈现轻微过拟合与表征饱和。学习率衰减平滑无震荡，未触发早停。整体收敛稳定，但模型容量或特征判别力已达当前配置瓶颈。",
  "conclusion_and_next": "训练效果良。模型收敛稳定但准确率遇瓶颈，存在轻微过拟合。下轮建议：引入DropEdge/特征掩码等图增强，调整Focal Loss参数或加标签平滑，尝试调整JK聚合方式或增加权重衰减以突破性能上限。"
}

## 第8轮 | Worker-2 | ACC: 0.4616
### 训练日志摘要
```
510 | Loss: 0.0309
Epoch 390 | Val ACC: 0.4510 | Loss: 0.0323
Epoch 400 | Val ACC: 0.4510 | Loss: 0.0298
Starting TTA for test set...
TTA completed. Final B1.csv and B1-softmax.csv saved successfully.
```
### 结果总结
{
  "best_model_dir": "./model_save_vb17/cls_model/worker_2/round_2/best_model.pth",
  "best_acc": 0.4616,
  "model_and_params": "GATv2图注意力网络(3层，4头注意力，LayerNorm归一化，残差直连，JK-Max跳跃聚合，Dropout=0.2)。关键超参：lr=5e-4，weight_decay=1e-4，hidden_dim=256，全图训练。核心操作：Z-score+PCA(95%)特征预处理，5轮Warmup+余弦退火调度，CrossEntropyLoss，50次Dropout TTA测试时增强。",
  "training_diagnosis": "Loss持续平稳下降至0.03，验证集ACC在Epoch 280达峰值0.4616后进入平台期小幅震荡，未触发早停。训练Loss极低而验证ACC仅0.46，存在明显泛化差距，呈现轻度过拟合。学习率调度合理无后期震荡，但验证集性能已触瓶颈。",
  "conclusion_and_next": "训练状态良。模型收敛稳定但泛化遇瓶颈，存在过拟合倾向。下一轮建议：引入DropEdge/特征掩码增强正则化，调整JK聚合策略或尝试图对比学习预训练，并优化数据划分。"
}

## 第9轮 | Worker-0 | ACC: 0.5335
### 训练日志摘要
```
139 | Loss: 0.7033
Epoch 390 | Val ACC: 0.5163 | Loss: 0.7031
Epoch 400 | Val ACC: 0.5172 | Loss: 0.6835
Starting TTA for test set...
TTA completed. Final B1.csv and B1-softmax.csv saved successfully.
```
### 结果总结
{
  "best_model_dir": "./model_save_vb17/cls_model/worker_0/round_2/best_model.pth",
  "best_acc": 0.5335,
  "model_and_params": "GATv2图神经网络架构，使用LayerNorm归一化，无显式残差连接，采用4头多头注意力机制，通过Jumping Knowledge(Max)实现多跳特征跳跃聚合，使用0.3 Dropout（无DropPath/DropEdge/DropNode）。关键超参：全图训练，隐藏层256维，3层，学习率3e-4，权重衰减5e-4，8分类。核心操作：Z-score特征标准化、Label Smoothing(0.05)交叉熵损失、10轮线性Warmup+余弦退火学习率调度、50轮Dropout TTA测试时增强。",
  "training_diagnosis": "训练Loss从2.31平稳降至0.68，学习率调度合理。验证集ACC在140轮达峰值0.5335后进入平台期并小幅震荡，未再突破，而训练Loss持续下降，呈现典型轻微过拟合与容量瓶颈特征。模型在140轮后已充分拟合当前数据分布，继续训练收益极低且早停机制未有效触发。",
  "conclusion_and_next": "训练效果中等。模型前期收敛快，但140轮后验证集性能停滞，存在轻微过拟合。下一轮建议：增强正则化（如引入DropEdge/增大权重衰减）、调整JK聚合策略或适度增加模型容量，并收紧早停阈值以优化算力分配。"
}

## 第10轮 | Worker-1 | ACC: 0.5163
### 训练日志摘要
```
123 | Loss: 0.8918
Epoch 390 | Val ACC: 0.5131 | Loss: 0.8922
Epoch 400 | Val ACC: 0.5131 | Loss: 0.8971
Starting TTA for test set...
TTA completed. Final B1.csv and B1-softmax.csv saved successfully.
```
### 结果总结
{
  "best_model_dir": "./model_save_vb17/cls_model/worker_1/round_3",
  "best_acc": 0.5163,
  "model_and_params": "3层GATv2图注意力网络，结合Jumping Knowledge(Max)多跳聚合与独立Ego路径；使用LayerNorm归一化、DropPath(0.05)与Dropout(0.2)正则化，无显式残差但含JK跳跃连接；核心超参：lr=3e-4(Cosine退火)，hidden=256，heads=4，epochs=400；特征Z-score标准化，图结构无向化加自环，损失函数含0.05标签平滑，测试端采用50次Dropout TTA。",
  "training_diagnosis": "训练Loss与Val ACC同步平稳改善，全程未触发早停，无过拟合或震荡。日志未提供完整训练集指标，但单步Loss持续下降且验证集精度稳步提升，未见明显泛化鸿沟。末期收敛放缓且8分类精度仅51.6%，提示模型容量可能不足或存在轻微欠拟合，Cosine退火后期学习率过低可能限制了性能上限。",
  "conclusion_and_next": "训练平稳收敛，表现中等。下轮建议：提升模型容量（增加层数/隐藏维度），优化学习率调度策略，引入图结构或特征数据增强，并同步监控训练集精度以确认欠拟合问题。"
}

## 第11轮 | Worker-2 | ACC: 0.5302
### 训练日志摘要
```
237 | Loss: 1.1528
Epoch 340 | Val ACC: 0.5229 | Loss: 1.1493
Epoch 350 | Val ACC: 0.5237 | Loss: 1.1572
Starting TTA for test set...
TTA completed. Final B1.csv and B1-softmax.csv saved successfully.
```
### 结果总结
{
  "best_model_dir": "./model_save_vb17/cls_model/worker_2/round_3/best_model.pth",
  "best_acc": 0.5302,
  "model_and_params": "2层GATv2图注意力网络，集成LayerNorm与JK-Max跳跃知识聚合，无显式残差连接，采用4头注意力与0.25 Dropout。关键超参：lr=3e-4(5%预热+余弦退火)，weight_decay=5e-4，hidden=64，epochs=350。核心操作：特征Z-score标准化、边无向化+自环、标签平滑(0.1)交叉熵损失、50次Dropout TTA测试增强。",
  "training_diagnosis": "训练Loss持续平稳下降至1.15，Val ACC于260轮达峰值0.5302后进入平台期小幅震荡，未触发早停。Loss持续下降而Val ACC停滞，表明模型已触及当前架构与特征下的性能瓶颈，存在轻微过拟合倾向或表征容量不足。学习率预热与余弦退火策略合理，未出现学习缓慢或后期震荡。",
  "conclusion_and_next": "状态良。模型收敛稳定但验证集精度遇瓶颈。下轮建议：增加网络深度或引入残差连接提升容量；适度调高Dropout/权重衰减抑制过拟合；探索图结构增强或特征筛选策略以突破精度天花板。"
}

## 第12轮 | Worker-1 | ACC: 0.4788
### 训练日志摘要
```
583 | Loss: 0.6895
Epoch 390 | Val ACC: 0.4616 | Loss: 0.6920
Epoch 400 | Val ACC: 0.4608 | Loss: 0.7015
Starting TTA for test set...
TTA completed. Final B1.csv and B1-softmax.csv saved successfully.
```
### 结果总结
{
  "best_model_dir": "./model_save_vb17/cls_model/worker_1/round_4/best_model.pth",
  "best_acc": 0.4788,
  "model_and_params": "3层GATv2图注意力网络，采用LayerNorm归一化、4头注意力机制与ReLU激活，无显式残差连接但引入Jumping Knowledge(Max)跨层跳跃聚合。核心超参：学习率2e-4(AdamW)，隐藏维度320，权重衰减8e-4，全图训练。核心操作：特征Z-score标准化、标签平滑(0.05)、10轮线性Warmup+余弦退火调度、Dropout(0.25)正则化及50次TTA推理增强。",
  "training_diagnosis": "Loss从2.37平稳降至0.68，Val ACC在Epoch 180达峰值0.4788后进入平台期小幅震荡，未触发早停。训练后期Loss持续缓降但Val ACC停滞，呈现轻微过拟合与表征瓶颈。学习率Warmup与余弦退火策略有效，但模型对复杂图结构信息的利用率已达上限，需优化泛化能力。",
  "conclusion_and_next": "训练效果良。收敛稳定但精度遇瓶颈。下轮建议：引入残差连接与DropEdge抑制过拟合，扩大隐藏层或改用Concat JK聚合，并探索边特征增强以突破48%天花板。"
}

## 第13轮 | Worker-2 | ACC: 0.5253
### 训练日志摘要
```
 | Val ACC: 0.5000 | Loss: 1.1495
Epoch 160 | Val ACC: 0.5163 | Loss: 1.1093
Early stopping at epoch 163
Starting TTA for test set...
TTA completed. Final B1.csv and B1-softmax.csv saved successfully.
```
### 结果总结
{
  "best_model_dir": "./model_save_vb17/cls_model/worker_2/round_4",
  "best_acc": 0.5253,
  "model_and_params": "3层GATv2架构，采用LayerNorm归一化、4头注意力与Max Jumping Knowledge机制；无显式残差连接及DropPath/Edge/Node。核心超参：lr=2e-4，hidden=256，dropout=0.15，weight_decay=1e-3。预处理含Z-score与PCA(95%)，损失函数为CrossEntropy+LabelSmoothing(0.1)，调度器采用5%线性Warmup+余弦退火，推理阶段启用50次Dropout TTA。",
  "training_diagnosis": "Loss持续平稳下降至1.10，Val ACC于Epoch 120达峰值0.5253后进入平台期并微幅震荡，触发Early Stopping。学习率调度合理无后期震荡，未见严重过拟合，但验证集精度提升遇阻，提示模型表征能力或特征区分度存在瓶颈，属轻微欠拟合/泛化上限受限。",
  "conclusion_and_next": "结论：良。训练稳定收敛但精度止步52.5%。下轮建议：增大hidden_dim或加深网络；改用Concat/LSTM型JK聚合；引入DropEdge/特征掩码增强；排查类别不平衡并调整采样策略。"
}

## 第14轮 | Worker-1 | ACC: 0.5082
### 训练日志摘要
```
984 | Loss: 1.0790
Epoch 390 | Val ACC: 0.5033 | Loss: 1.0736
Epoch 400 | Val ACC: 0.5016 | Loss: 1.0684
Starting TTA for test set...
TTA completed. Final B1.csv and B1-softmax.csv saved successfully.
```
### 结果总结
{
  "best_model_dir": "./model_save_vb17/cls_model/worker_1/round_5",
  "best_acc": 0.5082,
  "model_and_params": "3层GATv2结合Jumping Knowledge(JK)特征拼接架构，每层后接LayerNorm归一化，采用4头注意力机制与0.15 Dropout，无残差/DropPath；关键超参：lr=2e-4, weight_decay=1e-4, hidden=256, heads=4, layers=3, 全图训练；核心操作：特征L2行归一化、图结构无向化+自环、5%线性Warmup+余弦退火调度、标准交叉熵损失、50次TTA推理。",
  "training_diagnosis": "Loss从2.37平稳降至1.06，Val ACC在Epoch 270达峰值0.5082后进入平台期震荡，未触发早停。整体收敛稳定，无明显过拟合，但后期指标停滞表明模型触及表征瓶颈或图信息利用饱和，学习率衰减合理但后期优化空间有限。",
  "conclusion_and_next": "表现中等，已收敛但遇瓶颈。下轮建议：增加JK层数或引入残差结构；尝试DropEdge/特征掩码增强泛化；微调学习率或引入标签平滑优化边界。"
}

## 第15轮 | Worker-0 | ACC: 0.5466
### 训练日志摘要
```
458 | Loss: 1.1279
Epoch 390 | Val ACC: 0.5417 | Loss: 1.1350
Epoch 400 | Val ACC: 0.5417 | Loss: 1.1335
Starting TTA for test set...
TTA completed. Final B1.csv and B1-softmax.csv saved successfully.
```
### 结果总结
{
  "best_model_dir": "./model_save_vb17/cls_model/worker_0/round_3",
  "best_acc": 0.5466,
  "model_and_params": "3层GATv2图注意力网络，集成LayerNorm归一化与Jumping Knowledge(Max)跨层聚合机制，采用4头注意力。核心超参：hidden_channels=64, heads=4, lr=3e-4, weight_decay=6e-4, dropout=0.25, epochs=400。核心操作：特征Z-score标准化、Label Smoothing(0.05)、10轮线性Warmup+余弦退火调度、Early Stopping(patience=40)及50次TTA推理增强。",
  "training_diagnosis": "训练Loss持续下降至1.13，Val ACC稳步升至0.5466，全程未触发早停。后期Val ACC在0.54附近微幅震荡，Train Loss与Val ACC趋势一致，未见明显过拟合。提升斜率放缓表明模型可能触及容量瓶颈或存在轻微欠拟合。学习率Warmup+余弦退火策略有效，整体收敛平稳健康。",
  "conclusion_and_next": "结论：良。训练稳定收敛且无过拟合，但8分类精度仅54.66%，仍有提升空间。下轮建议：适度增加hidden_dim或网络深度以突破容量瓶颈，尝试引入DropEdge或调整Dropout比例，并可微调学习率峰值或延长训练轮数。"
}

## 第16轮 | Worker-2 | ACC: 0.4436
### 训练日志摘要
```
363 | Loss: 0.0026
Epoch 390 | Val ACC: 0.4363 | Loss: 0.0024
Epoch 400 | Val ACC: 0.4355 | Loss: 0.0029
Starting TTA for test set...
TTA completed. Final B1.csv and B1-softmax.csv saved successfully.
```
### 结果总结
{
  "best_model_dir": "./model_save_vb17/cls_model/worker_2/round_5",
  "best_acc": 0.4436,
  "model_and_params": "GATv2图注意力网络，结合Ego-Neighbor分离与JK-Concat机制；使用LayerNorm归一化（无BN），含多头注意力(heads=4)与层间JK跳跃连接；3层结构，hidden=320；特征经Z-score标准化；优化器AdamW(lr=2e-4, wd=3e-4)，10轮Warmup+余弦退火；损失为CrossEntropy；正则化含Dropout(0.15)及TTA(40次)，未使用DropPath/Edge/Node。",
  "training_diagnosis": "训练Loss单调降至0.003，验证集ACC升至0.4436后进入平台期并小幅震荡，未触发早停。Train-Val差距显著，存在明显过拟合倾向，但绝对精度偏低表明模型对复杂图模式拟合不足或数据存在噪声/类别不平衡。学习率调度平稳无震荡，整体处于泛化瓶颈期。",
  "conclusion_and_next": "良。验证集峰值44.36%，泛化遇瓶颈。下轮建议：引入DropEdge/特征掩码增强鲁棒性，添加标签平滑或类别重加权，微调学习率衰减曲线，并检查数据分布与特征有效性。"
}

## 第17轮 | Worker-1 | ACC: 0.4534
### 训练日志摘要
```
.3738
Epoch 400 | Val ACC: 0.4453 | Loss: 0.3740
Starting TTA for test set...
TTA run 10/45
TTA run 20/45
TTA run 30/45
TTA run 40/45
TTA completed. Final B1.csv and B1-softmax.csv saved successfully.
```
### 结果总结
{
  "best_model_dir": "./model_save_vb17/cls_model/worker_1/round_6",
  "best_acc": 0.4534,
  "model_and_params": "3层GATv2图注意力架构，采用LayerNorm归一化与4头注意力机制，无显式残差连接，使用JK-Concat跳跃连接聚合多跳特征。关键超参：lr=2e-4(AdamW, wd=5e-4)，hidden=384，全图训练。核心操作：Z-score特征标准化、Label Smoothing(0.05)、5%线性Warmup+余弦退火调度、45次Dropout TTA推理。",
  "training_diagnosis": "训练Loss平稳下降至0.37，验证集ACC于Epoch 210达峰值0.4534后进入平台期并微幅震荡。Loss持续下降而Val ACC停滞，表明210轮后出现轻微过拟合或表征能力瓶颈。学习率调度合理，整体收敛稳定但性能触顶，未触发早停可能受精度波动影响。",
  "conclusion_and_next": "良。模型收敛稳定但验证集精度在45.3%遇瓶颈，后期轻微过拟合。下轮建议：增强正则化、引入残差连接优化JK聚合，或尝试图数据增强以突破性能上限。"
}

## 第18轮 | Worker-0 | ACC: 0.5212
### 训练日志摘要
```
082 | Loss: 0.2425
Epoch 390 | Val ACC: 0.5065 | Loss: 0.2384
Epoch 400 | Val ACC: 0.5098 | Loss: 0.2364
Starting TTA for test set...
TTA completed. Final B1.csv and B1-softmax.csv saved successfully.
```
### 结果总结
{
  "best_model_dir": "./model_save_vb17/cls_model/worker_0/round_4/best_model.pth",
  "best_acc": 0.5212,
  "model_and_params": "3层GCN架构，使用LayerNorm归一化、逐层残差连接与Jumping Knowledge(Concat)机制，无多头注意力/DropPath等；隐藏层128维，Dropout=0.2，全图训练；AdamW(lr=2e-3, wd=5e-4)，5%线性Warmup+余弦退火；特征Z-score标准化，图结构无向化+自环；交叉熵损失，推理采用50次TTA。",
  "training_diagnosis": "Loss从2.31平稳降至0.23，Val ACC在Epoch 270达0.5212后陷入平台期震荡。训练Loss持续下降而验证ACC停滞，表明模型已充分学习但泛化能力受限，未触发早停。学习率调度合理，整体呈现性能瓶颈/轻微欠拟合状态。",
  "conclusion_and_next": "训练效果良。模型收敛平稳但验证集精度遭遇瓶颈。下一轮建议：引入DropEdge/图增强提升泛化，调整JK融合策略或增大隐藏层，并针对低同质率优化邻接矩阵或引入异构图机制。"
}

## 第19轮 | Worker-0 | ACC: 0.4959
### 训练日志摘要
```
902 | Loss: 1.0953
Epoch 390 | Val ACC: 0.4910 | Loss: 1.1099
Epoch 400 | Val ACC: 0.4918 | Loss: 1.1101
Starting TTA for test set...
TTA completed. Final B1.csv and B1-softmax.csv saved successfully.
```
### 结果总结
{
  "best_model_dir": "./model_save_vb17/cls_model/worker_0/round_5/best_model.pth",
  "best_acc": 0.4959,
  "model_and_params": "3层GATv2图注意力网络，结合LayerNorm、GELU与Dropout(0.2)，采用Max Jumping Knowledge聚合多跳特征。核心超参：隐藏维度64，头数4，全图训练，AdamW(lr=3e-4, wd=5e-4)，10轮Warmup+余弦退火。预处理Z-score标准化，损失函数含0.05标签平滑，测试期启用50次Dropout TTA。",
  "training_diagnosis": "Loss从2.20平稳降至1.11，Val ACC从0.137稳步升至0.496后进入平台期，未见明显过拟合或剧烈震荡。学习率调度合理，初期Warmup有效，后期余弦退火稳定收敛。因每10轮评估一次且patience=40，未触发早停而自然跑满400轮。模型已充分学习但遭遇表达瓶颈。",
  "conclusion_and_next": "训练效果中等。模型平稳收敛但精度停滞于0.496。下一轮建议：增加网络深度/引入残差连接、尝试DropEdge/特征增强，或优化图拓扑结构以突破性能瓶颈。"
}

## 第20轮 | Worker-2 | ACC: 0.4404
### 训练日志摘要
```
387 | Loss: 0.2782
Epoch 390 | Val ACC: 0.4395 | Loss: 0.2762
Epoch 400 | Val ACC: 0.4387 | Loss: 0.2768
Starting TTA for test set...
TTA completed. Final B1.csv and B1-softmax.csv saved successfully.
```
### 结果总结
{
  "best_model_dir": "./model_save_vb17/cls_model/worker_2/round_6",
  "best_acc": 0.4404,
  "model_and_params": "GATv2结合Jumping Knowledge机制(3层，4头注意力，LayerNorm归一化，通过MaxPooling聚合各层特征)，全图训练。关键超参：初始lr=3e-4(10轮Warmup+余弦退火)，weight_decay=5e-4，hidden_dim=128，dropout=0.2，epochs=400。核心操作：特征Z-score标准化、边权对称归一化、Label Smoothing(0.05)交叉熵、20次Dropout TTA推理。",
  "training_diagnosis": "训练Loss持续平稳下降至0.276，验证集ACC稳步提升至0.4404后进入平台期微幅震荡，未触发早停。学习曲线健康无异常震荡，但验证集精度在44%左右停滞且训练Loss仍缓降，表明模型已触及当前配置瓶颈，存在轻微过拟合倾向，泛化能力受限。",
  "conclusion_and_next": "状态良。模型有效学习但精度卡在44%瓶颈。下轮建议：引入DropEdge/特征掩码增强鲁棒性，调整隐藏层维度或注意力头数，并检查类别分布以优化损失权重。"
}

## 第21轮 | Worker-1 | ACC: 0.4820
### 训练日志摘要
```
747 | Loss: 0.3440
Epoch 440 | Val ACC: 0.4739 | Loss: 0.3408
Epoch 450 | Val ACC: 0.4739 | Loss: 0.3408
Starting TTA for test set...
TTA completed. Final B1.csv and B1-softmax.csv saved successfully.
```
### 结果总结
{
  "best_model_dir": "./model_save_vb17/cls_model/worker_1/round_7",
  "best_acc": 0.482,
  "model_and_params": "GATv2结合Jumping Knowledge(Max)架构，3层图网络。含LayerNorm、残差连接与DropPath(0.05)，多头注意力(heads=4)。关键超参：hidden=128, dropout=0.2, lr=3e-4, weight_decay=5e-4, epochs=450。核心操作：Z-score标准化、Label Smoothing(0.05)、Linear Warmup+Cosine退火、20次TTA。",
  "training_diagnosis": "Loss从2.27平稳降至0.34，Val ACC稳步升至0.482(第320轮)后进入平台期。无明显过拟合或震荡，学习率调度合理。Val ACC停滞而Loss微降，表明模型已充分学习图结构，但受限于特征判别力或任务难度，泛化性能触及瓶颈。",
  "conclusion_and_next": "结论：良。训练稳定收敛无过拟合，但准确率遇瓶颈。下轮建议：引入图数据增强或特征扰动，调整JK聚合策略/增加网络宽度，并微调正则化强度以突破泛化上限。"
}

## 第22轮 | Worker-0 | ACC: 0.5351
### 训练日志摘要
```
180 | Loss: 0.8154
Epoch 390 | Val ACC: 0.5237 | Loss: 0.8105
Epoch 400 | Val ACC: 0.5221 | Loss: 0.7982
Starting TTA for test set...
TTA completed. Final B1.csv and B1-softmax.csv saved successfully.
```
### 结果总结
{
  "best_model_dir": "./model_save_vb17/cls_model/worker_0/round_6/best_model.pth",
  "best_acc": 0.5351,
  "model_and_params": "GATv2结合Jumping Knowledge(Concat)机制，3层结构，每层后接LayerNorm与GELU，采用4头注意力与0.2 Dropout。全图训练，隐藏维度128，8分类。特征经Z-score标准化，图结构强制无向化并加自环。优化器AdamW(lr=2e-4, wd=5e-4)，5%线性Warmup+余弦退火调度，CrossEntropyLoss。测试阶段采用50次Dropout TTA集成。",
  "training_diagnosis": "训练Loss从2.12平稳降至0.80，验证集ACC由0.15升至0.5351后进入平台期震荡。训练Loss持续下降而验证集指标停滞，存在轻微泛化瓶颈，未见严重过拟合或学习率后期震荡。早停机制未在日志中触发，完整跑满400轮。",
  "conclusion_and_next": "训练效果良。模型收敛稳定但验证集精度卡在53.5%瓶颈。下轮建议：增加模型深度或引入残差连接；尝试DropEdge/特征增强；调整学习率策略或引入标签平滑；排查类别不平衡与图拓扑质量。"
}

## 第23轮 | Worker-2 | ACC: 0.5090
### 训练日志摘要
```
rting TTA for test set...
  TTA run 10/45 completed.
  TTA run 20/45 completed.
  TTA run 30/45 completed.
  TTA run 40/45 completed.
TTA completed. Final B1.csv and B1-softmax.csv saved successfully.
```
### 结果总结
{
  "best_model_dir": "./model_save_vb17/cls_model/worker_2/round_7/best_model.pth",
  "best_acc": 0.5090,
  "model_and_params": "GATv2图注意力网络结合Jumping Knowledge(JK-Concat)机制，共2层，使用LayerNorm归一化与多头注意力(heads=4)，无显式残差连接及DropPath/Edge/Node。核心超参：lr=2e-4，weight_decay=5e-4，hidden_dim=128，dropout=0.2，全图训练。采用Z-score标准化、边对称化加自环、Linear Warmup+Cosine退火调度、CrossEntropyLoss，配合Early Stopping(patience=40)及基于Dropout的TTA(45次)增强。",
  "training_diagnosis": "训练Loss从2.14平稳降至1.07，Val ACC稳步升至0.509，全程无震荡且未触发早停。学习率Warmup+Cosine策略合理，后期衰减平稳。训练Loss持续下降与Val ACC稳步提升趋势一致，差距合理，排除过拟合；模型未达性能饱和，呈轻微欠拟合或容量瓶颈状态。",
  "conclusion_and_next": "良。训练平稳收敛无过拟合，但8分类准确率仅50.9%遇瓶颈。下轮建议：增加隐藏层维度或引入残差连接提升容量；尝试DropEdge/特征增强；微调学习率下限或引入标签平滑以突破平台期。"
}

## 第24轮 | Worker-0 | ACC: 0.4771
### 训练日志摘要
```
730 | Loss: 0.7634
Epoch 390 | Val ACC: 0.4747 | Loss: 0.7463
Epoch 400 | Val ACC: 0.4755 | Loss: 0.7560
Starting TTA for test set...
TTA completed. Final B1.csv and B1-softmax.csv saved successfully.
```
### 结果总结
{
  "best_model_dir": "./model_save_vb17/cls_model/worker_0/round_7/best_model.pth",
  "best_acc": 0.4771,
  "model_and_params": "GATv2架构(3层，4头注意力，隐藏层128)，结合LayerNorm、0.6/0.4残差直连、GELU激活与Dropout(0.2)，末端采用Max Jumping Knowledge聚合多跳特征。全图训练，学习率2.5e-4(AdamW)，权重衰减5e-4，配合5%线性Warmup与余弦退火。特征经Z-score标准化，损失函数含0.05标签平滑，测试阶段启用50次Dropout TTA增强。",
  "training_diagnosis": "训练Loss持续平稳下降至0.75左右，验证集ACC稳步提升至0.4771后进入平台期并小幅震荡，未触发早停。整体收敛平滑，无明显过拟合或学习率震荡。但验证集精度在47%左右停滞，表明模型可能触及当前架构与特征表达瓶颈，存在一定欠拟合或泛化能力上限。",
  "conclusion_and_next": "训练状态良。模型收敛稳定但精度遇瓶颈。下轮建议：增加隐藏层维度或网络深度；引入DropEdge等图增强或对比预训练；微调残差权重与JK聚合策略，以突破泛化上限。"
}

## 第25轮 | Worker-2 | ACC: 0.5098
### 训练日志摘要
```
008 | Loss: 0.0382
Epoch 390 | Val ACC: 0.5025 | Loss: 0.0490
Epoch 400 | Val ACC: 0.5033 | Loss: 0.0444
Starting TTA for test set...
TTA completed. Final B1.csv and B1-softmax.csv saved successfully.
```
### 结果总结
{
  "best_model_dir": "./model_save_vb17/cls_model/worker_2/round_8",
  "best_acc": 0.5098,
  "model_and_params": "GATv2图注意力网络(3层，4头，隐藏维256)，内置LayerNorm、逐层残差连接与GELU，末端采用Jumping Knowledge(Max)聚合。关键超参：lr=3e-4(AdamW)，wd=5e-4，全图Batch，Dropout=0.15。核心操作：Z-score标准化，5%线性Warmup+余弦退火调度，标准交叉熵损失，测试期50轮Dropout TTA。",
  "training_diagnosis": "训练Loss持续下降至0.044，模型充分拟合训练数据；Val ACC在Epoch 150达峰值0.5098后长期震荡停滞(0.49~0.505)，未随Loss下降而提升，呈现典型过拟合。学习率衰减合理，但泛化能力遇瓶颈，Early Stopping因日志间隔未触发，实际有效训练在150轮后已饱和。",
  "conclusion_and_next": "结论：良。训练充分但验证集停滞，呈典型过拟合。下轮需强化正则化(如DropEdge/特征掩码)、调整JKNet聚合策略或引入标签平滑，以缓解过拟合并提升泛化上限。"
}

## 第26轮 | Worker-1 | ACC: 0.4739
### 训练日志摘要
```
673 | Loss: 0.3593
Epoch 390 | Val ACC: 0.4673 | Loss: 0.3511
Epoch 400 | Val ACC: 0.4665 | Loss: 0.3501
Starting TTA for test set...
TTA completed. Final B1.csv and B1-softmax.csv saved successfully.
```
### 结果总结
{
  "best_model_dir": "./model_save_vb17/cls_model/worker_1/round_8",
  "best_acc": 0.4739,
  "model_and_params": "3层GATv2架构，自定义层融合自环线性投影、LayerNorm与GELU激活，采用4头注意力与Max Jumping Knowledge跨层聚合。全图训练，lr=3e-4（10轮线性预热+余弦退火），weight_decay=5e-4，hidden=64，dropout=0.2。特征Z-score标准化，使用0.05标签平滑交叉熵损失，配合5轮Dropout TTA。",
  "training_diagnosis": "Loss从2.1平稳降至0.35，Val ACC在280轮达峰值0.4739后进入平台期微幅震荡，未触发早停。学习率调度合理，训练稳定。后期Train Loss持续下降而Val ACC停滞，表明模型已充分拟合训练分布，验证集性能触及当前架构与特征表达的瓶颈，存在轻微过拟合倾向。",
  "conclusion_and_next": "训练效果良。收敛稳定但验证集精度后期饱和。下轮建议：扩大隐藏层维度或增加层数，引入DropEdge/特征扰动增强泛化，或尝试Mean/Concat JK聚合策略以突破瓶颈。"
}

## 第27轮 | Worker-0 | ACC: 0.5408
### 训练日志摘要
```
 TTA run 10/50 completed.
  TTA run 20/50 completed.
  TTA run 30/50 completed.
  TTA run 40/50 completed.
  TTA run 50/50 completed.
TTA completed. Final B1.csv and B1-softmax.csv saved successfully.
```
### 结果总结
{
  "best_model_dir": "./model_save_vb17/cls_model/worker_0/round_8",
  "best_acc": 0.5408,
  "model_and_params": "3层GATv2架构，含LayerNorm、残差直连、4头注意力与Dropout(0.15)，末端采用JK-Net Concat跳跃连接聚合多跳特征。关键超参：hidden=256, lr=2e-4, weight_decay=1e-4, epochs=400。核心操作：特征Z-score标准化、图对称化+自环、AdamW优化、5%线性Warmup+余弦退火、交叉熵损失、50次Dropout TTA。",
  "training_diagnosis": "Loss平稳降至0.09，Val ACC于Epoch 150达峰值0.5408后停滞震荡。Train Loss持续下降而Val ACC不升，呈现典型过拟合。因每10轮评估一次，Early Stopping计数器未达阈值致冗余训练。学习率衰减合理，但正则化强度不足限制泛化上限。",
  "conclusion_and_next": "良。前期收敛佳，150轮后过拟合明显。下轮需增强正则化（增大Dropout/Weight Decay或引入DropEdge），缩短评估间隔以激活Early Stopping，并尝试特征筛选或图数据增强。"
}

## 第28轮 | Worker-2 | ACC: 0.5155
### 训练日志摘要
```
025 | Loss: 0.1706
Epoch 390 | Val ACC: 0.5041 | Loss: 0.1773
Epoch 400 | Val ACC: 0.5033 | Loss: 0.1653
Starting TTA for test set...
TTA completed. Final B1.csv and B1-softmax.csv saved successfully.
```
### 结果总结
{
  "best_model_dir": "./model_save_vb17/cls_model/worker_2/round_9",
  "best_acc": 0.5155,
  "model_and_params": "GATv2结合Jumping Knowledge(Max)架构，3层图注意力网络，4头注意力，LayerNorm归一化，无显式残差，Dropout(0.15)。关键超参：hidden_channels=256，lr=3e-4，weight_decay=3e-4，全图Batch，400轮。核心操作：特征Z-score标准化、边无向化加自环、AdamW优化器、10轮线性Warmup+余弦退火调度、CrossEntropyLoss、50轮Dropout开启的TTA测试时增强。",
  "training_diagnosis": "训练Loss从2.29持续平稳降至0.165，优化充分；但Val ACC在Epoch 100达0.5155后长期震荡停滞于0.50-0.51区间。Train Loss极低而Val Acc无法突破，呈现明显过拟合与泛化瓶颈。学习率衰减策略合理，但模型容量或正则化强度未能有效匹配数据分布，导致后期验证集性能无法提升。",
  "conclusion_and_next": "良。模型收敛但验证集准确率在51.5%后陷入瓶颈，存在过拟合倾向。下轮建议：增强正则化（如引入DropEdge/特征掩码/调大Weight Decay），或调整JKNet聚合方式与学习率衰减策略，以突破泛化天花板。"
}
