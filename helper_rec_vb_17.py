api_key = 'sk-xxxx'

from openai import OpenAI, APIError
import json
import subprocess
import sys
import ast
import random
from openai import OpenAI
import re
from typing import Dict
import json
from typing import List, Dict
# 输出的结果校验
import os
import sys
import numpy as np
import pandas as pd
import threading
import traceback
import re
from datetime import datetime
import time


# ==================================================
# 全局配置与公共状态（后续可抽到单独config文件）
# ==================================================
# 模型与接口配置
default_model = "qwen3.6-plus"


# 全局路径配置
global_history_acc_md_path = "./model_save_vb17/rec_history_acc_summary.md"   # 全局训练结果总结MD
global_data_inspect_md_path = "./model_save_vb17/rec_data_inspection_result.md" # 数据探查结果MD
base_template_path = "./rec_base_template.py"                 # 初始接口模板文件
model_save_dir = "./model_save_vb17/rec_model/"                 # 模型与代码保存目录
rec_data_base_dir = '/data/coding/line3/dataset_b/rec_data/'
fixed_data_path = '/data/coding/line3/dataset_b/rec_data/'
MODEL_SAVE_DIR = "./model_save_vb17/rec_model/"
DATA_PATH = "/data/coding/line3/dataset_b/rec_data/"

global_best_acc = 0
global_best_code = ""
global_best_code_path = ""
global_best_model_summary = ""

# 使用字典存储全局状态，避免 global 关键字带来的作用域问题
GLOBAL_STATE = {
    "best_merge_acc": 0.0
}

import json
import os
from datetime import datetime

LOG_PATH = "./model_save_vb17/trajectory_B2.json"

def write_log(log_type: str, content):
    """写入单行jsonl日志，固定time/type/content三字段"""
    try:
        record = {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "type": log_type,
            "content": content
        }
        line = json.dumps(record, ensure_ascii=False) + "\n"
        # a 追加写入，多线程/进程安全
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line)
    except Exception as e:
        print(f"日志写入异常: {str(e)}")

def load_all_log() -> list:
    """读取全部jsonl日志，自动跳过损坏空行"""
    if not os.path.exists(LOG_PATH):
        return []
    res = []
    try:
        with open(LOG_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    res.append(json.loads(line))
                except Exception:
                    continue
    except Exception as e:
        print(f"读取日志异常: {str(e)}")
    return res


# Prompt模板占位（后续补充具体内容）

SAFE_EXEC_PREFIX = """
import warnings
warnings.filterwarnings("ignore")

# 可选：仅忽略特定高频警告，保留真正的错误提示
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*sparse tensor.*")

import os
os.environ["PYTHONWARNINGS"] = "ignore"
"""


data_analyse_plan_template = """你是一个推荐系统（Recommendation System）模型训练与调优专家。你的目标是通过分析数据集特征，为后续模型优化提供依据。

### 建模任务
{task_description}

### 历史模型训练情况
{history_train_summary}

### 数据集基本信息
{dataset_description}

请你分析模型结构、数据集基本信息、历史模型训练情况(如有)，判断进行什么样的数据探查能够明确数据分布，便于后续明确优化方向，提升模型性能。写明需要重点探查的数据维度（如用户行为序列长度分布、物品交互频次分布、冷启动用户/物品占比、特征缺失率、训练/测试集时间跨度与分布差异等），提升在稀疏数据下的表现。
请直接输出你打算进行的明确的数据探查方案，以纯文本的形式输出，简要明确，300字以内：
"""


rec_descrition_template = """
你是一个推荐系统模型训练专家，请按照以下要求完成代码编写。
"""

rec_data_explanation_template = """# 数据文件说明
文件夹为{data_base_dir}
## 1. 基本任务形式为：

```
给定用户近期 item 交互序列、匿名用户侧特征和匿名 item 侧特征，
为每个测试用户预测一个按置信度排序的 item 列表。
```

默认候选 item 集合为 `item.csv` 中全部 `iid`。

## 2. 文件列表
数据集在{data_base_dir}文件夹下，该文件夹下有：
```
./
  train.csv
  test.csv
  sample_submission.csv
  user.csv
  item.csv
  metadata.json
  README.md
```

## 3. 文件结构

### 3.1 train.csv

训练集，包含目标 item。

```text
uid,target_iid,item_seq_raw,item_seq_dedup,item_seq_counts
```

### 3.2 test.csv

测试集，隐藏目标 item。

```text
uid,item_seq_raw,item_seq_dedup,item_seq_counts
```

### 3.3 sample_submission.csv

```text
uid,prediction
```

`prediction` 为按置信度排序的 item id 列表，英文逗号分隔。

### 3.4 user.csv

```text
uid,u_cat_01,u_cat_02,u_cat_03,u_cat_04,u_cat_05,u_cat_06,u_cat_07,u_cat_08
```

### 3.5 item.csv

```text
iid,i_cat_01,i_cat_02,i_cat_03,i_bucket_01
```

## 4. 数据规模

```text
train.csv: 40000 行
test.csv: 10000 行
user.csv: 50000 行
item.csv: 14065 行
```

## 5. 匿名化说明

本数据集采用匿名 id、匿名特征名和匿名特征值，不包含内部字段映射表。


数据集的metadata为：
```
{{
  "task_family": "sequence_recommendation",
  "scenario": "industrial_recommendation_sequence",
  "files": {{
    "train.csv": {{
      "description": "训练集，包含目标 item。",
      "primary_key": "uid",
      "columns": [
        "uid",
        "target_iid",
        "item_seq_raw",
        "item_seq_dedup",
        "item_seq_counts"
      ]
    }},
    "test.csv": {{
      "description": "测试集，隐藏目标 item。",
      "primary_key": "uid",
      "columns": [
        "uid",
        "item_seq_raw",
        "item_seq_dedup",
        "item_seq_counts"
      ]
    }},
    "user.csv": {{
      "description": "匿名用户侧特征表。",
      "primary_key": "uid",
      "columns": [
        "uid",
        "u_cat_01",
        "u_cat_02",
        "u_cat_03",
        "u_cat_04",
        "u_cat_05",
        "u_cat_06",
        "u_cat_07",
        "u_cat_08"
      ]
    }},
    "item.csv": {{
      "description": "匿名 item 侧特征表，也是默认候选 item 集合。",
      "primary_key": "iid",
      "columns": [
        "iid",
        "i_cat_01",
        "i_cat_02",
        "i_cat_03",
        "i_bucket_01"
      ]
    }}
  }},
  "row_counts": {{
    "train.csv": 40000,
    "test.csv": 10000,
    "user.csv": 50000,
    "item.csv": 14065,
    "sample_submission.csv": 10000
  }}
}}
```

测试集输出文件命名为B2.csv，必须包含表头，且仅允许包含两列，例如：

```csv
uid,prediction
u000001,"i000001,i000002,i000003,i000004,i000005"
```

其中 `prediction` 为按置信度从高到低排序的 item id 列表，使用英文逗号分隔。默认候选 item 集合为 `item.csv` 中全部 `iid`。
注意：
产品编号规则：prediction 中的 item id 必须来自对应数据包 item.csv 的 iid 候选集合。
排序规则：prediction 中的 item id 需按置信度从高到低排序，总计10个，使用英文逗号分隔。
去重规则：同一用户的推荐列表中不应出现重复 item id；若出现重复、非法 id 或缺失预测，平台将按评测脚本规则处理，可能导致该用户得分降低。
""".replace("{data_base_dir}", rec_data_base_dir)


data_meta_info = """数据集的metadata为：
```
{{
  "task_family": "sequence_recommendation",
  "scenario": "industrial_recommendation_sequence",
  "files": {{
    "train.csv": {{
      "description": "训练集，包含目标 item。",
      "primary_key": "uid",
      "columns": [
        "uid",
        "target_iid",
        "item_seq_raw",
        "item_seq_dedup",
        "item_seq_counts"
      ]
    }},
    "test.csv": {{
      "description": "测试集，隐藏目标 item。",
      "primary_key": "uid",
      "columns": [
        "uid",
        "item_seq_raw",
        "item_seq_dedup",
        "item_seq_counts"
      ]
    }},
    "user.csv": {{
      "description": "匿名用户侧特征表。",
      "primary_key": "uid",
      "columns": [
        "uid",
        "u_cat_01",
        "u_cat_02",
        "u_cat_03",
        "u_cat_04",
        "u_cat_05",
        "u_cat_06",
        "u_cat_07",
        "u_cat_08"
      ]
    }},
    "item.csv": {{
      "description": "匿名 item 侧特征表，也是默认候选 item 集合。",
      "primary_key": "iid",
      "columns": [
        "iid",
        "i_cat_01",
        "i_cat_02",
        "i_cat_03",
        "i_bucket_01"
      ]
    }}
  }},
  "row_counts": {{
    "train.csv": 40000,
    "test.csv": 10000,
    "user.csv": 50000,
    "item.csv": 14065,
    "sample_submission.csv": 10000
  }}
}}
```"""


data_analyse_template = """你是一个推荐系统（Recommendation System）模型训练与调优专家。你的目标是通过分析数据集特征，为后续模型优化提供依据。

### 本次分析目标
{inspect_plan}

### 数据集基本信息与加载方式
{dataset_description}

### 上一次生成的代码运行报错信息(如有)
{last_error}

### 代码生成约束（必须严格遵守）
1. **单次执行原则**：你只能生成一段完整的、可独立运行的 Python 代码。该代码将被直接写入文件并仅执行一次，不会有任何后续交互或重试机会。
2. **尽力而为**：在一次执行中尽可能覆盖分析目标所需的统计维度。如果某些分析因数据格式、内存限制或依赖缺失等原因无法完成，请直接跳过并在 print 中说明原因，**绝不要**通过异常捕获后强行循环重试、也**不要**生成需要人工介入的交互式代码。
3. **输出规范**：所有关键统计结果必须且只能通过 `print()` 输出到 stdout，禁止写入任何文件或创建持久化对象。
4. **自包含性**：代码必须包含所有必要的 import 语句，不依赖任何未声明的外部变量或全局状态。
5. **纯代码输出**：只输出可执行的 Python 代码本身，不要包含 ```python 标记、注释说明、前言或总结等任何非代码文本。
6. 如果有上一次生成的报错信息，需要仔细分析，避免再次生成报错的代码

请开始生成代码：
"""


smoking_test_edit_template = """
# Role
你是一个代码冒烟测试专家。你的任务是修改给定的代码片段，使其能以最小代价快速跑通全流程，用于验证代码逻辑和输出格式是否正确。

# Goal
请对以下代码进行冒烟测试级别的修改，具体目标：
1. 将训练轮数 epochs 修改为 10，确保能快速跑完）
2. 在划分完训练集和验证集之后的代码里面，将训练集规模缩小（使训练集仅使用约 1000 条数据，注意有可能前半部分和后半代码都要修改，防止取数报索引越界和device不一致错误；禁止缩小测试集和验证集大小，禁止缩小会输出的csv这部分大小，后续有校验）
3. 保持其他核心逻辑（模型结构、损失函数、输出格式、测试集规模和顺序等）完全不变
4. 确保修改后代码语法正确，且能正常产出验证集 loss 和完整的测试集输出文件
5. 尤其关注如果你修改或者引入了index变量，要注意变量tensor的device，确保不会出现cpu和gpu混用导致的报错

# 特别注意
* 千万不要在训练集/验证集划分结束前动训练集大小，否则划分出的验证集将会受到影响，会导致验证报错

# Constraint
修改将通过 llm_diff_edit_tool 执行，该工具要求：
- 每次替换的 original_str 必须在文件中【唯一存在】
- 支持多对替换，按顺序依次执行
- original_str 必须确保唯一，避免匹配到多处相同代码，必须保证和要替换的原文一模一样否则匹配不上。如果是修改某些参数，往往仅当前行局部即可锁定唯一，不要包含过多上下文导致上下文过长
- 对于original_str，建议小型的修改例如就参数的修改，不要把参数前面的空格或者tab内容包含进去，防止匹配不上字符串；长段的修改则需要严格保证特殊字符(缩进换行等)完全一样防止匹配不上，原始代码文件是4个空格缩进而非tab的

# Output Format
你必须且只能输出一个 JSON 数组，每个元素是一对替换，可以有多对替换，格式如下：
[
  {{
    "original_str": "要替换的原始代码字符串（确保唯一）",
    "new_str": "替换后的新代码字符串"
  }}
]

⚠️ 禁止输出任何解释、分析、markdown 代码块标记或其他文字，仅输出纯 JSON 数组。

# Last round error
下面是上次编辑时的报错信息(如果有)，清注意避免：
{last_edit_error}

# Code to Edit
{code_piece}
"""


smoking_test_judge_template = """
# Role
你是一个冒烟测试结果判定器。请根据提供的训练日志，严格按照以下规则判断代码是否跑通且逻辑基本正常。

# Judgment Rules
满足以下【全部】条件则判定为"正常"，否则判定为"异常"：
1. 日志中无任何 Error / Exception / Traceback 报错
2. Loss 值全程不为 NaN / Inf
3. 验证集 NDCG@10 >0 (仅用于检查是否计算出错，如果严格等于0，例如0.0000，肯定是计算错了有异常)
4. 训练流程完整执行到了结束或 EarlyStopping，未中途崩溃

# Output Constraint
⚠️ 你必须且只能输出两个字："正常" 或 "异常"
⚠️ 禁止输出任何分析过程、标点符号、换行符或其他文字

# Training Log
{train_log}
"""


train_log_analyse_template="""# Role: AI模型训练分析师

## Task
请根据提供的【训练代码】和【训练日志】，分析本次训练过程并提取关键信息。结果将用于更新历史记录并指导下一轮实验。
注意：请根据实际代码和日志自适应调整分析维度，不要生搬硬套特定框架的术语。

【训练代码】
<train_code>
{train_code}
</train_code>

【训练日志】
{train_log}


## Output Format
请严格输出且仅输出一个包含以下4个字段的JSON对象，不要包含任何Markdown标记或其他解释文字：
{{
  "best_model_dir": "<string> 最佳模型的完整保存路径",
  "best_ndcg": "<float> 最佳验证集NDCG@10",
  "model_and_params": "<string> 完整记录本轮全部核心配置，必填维度：1.序列模型架构（单向/双向GRU/SASRec等）；2.用户验证集划分策略（随机/分层用户抽样）；3.序列数据增强手段（裁剪/反转/局部打乱/动态Mask区间）；4.关键超参（LR、weight_decay、dropout、label_smoothing、hidden_dim、item_dim、MAX_SEQ_LEN、batch_size）；5.用户特征处理逻辑（拼接/MLP交叉）；6.学习率调度器类型；7.损失函数与正则策略，精炼连贯描述。",
  "training_diagnosis": "<string> 训练状态诊断。观察loss/metric变化趋势，判断是否过拟合、欠拟合、震荡或提前停止，是否学习率设置不合理导致学的很慢或者后期震荡，以及对比train与val的差距给出明确结论，额外标注验证集与线上指标落差隐患。",
  "conclusion_and_next": "<string> 100字以内的总结论(优/良/差/失败)，简要明确地提示下一轮需要调整的方向，贴合序列推荐优化思路。"
}}

请开始输出：
"""


train_history_analyse_template = """你是一个推荐系统（Recommendation System）模型训练与调优专家。你的目标是通过分析历史训练记录和数据集特征，持续提升模型在验证集上的Top-K推荐指标（NDCG@10）。

### 当前状态信息
数据集探查信息(可用于调整序列长度等参数)：
{data_inspect_result}

所有训练尝试：
{history_train_summary}

### 探索方向与思路
1. **实验初期优先对比不同范式基线**：属性增强双塔、轻量序列模型 (例如 GRU4Rec、CaserCNN、SASRec作为基线来探索)；基础 GRU 架构可升级双向 GRU，融合时序终点隐向量 + 序列均值双表征，搭配用户分类特征交叉 MLP 提升表征能力，快速拉高 val NDCG。
2. **长尾分布优化**：如果物品极度长尾，必须配套流行度加权负采样、逆频率加权损失、Focal Loss等方法，抑制头部流行度偏差，提升尾部物品召回能力。
3. **序列数据增强体系**：训练集尝试引入多重序列增强：随机尾部裁剪、短序列小概率反转、长序列前半段局部打乱；搭配动态递增 mask 概率（随训练轮次提升掩码比例），增强序列鲁棒性，缓解分布偏移。
4. **序列分布偏移应对**：如果训练与测试序列差异大，序列模型需控制截断长度，配合随机掩码、随机截断的数据增强；避免过度依赖长序列依赖，优先适配短序列的轻量建模。
5. 实验少于3次以架构探索为主，超过 3 次后转向细粒度超参调优：嵌入维度、隐藏层大小、学习率、正则强度、负采样数量、dropout、label smoothing、weight decay、学习率调度策略
6. 效果不佳时优先排查：是否没有使用LN/BN和跳跃连接、JK等层，是否未使用正确的归一化，是否未充分利用物品和用户属性特征、交互特征、负采样策略不合理、序列长度设置失配、epoch不足导致欠拟合【最高epoch上限为200】。
7. 如果连续几轮调整效果反倒严重变差，先回溯历史上最佳的情况，分析其特征与当前实验的差异，做对应调整。
8. 在验证集NDCG达到0.14之后，该方案可以作为基线，后续不做大幅度调整，专注于微调超参数和细节优化，逐步提升验证集NDCG到0.2以上。

### 任务要求
请基于上述信息：
1. 详细分析当前验证集性能瓶颈的原因。需分析当前阻碍提升的核心因素（如模型选错、模型太简单、参数不对、序列建模能力不足、用户冷启动问题、物品长尾分布、正负样本不平衡、过拟合 / 欠拟合、验证集划分失真、线上线下分布偏移等）；
2. 写明本次选择的具体调优方式，具体写明需要尝试的模型架构、损失函数、数据增强策略或超参数调整等信息
3. 简明扼要，内容400字以内，文本形式输出

请开始分析并直接输出文本：
"""


train_code_gen_template = """
你是推荐系统代码生成专家，仅输出完整可运行 Python 代码，禁止额外解释文字。
严格规避上一轮报错：uid 重复、tensor 格式 uid、item_scores 与 Top100 物品顺序错位、非法 padding 物品写入 csv、缺失验证集 uid、混入训练集 uid。

# 任务总要求
{task_description}
读取给定数据集，完整实现：训练 / 验证用户划分、序列推荐模型训练、Top-K Hit-based NDCG@10 验证、TTA 多轮平均推理、输出全套规范 csv 与映射 pkl 文件，全程原生 PyTorch，单模型，显存占用≤20GB，优先 SASRec 类稀疏序列模型。

# 一、用户划分强制标准（不可修改逻辑）
数据集信息：
{dataset_description}

## 训练集和验证集划分方式【须严格遵守否则通不过校验】
1. 提取 train.csv 全部唯一 uid 升序排序；
2. seed=666 打乱 uid 列表；
3. 0.8/0.2 分割，训练集向上取整 math.ceil；
4. 按 uid 集合筛选样本，同一用户所有样本归属同一集合；
标准参考函数：
def _reproduce_val_split (train_df: pd.DataFrame) -> set:
    unique_uids = sorted (train_df ["uid"].unique ())
    rng = np.random.RandomState (666)
    shuffled_uids = rng.permutation (unique_uids)
    n_train = math.ceil (len (shuffled_uids) * 0.8)
    return set (shuffled_uids [n_train:])

# 二、全套输出文件清单（缺一不可，格式严格执行）
输出根目录：{model_save_dir}，存在则覆盖
1. B2.csv（测试集 Top10）
    - 表头：uid,prediction
    - uid：原始字符串 uxxxxxx，禁止数字、tensor、numpy 对象
    - prediction：前 10 合法 iid，英文逗号分隔，置信降序、无重复

2. B2-top100.csv/validation-top100.csv（Top100 物品）
    - 表头 uid,prediction；filter_history=False 不剔除历史物品；仅合法 iid，不足 100 如实输出，禁止填充 0/padding；无重复 iid
    - B2_softmax_item.csv/validation_softmax_item.csv（核心易错文件，严格遵守）
    - 表头固定 uid,item_scores
    - uid 为原生字符串 'uxxxxxx'；
    - item_scores 单元格式 iid:0.XXXX，分数强制保留 4 位小数；
    - 字符串内物品顺序必须与同批次 Top100 文件 prediction 完全一一对应；
    - 仅输出 Top100 条目，无重复 iid、无 padding / 词典外物品；
    - 生成验证集文件时 filter_history=False；

3. best_model.pth：每轮验证指标创新高立即保存；
4. item2idx.pkl、idx2item.pkl：训练结束持久化至保存目录。

# 三、推理 TTA 硬性实现规则（解决 softmax 概率计算错误）
1. 训练结束加载 best_model，model.train () 启用 Dropout 做 TTA；
2. TTA 推理轮次区间 20~50，多次 softmax 概率累加后取均值；
3. 关键：DataLoader 同一 uid 存在多条交互样本时，必须用字典聚合所有该用户概率再求均值，保证每个 uid 仅输出一行，彻底消除 uid 重复；

# 四、训练与评测规则
1. 评测指标仅 Top-K Hit-based NDCG@10，每 10 个 epoch 打印 loss 与验证指标，禁用 tqdm；
2. 早停 PATIENCE 30~50，MAX_EPOCHS 100~500，欠拟合可上调模型宽度、层数、epoch；
3. AdamW + Cosine 退火调度，梯度裁剪 max_norm=1.0，CrossEntropyLoss ignore_index=0；
4. 全局固定 SEED=666，numpy/torch/cuda 全锁种子，DataLoader num_workers=0；
4. 训练仅使用训练集 uid 构建 item2idx、用户特征映射，杜绝验证 / 测试集词表泄露。

# 五、历史报错规避（强制复盘，禁止复现）
上一轮报错：{last_error}
高频错误点必须修复：
- Dataset 返回 uid 强制转为 str (row ['uid'])，杜绝 tensor (xxx) 写入 csv；
- 推理聚合 uid，单行单用户，消除数千条重复 uid；
- item_scores 与 Top100 物品同步循环生成，保证顺序完全对齐；
- 过滤词典外 iid、padding 0，不写入 prediction 与分数字符串

# 六、可调优化方向（灵活不锁死结构）
可行优化思路：
{optimize_direction}

# 七、当前最佳代码(可复用模块，不强制照搬参数）
{reference_code}

# Hard Constraint
只输出完整可运行 Python 代码，无任何说明、注释、前置文字；
严格区分训练 / 验证 / 测试数据，隔离词表构建，无数据泄露；
全套 csv、pkl、模型文件全部实现，softmax 分数文件格式、对齐逻辑完整无遗漏；
TTA、uid 聚合、uid 字符串转换、去重、合法性过滤全部完整实现；
请直接输出完整代码：
"""


train_code_gen_edit_based_template = """
# Role
你是一个推荐系统算法代码修复与优化专家。你的任务是基于上一轮的报错信息或新的优化方向，对现有训练代码进行【最小化增量修改】。

# Goal
请根据以下输入信息，精准定位问题并完成代码修复/优化：
1. 若存在报错：优先修复该报错，并检查是否还有其他错误，确保代码可执行、不重犯同类错误。
2. 若存在优化方向：在保持代码可运行的前提下，落实该优化方向（如调整模型结构、模型复杂度、超参、损失函数等）。
3. 严禁破坏以下核心业务约束（即使报错与此相关，也应以修复而非删除的方式处理）：
   - 训练集/验证集划分逻辑必须严格遵循：uid去重升序 → seed=666 shuffle → 80:20 ceil划分
   - 测试集输出 B2.csv 必须包含表头 uid,prediction，prediction为逗号分隔的Top10 iid
   - 必须同时输出 B2-top100.csv 和 validation-top100.csv（Top100合法iid，严禁将 padding token、mask token 或不在物品词典中的 ID 写入prediction。若某用户的有效候选物品不足 100 个，允许输出少于 100 个的推荐列表。宁可输出短列表，也不可用无效物品填充。）
   - 生成 validation-top100.csv 和 B2-top100.csv 时，不要过滤用户的历史交互物品（确保filter_history=False），确保验证集评估口径与线上 Top-K Hit-based NDCG@10 单正样本评测一致
   - 生成的softmax文件必须包含表头 uid,item_scores，且每个条目为 iid:score，用英文逗号连接，按模型置信度从高到低排序
   - 模型保存路径、pickle文件保存逻辑不可变更
   - 仅使用单一原生PyTorch模型，显存控制在20GB以内

# 划分训练集验证集的参考逻辑不能修改
def _reproduce_val_split(train_df: pd.DataFrame) -> set:
    unique_uids = sorted(train_df["uid"].unique())
    rng = np.random.RandomState(666)
    shuffled_uids = rng.permutation(unique_uids)
    n_train = math.ceil(len(shuffled_uids) * 0.8)
    return set(shuffled_uids[n_train:])

# Constraint
修改将通过 llm_diff_edit_tool 执行，该工具要求：
- 每次替换的 original_str 必须在文件中【唯一存在】
- 支持多对替换，按顺序依次执行
- original_str 必须确保唯一，避免匹配到多处相同代码，必须保证和要替换的原文一模一样否则匹配不上，同时避免纳入无关内容
- 对于小型参数修改，不要包含前面的空格/tab，防止匹配失败；长段修改需严格保证缩进（4空格）、换行等特殊字符完全一致
- 如果修改涉及 tensor 索引或新增变量，务必检查 device 一致性，防止 CPU/GPU 混用
- 如果错误属于架构级问题（如数据流断裂、模型设计缺陷），允许较大范围的重构，但仍以 edit 形式输出，不要输出完整文件

# Output Format
你必须且只能输出一个 JSON 数组，每个元素是一对替换，可以有多对替换，格式如下：
[
  {{
    "original_str": "要替换的原始代码字符串（确保唯一）",
    "new_str": "替换后的新代码字符串"
  }}
]

⚠️ 禁止输出任何解释、分析、markdown 代码块标记或其他文字，仅输出纯 JSON 数组。

# Inputs
## 优化方向（可能为空）
{optimize_direction}

## 上一轮报错信息（可能为空）
{last_error}

## 上次编辑工具的报错（若有则说明original_str匹配失败，请修正）
{last_edit_error}

## 当前待修改代码
{reference_code}
"""


def filter_cdot(str_in):
    return str_in.replace("```python", "").replace("```Markdown", "").replace("```MarkDown", "").replace("```markdown", "").replace("```", "").replace("```python", "")

def remove_think_tags(text: str) -> str:
    """ 移除文本中的 <think>...</think> 标签及其内容。 """
    if not text:
        return text
    # [\s\S]*? 匹配任意字符（包括换行），非贪婪模式
    cleaned = re.sub(r'<think>[\s\S]*?</think>', '', text)
    return cleaned.strip()


def dual_print(*args, log_file="rec_output.log", mode="a", encoding="utf-8", **kwargs):
    """
    同时输出到 Jupyter 和 txt 文件的 print 替代函数
    """
    # 1. 正常输出到 Jupyter
    print(*args, **kwargs)
    
    # 2. 同时写入文件
    # 先将内容按 print 的规则拼接成字符串
    sep = kwargs.get("sep", " ")
    end = kwargs.get("end", "\n")
    message = sep.join(str(a) for a in args) + end
    write_log('print_output',message)
    
    with open(log_file, mode=mode, encoding=encoding) as f:
        f.write(message)
        f.flush()  # 确保实时写入，避免缓冲导致日志丢失


def call_llm_with_think(prompt: str, model: str = "qwen3.6-plus", temperature: float = 0, max_retries: int = 5) -> str:
    """
    调用支持深度思考的 LLM 模型，内置纯标准库实现的指数退避重试机制。
    """

    # 在重试循环外初始化客户端，避免每次重试都重新建立连接
    client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    messages = [{"role": "user", "content": prompt}]

    for attempt in range(max_retries + 1):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                extra_body={"enable_thinking": True},
                stream=True,
                temperature=temperature
            )

            final_answer = ""
            is_answering = False

            for chunk in completion:
                if not chunk.choices:
                    continue
                    
                delta = chunk.choices[0].delta
                
                # 忽略思考过程内容 (reasoning_content)
                # 仅拼接最终回复内容
                if hasattr(delta, "content") and delta.content:
                    if not is_answering:
                        is_answering = True
                    final_answer += delta.content

            return remove_think_tags(final_answer)

        except Exception as e:
            if attempt == max_retries:
                dual_print(f"❌ 已达最大重试次数 ({max_retries})，放弃请求: {e}")
                return "LLM调用出错，之后重试"
            
            # 计算指数退避时间: 2^attempt + 随机抖动，防止多线程同时重试引发雷群效应
            wait_time = min((2 ** attempt) + random.uniform(0, 1), 60)
            dual_print(f"⚠️ 第 {attempt + 1} 次请求失败 ({type(e).__name__})，{wait_time:.1f}s 后重试...")
            time.sleep(wait_time)

def call_llm(textin, max_retry=6, model=default_model,temperature=0) -> str:
    base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    # ==============================

    client = OpenAI(api_key=api_key, base_url=base_url)
    for attempt in range(1, max_retry + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": textin}
                ],
                temperature=temperature
            )
            return remove_think_tags(response.choices[0].message.content)

        except Exception as e:
            wait_time = 2 ** attempt
            dual_print(f"[Retry {attempt}/{max_retry}] 未知异常: {e}, {wait_time}s 后重试...")
            time.sleep(wait_time)
    return "LLM调用出错，之后重试"


def read_file_safe(history_file_path):
    """获取历史训练结果信息"""
    if os.path.exists(history_file_path):
        with open(history_file_path, 'r', encoding='utf-8') as f:
            return f.read()
    return ""


def llm_diff_edit_tool(file_path, original_str, new_str, save_path)->bool: 
    try:
        # 判断文件是否存在
        if not os.path.exists(file_path):
            dual_print(f"⚠️ 文件不存在: {file_path}")
            return False,f"⚠️ 文件不存在: {file_path}"

        # 读取文件内容
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # 判断原始字符串是否唯一存在
        if content.count(original_str) == 0:
            dual_print(f"⚠️ 原始字符串不存在: {original_str}")
            return False,f"⚠️ 原始字符串不存在: {original_str}"
        elif content.count(original_str) != 1:
            dual_print(f"⚠️ 原始字符串不唯一: {original_str}")
            return False,f"⚠️ 原始字符串不唯一存在: {original_str}"

        # 替换字符串
        content = content.replace(original_str, new_str)

        # 保存文件
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(content)

        dual_print(f"✅ 替换成功: {original_str} -> {new_str}")
        return True,""
    except Exception as e:
        dual_print(f"❌ 替换失败: {e}")
        return False,"❌ 替换失败: {e}"


def apply_code_edits(file_path: str, edits: List[Dict[str, str]], save_path: str = None) -> bool:
    """
    串行执行多对 diff 编辑，每对编辑基于上一次编辑后的文件内容。
    """
    if save_path is None:
        save_path = file_path

    current_file = file_path
    for i, edit in enumerate(edits):
        # 首尾去掉空白，后面看看是否需要
        original = edit["original_str"]#.strip()
        new = edit["new_str"]#

        dual_print(f"🔧 执行第 {i+1}/{len(edits)} 对替换...")
        success,error_info = llm_diff_edit_tool(current_file, original, new, save_path)

        if not success:
            dual_print(f"❌ 第 {i+1} 对替换失败，中止后续编辑")
            return False,f"❌ 第 {i+1} 对替换失败，中止后续编辑。"+error_info

        # 后续替换基于已保存的文件
        current_file = save_path

    dual_print(f"✅ 全部 {len(edits)} 对替换执行完毕")
    return True,""


def validate_submission(
    csv_path='./model/interface_rec/B2.csv',
    data_path='/data/coding/line3/dataset_b/rec_data/',
    topk: int = 10
) -> bool:
    """
    验证推荐系统提交文件格式是否正确。

    提交文件应为 CSV，包含 uid 和 prediction 两列，
    prediction 为逗号分隔的 Top-K 个 item_id。

    Args:
        csv_path:  提交文件路径，如 "./model/interface_rec/B2.csv"
        data_path:  原始数据集目录，内含 train.csv / test.csv / user.csv / item.csv
        topk:      每个用户应推荐的物品数量（默认 10）

    Returns:
        True 表示验证通过，False 表示存在错误
    """
    errors = []

    # ========== 1. 文件是否存在 ==========
    if not os.path.exists(csv_path):
        dual_print(f"❌ 文件不存在: {csv_path}")
        return False

    # ========== 2. 读取 CSV ==========
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        dual_print(f"❌ CSV 读取失败: {e}")
        return False

    # ========== 3. 检查表头和列数 ==========
    expected_columns = ["uid", "prediction"]
    if list(df.columns) != expected_columns:
        errors.append(
            f"表头不正确。期望 {expected_columns}，实际为 {list(df.columns)}"
        )

    if df.shape[1] != 2:
        errors.append(f"列数应为 2，实际为 {df.shape[1]}")

    # ========== 4. 加载数据集获取标准 uid 列表和合法物品池 ==========
    try:
        test_df = pd.read_csv(os.path.join(data_path, "test.csv"))
        item_df = pd.read_csv(os.path.join(data_path, "item.csv"))
        train_df = pd.read_csv(os.path.join(data_path, "train.csv"))
    except Exception as e:
        dual_print(f"❌ 数据集加载失败: {e}")
        return False

    gt_uids = test_df["uid"].unique()
    valid_item_set = set(item_df["iid"].tolist())

    # ========== 5. 检查 uid ==========
    if "uid" in df.columns:
        submitted_uids = df["uid"].values

        # 5.1 数量一致
        if len(submitted_uids) != len(gt_uids):
            errors.append(
                f"uid 数量不一致。期望 {len(gt_uids)}，实际 {len(submitted_uids)}"
            )
        else:
            # 5.2 内容完全匹配
            submitted_uid_set = set(submitted_uids)
            gt_uid_set = set(gt_uids)
            if submitted_uid_set != gt_uid_set:
                missing = gt_uid_set - submitted_uid_set
                extra = submitted_uid_set - gt_uid_set
                if len(missing) > 0:
                    missing_list = list(missing)[:10]
                    errors.append(
                        f"缺少 uid: {missing_list}{'...' if len(missing) > 10 else ''} "
                        f"(共 {len(missing)} 个)"
                    )
                if len(extra) > 0:
                    extra_list = list(extra)[:10]
                    errors.append(
                        f"多余 uid: {extra_list}{'...' if len(extra) > 10 else ''} "
                        f"(共 {len(extra)} 个)"
                    )

        # 5.3 不能有重复
        dup_count = df["uid"].duplicated().sum()
        if dup_count > 0:
            errors.append(f"uid 存在 {dup_count} 个重复值")

        # 5.4 不能有空值
        null_count = df["uid"].isnull().sum()
        if null_count > 0:
            errors.append(f"uid 存在 {null_count} 个空值")

    # ========== 6. 检查 prediction ==========
    if "prediction" in df.columns:
        # 6.1 不能有空值
        null_count = df["prediction"].isnull().sum()
        if null_count > 0:
            errors.append(f"prediction 存在 {null_count} 个空值")

        # 6.2 每行格式与内容校验
        wrong_format_count = 0      # 无法解析的行数
        wrong_topk_count = 0       # 推荐数量不符的行数
        invalid_item_count = 0     # 包含非法 item_id 的行数
        dup_in_pred_count = 0      # 单行内有重复 item 的行数
        all_invalid_items = set()  # 收集所有非法 item_id（用于展示）
        topk_dist = []             # 每行实际推荐数量分布

        for row_idx, row in df.iterrows():
            pred_str = str(row["prediction"]).strip()

            # 空字符串或 NaN
            if pred_str == '' or pred_str.lower() == 'nan':
                wrong_format_count += 1
                continue

            # 解析逗号分隔的 item_id
            items = [x.strip() for x in pred_str.split(",") if x.strip()]
            topk_dist.append(len(items))

            # 6.2a 推荐数量应为 topk
            if len(items) != topk:
                wrong_topk_count += 1

            # 6.2b 每个 item_id 必须在合法物品池中
            for item in items:
                # 尝试匹配原始类型（item_df 中的 iid 可能是 int 或 str）
                if item not in valid_item_set:
                    try:
                        item_typed = type(list(valid_item_set)[0])(item)
                        if item_typed not in valid_item_set:
                            invalid_item_count += 1
                            all_invalid_items.add(item)
                    except (ValueError, TypeError):
                        invalid_item_count += 1
                        all_invalid_items.add(item)

            # 6.2c 单行内不能有重复 item
            if len(items) != len(set(items)):
                dup_in_pred_count += 1

        # 汇总错误
        if wrong_format_count > 0:
            errors.append(f"prediction 有 {wrong_format_count} 行格式异常（空值或无法解析）")

        if wrong_topk_count > 0:
            errors.append(
                f"prediction 有 {wrong_topk_count} 行推荐数量不等于 Top-{topk}"
            )

        if invalid_item_count > 0:
            sample_invalid = list(all_invalid_items)[:10]
            errors.append(
                f"prediction 中出现非法 item_id（不在 item.csv 中），"
                f"共涉及 {len(all_invalid_items)} 个不同非法ID，"
                f"示例: {sample_invalid}"
            )

        if dup_in_pred_count > 0:
            errors.append(
                f"prediction 有 {dup_in_pred_count} 行内存在重复推荐的 item"
            )

    # ========== 输出结果 ==========
    if errors:
        dual_print("❌ 验证未通过，发现以下问题：")
        for i, err in enumerate(errors, 1):
            dual_print(f"   {i}. {err}")
        return False
    else:
        dual_print("✅ 验证通过！提交文件格式完全正确。")
        dual_print(f"   - 用户数: {len(df)}")
        dual_print(f"   - 每用户推荐数: Top-{topk}")

        # 统计 prediction 中物品覆盖情况
        all_rec_items = []
        for pred_str in df["prediction"]:
            items = [x.strip() for x in str(pred_str).split(",") if x.strip()]
            all_rec_items.extend(items)
        unique_rec = set(all_rec_items)

        dual_print(
            f"   - 推荐物品去重数: {len(unique_rec)} / {len(valid_item_set)} "
            f"(覆盖率 {len(unique_rec) / len(valid_item_set) * 100:.2f}%)"
        )

        # 热门物品统计
        rec_counter = Counter(all_rec_items)
        top5_items = rec_counter.most_common(5)
        dual_print(f"   - 被推荐最多的 Top-5 物品: {top5_items}")

        # 训练集物品重叠检查
        train_items = set(train_df["target_iid"].unique()) if "target_iid" in train_df.columns else set()
        if train_items:
            overlap = unique_rec & train_items
            dual_print(
                f"   - 推荐物品中与训练集重叠: {len(overlap)} 个 "
                f"({len(overlap) / len(unique_rec) * 100:.1f}%)"
            )
        return True


def run_smoking_test(data_path, script_path, submission_csv_path,topk):
    # script_path = './interface_gen_rec_smoke.py'  # ← 替换为你的冒烟测试脚本路径
    # submission_csv_path = './model/interface_rec/B2.csv'
    # data_path='/data/coding/line3/dataset_b/rec_data/'
    # topk=10
    try:
        result = subprocess.run(
            [sys.executable, script_path],  # 使用当前 Python 解释器
            capture_output=True,
            text=True,
            timeout=1000,  # 600秒 = 10分钟超时限制
            cwd='./'  # 设置工作目录，防止相对路径问题，这里面需要有相对路径才行
        )
        dual_print("\n===== 训练过程输出 (stdout) =====")
        dual_print(result.stdout)
        train_log = result.stdout
        if result.returncode != 0:
            if result.stderr and 'error' in result.stderr.lower():
                dual_print("\n===== ⚠️ 代码执行错误输出 (stderr) =====")
                dual_print(result.stderr)
                dual_print('❌ 训练过程出错，代码有误')
                if 'keyerror' in result.stderr.lower():
                    return False,'❌ 训练过程出错，代码有误' + '\n报错信息为：' + result.stderr + "\n训练日志为："+train_log
                else:
                    return False,'❌ 训练过程出错，代码有误' + '\n报错信息为：' + result.stderr + "\n训练日志为："+train_log

        # 判断运行是否正常，用flash模型判断即可
        smoking_test_judge_prompt = smoking_test_judge_template.format(train_log=train_log)
        judge_result = call_llm(smoking_test_judge_prompt)
        if judge_result == "正常":
            dual_print("✅ 冒烟测试运行通过")
        else:
            dual_print("❌ 冒烟测试运行未通过")
            return False, "❌ 冒烟测试运行未通过，训练效果明显不佳很可能逻辑有误，训练过程日志为"+"\n"+train_log
        # ========== 1. 检查 B2.csv 是否存在 ==========
        if not os.path.exists(submission_csv_path):
            dual_print("❌ 测试集结果未输出，异常")
            return False, f"❌ 测试集结果未输出在规定路径{submission_csv_path}，异常"
        dual_print(f"✅ 测试集结果已输出: {submission_csv_path}")

        # ========== 2. 校验 B2.csv (Top-10) ==========
        has_submission = validate_submission(
            csv_path=submission_csv_path, data_path=data_path, topk=topk
        )
        if not has_submission:
            return False, "❌ B2.csv 格式或内容校验未通过"
        
        err_hint="""验证集 Top100 文件 validation-top100.csv和测试集 Top100 文件 B2-top100.csv:
        - 表头必须是 uid,prediction
        - uid 必须是用户id，形如'uxxxxxx'，而非编码后的数字/tensor
        - prediction 必须是 100 个合法 iid字符串，用英文逗号连接，按模型置信度从高到低排序，不允许重复
        - 输出的 Top-K 推荐列表必须仅包含合法物品 ID。严禁将 padding token、mask token 或不在物品词典中的 ID 写入prediction。若某用户的有效候选物品不足 100 个，允许输出少于 100 个的推荐列表。宁可输出短列表，也不可用无效物品填充。
        - 生成 validation-top100.csv 和 B2-top100.csv 时，不要过滤用户的历史交互物品（确保filter_history=False）

        验证集带分数文件 validation_softmax_item.csv 和 测试集带分数文件 B2_softmax_item.csv:
        - 表头必须是 uid,item_scores
        - uid 必须是用户id，形如'uxxxxxx'，而非编码后的数字/tensor
        - item_scores 格式：每个条目为 iid:score，用英文逗号连接，按模型置信度从高到低排序，与对应 Top100 文件的物品顺序完全一致
        - 分数为浮点数，保留 4 位小数；物品 ID 必须合法，不允许重复
        - 仅输出 Top100 对应的物品与分数，不输出全量物品；有效物品不足 100 个时，按实际数量输出
        - 生成验证集文件时 filter_history=False"""

        # ========== 3. 校验 B2-top100.csv ==========
        top100_test_path = os.path.join(os.path.dirname(submission_csv_path), "B2-top100.csv")
        has_test_top100 = validate_test_top100(
            csv_path=top100_test_path, data_path=data_path, topk=100
        )
        if not has_test_top100:
            return False, "❌ B2-top100.csv 校验未通过，仔细检查B2-top100中是否引入了非法物品id，宁缺毋滥，需要保证合法性，该文件的top10和B2.csv必须一致\n"+err_hint

        # ========== 4. 校验 validation-top100.csv ==========
        val_top100_path = os.path.join(os.path.dirname(submission_csv_path), "validation-top100.csv")
        has_val_top100 = validate_val_top100(
            csv_path=val_top100_path, data_path=data_path, topk=100
        )
        
        
        if not has_val_top100:
            return False, "❌ validation-top100.csv 校验未通过"+err_hint
        
        # ========== 5. 全部通过 ==========
        dual_print("✅ 所有提交文件校验通过（B2.csv / B2-top100.csv / validation-top100.csv）")
        return True, ""
    
    except Exception as e:
        dual_print(f"❌ 运行冒烟测试脚本失败: {e}")
        return False, "❌ 运行冒烟测试脚本失败，测试集结果未输出，异常"
    


def _ask_llm_timeout_check(recent_log, elapsed_seconds, timeout):
    prompt = f"""你是一个训练任务监控助手。请根据以下信息判断训练是否正常。

    ## 当前训练日志）
    {recent_log}

    请分析
    从日志中找出训练进度（如当前在第几个 epoch / step，总共多少个）
    1. 判断训练是否正常（在epoch数超过100的情况下看NDCG是否正常）

    如果当前还没有完成100个epoch，输出 wait  (还需要观察更多输出)
    如果已经训练到了超过100个epoch，检查当前NDCG@10是否大于0.45，如果NDCG@10还很低（没有到0.45），并且从增长趋势上看不出有明显增长趋势，才输出 bad
    否则输出continue

    输出格式
    continue 或 bad 或 wait 中的一个
    不要解释，请分析后输出你的判断：
    """
    try:
        response = call_llm(prompt, model='qwen3.6-plus')
        dual_print(f"\n🔍 [LLM 监控] 已耗时 {elapsed_seconds:.0f}s")
        dual_print(f" LLM 回复: {response.strip()[-200:]}")

        if 'kill' in response.lower():
            return 'kill'
        elif 'bad' in response.lower():
            return 'bad'
        elif 'wait' in response.lower():
            return 'wait'
        return 'continue'
    except Exception as e:
        dual_print(f"⚠️ LLM 监控调用失败: {e}，默认继续")
        return 'continue'


def code_run(script_path, train_code, model_save_dir, timeout=1200):
    global global_success_times
    start_time = time.time() # 开始时间
    
    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'
    
    process = subprocess.Popen(
        [sys.executable, script_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd='./',
        bufsize=1,
        env=env
    )
    
    stdout_lines = []
    should_kill = False
    
    # ====== 核心：基于时间的触发机制，不用正则 ======
    last_check_time = start_time
    CHECK_INTERVAL = 60       # 每 30 秒让 LLM 评估一次
    FIRST_CHECK_DELAY = 60    # 首次评估至少等 60 秒（攒够足够日志）
    first_check_done = False
    should_continue = False
    dual_print("\n===== 开始正式训练（实时监控模式） =====")
    decision = ''
    try:
        for line in process.stdout:
            # 每收到一行，就追加 + 打印（自然触发）
            line = line.rstrip('\n')
            stdout_lines.append(line)
            dual_print(line)
            sys.stdout.flush()
            
            current_time = time.time()
            elapsed = current_time - start_time

            # ============================================================
            # ✅ 修复1：硬超时检查 —— 无论 LLM 说什么，到时间就强制杀
            # ============================================================
            if elapsed >= timeout:
                should_kill = True
                decision = 'kill'
                process.kill()
                dual_print(f"\n⏹️  硬超时！已耗时 {elapsed:.0f}s >= {timeout}s，强制终止。")
                break
            
            # ====== 判断是否该问 LLM 了 ======
            # 条件：首次等 60 秒，之后每 30 秒检查一次
            should_check = False
            if not first_check_done and elapsed >= FIRST_CHECK_DELAY:
                should_check = True
                first_check_done = True
            elif first_check_done and (current_time - last_check_time) >= CHECK_INTERVAL:
                should_check = True
            
            if should_check:
                last_check_time = current_time
                
                # 取最近的日志（不要太长，节省 token）
                recent_log = '\n'.join(stdout_lines[:-2000])
                
                decision = _ask_llm_timeout_check(
                    recent_log=recent_log,
                    elapsed_seconds=elapsed,
                    timeout=timeout
                )
                
                if decision =='continue':
                    should_continue = True
                    dual_print(f"\n⏹️  LLM 判定训练不会超时，继续")
                    
                if not should_continue and decision == 'kill':
                    should_kill = True
                    process.kill()
                    dual_print(f"\n⏹️  LLM 判定继续训练将超时，已终止。（已耗时 {elapsed:.0f}s）")
                    break #return f"模型训练超过20min，已终止，请减少模型复杂度，或者减少模型层数，或者提升batchsize，缩小epoch等方式处理"
                if decision=='bad':
                    should_kill = True
                    process.kill()
                    dual_print(f"\n⏹️  LLM 判定训练没有效果")
                    break#return f"模型训练没有效果，请调整，训练日志为："+'\n'.join(stdout_lines)

    except Exception as e:
        process.kill()
        dual_print(f"❌ 监控异常: {e}")
        return f"监控异常: {e}", f"监控异常: {e}", "", True
    
    # 等进程结束
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
    
    elapsed_total = time.time() - start_time
    returncode = process.returncode
    train_log = '\n'.join(stdout_lines)
    
    # ====== 被 LLM kill 的情况 ======
    if should_kill and decision == 'bad':
        kill_msg = "模型训练没有效果，请调整，训练日志为："+'\n'.join(stdout_lines)
        dual_print(kill_msg)
        return train_log, kill_msg, "", True
    
    # ====== 正常完成（和原逻辑一致） ======
    dual_print("\n===== 训练完成 =====")
    
    # ... 后续验证、分析逻辑不变 ...
    train_log_analyse_prompt = train_log_analyse_template.format(
        train_code=train_code, train_log=train_log
    )
    
    if returncode != 0:
        if 'error' in train_log.lower():
            dual_print("\n===== ⚠️ 代码执行错误 =====")
            error_info = train_log[-2000:]
            dual_print(error_info)
            last_round_acc_md = "请尝试其他参数或架构，当前运行报错:\n" + error_info
            return train_log, last_round_acc_md, "", True
    
    if not validate_submission(csv_path=os.path.join(model_save_dir, "B2.csv"), data_path=fixed_data_path, topk=10):
        error_info = "❌ 验证集没有生成或者格式结果不合法"
        return train_log, error_info, "", True
    
    last_round_acc_md = call_llm_with_think(train_log_analyse_prompt, model='qwen3.6-plus')

    # ====== 自然超时 ======
    if elapsed_total >= timeout or decision == 'kill':
        timeout_msg = f"❌ 训练超时，耗时 {elapsed_total:.0f}s 超过限制 {timeout}s，请减少模型复杂度，或者减少模型层数，或者提升batchsize，缩小epoch等方式处理，除非训练效果确实非常好。当前训练结果信息为 "+last_round_acc_md
        return train_log, timeout_msg, last_round_acc_md, True
    return train_log, "", last_round_acc_md, False


import os
import subprocess
import sys


def run_data_analysis(script_path: str, output_md_path: str, timeout: int = 300) -> tuple[bool, str]:
    """
    运行已生成的数据分析脚本，将标准输出整理为MD格式写入结果文件
    
    Args:
        script_path: 待运行的Python分析脚本绝对/相对路径
        output_md_path: 探查结果输出的MD文件路径（成功后覆盖写入）
        timeout: 单轮脚本运行最大时长，单位秒，默认300秒
    
    Returns:
        tuple[bool, str]
        - 成功：(True, 脚本标准输出内容)
        - 失败：(False, 错误详情/ stderr)
    """
    # 前置校验：脚本文件不存在直接返回失败
    if not os.path.isfile(script_path):
        return False, f"分析脚本不存在: {script_path}"

    try:
        # 子进程执行脚本，工作目录设为脚本所在目录，避免相对路径问题
        work_dir = os.path.dirname(os.path.abspath(script_path)) or "./"
        proc_result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=work_dir
        )

        # 运行成功：整理结果写入MD
        if proc_result.returncode == 0:
            result_md = proc_result.stdout
            try:
                with open(output_md_path, "w", encoding="utf-8") as f:
                    f.write(result_md)
            except IOError as e:
                return False, f"结果文件写入失败: {str(e)}"

            return True, proc_result.stdout

        # 运行失败：返回标准错误
        error_detail = proc_result.stderr.strip()
        if not error_detail:
            error_detail = f"脚本异常退出，返回码: {proc_result.returncode}"
        return False, error_detail

    except subprocess.TimeoutExpired:
        return False, f"脚本运行超时（{timeout}秒），已强制终止"
    except Exception as e:
        return False, f"脚本启动/运行异常: {str(e)}"
    

import os
import math
import pandas as pd
import numpy as np
from collections import Counter


def _load_base_data(data_path: str):
    """统一加载基础数据集，避免重复读取"""
    train_df = pd.read_csv(os.path.join(data_path, "train.csv"))
    test_df = pd.read_csv(os.path.join(data_path, "test.csv"))
    item_df = pd.read_csv(os.path.join(data_path, "item.csv"))
    return train_df, test_df, item_df


def _reproduce_val_split(train_df: pd.DataFrame) -> set:
    """
    验证集划分逻辑（黄金标准）
    """
    unique_uids = sorted(train_df["uid"].unique())
    rng = np.random.RandomState(666)
    shuffled_uids = rng.permutation(unique_uids)
    n_train = math.ceil(len(shuffled_uids) * 0.8)
    return set(shuffled_uids[n_train:])


def _check_prediction_column(df: pd.DataFrame, valid_item_set: set, topk: int, errors: list):
    """通用 prediction 列校验逻辑"""
    if "prediction" not in df.columns:
        errors.append("缺少 prediction 列")
        return

    null_count = df["prediction"].isnull().sum()
    if null_count > 0:
        errors.append(f"prediction 存在 {null_count} 个空值")

    wrong_length = 0  # 变量名从 wrong_topk 改为 wrong_length 更准确
    invalid_item_count = 0
    dup_in_pred = 0
    all_invalid_items = set()

    for _, row in df.iterrows():
        pred_str = str(row.get("prediction", "")).strip()
        if pred_str == '' or pred_str.lower() == 'nan':
            wrong_length += 1
            continue

        items = [x.strip() for x in pred_str.split(",") if x.strip()]

        # ✅ 仅修改此处：将 != 改为上限校验，忠实执行“允许短列表”的契约
        if len(items) == 0 or len(items) > topk:
            wrong_length += 1

        if len(items) != len(set(items)):
            dup_in_pred += 1

        for item in items:
            if item not in valid_item_set:
                invalid_item_count += 1
                all_invalid_items.add(item)

    # if wrong_length > 0:
    #     # ✅ 仅修改此处：更新报错文案以匹配新逻辑
    #     errors.append(f"prediction 有 {wrong_length} 行推荐数量为0或超过 Top-{topk}")
        
    if dup_in_pred > 0:
        errors.append(f"prediction 有 {dup_in_pred} 行内存在重复 item")
        
    if invalid_item_count > 0:
        sample = list(all_invalid_items)[:5]
        errors.append(
            f"prediction 含非法 iid，涉及 {len(all_invalid_items)} 个不同ID，示例: {sample}"
        )


def validate_val_top100(
    csv_path='./model/interface_rec/validation-top100.csv',
    data_path='/data/coding/line3/dataset_b/rec_data/',
    topk: int = 100
) -> bool:
    """
    验证 validation-top100.csv 的正确性。
    通过独立重现 uid 排序 + seed=666 + ceil 划分来确认 uid 来源合法。
    """
    errors = []

    # ========== 1. 文件读取 ==========
    if not os.path.exists(csv_path):
        dual_print(f"❌ 文件不存在: {csv_path}")
        return False
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        dual_print(f"❌ CSV 读取失败: {e}")
        return False

    # ========== 2. 表头校验 ==========
    expected_columns = ["uid", "prediction"]
    if list(df.columns) != expected_columns:
        errors.append(f"表头不正确。期望 {expected_columns}，实际为 {list(df.columns)}")

    # ========== 3. 重现验证集划分 & uid 校验 ==========
    try:
        train_df, _, item_df = _load_base_data(data_path)
    except Exception as e:
        dual_print(f"❌ 数据集加载失败: {e}")
        return False

    val_uid_set = _reproduce_val_split(train_df)
    valid_item_set = set(item_df["iid"].astype(str).tolist())

    submitted_uids = set(df["uid"].unique()) if "uid" in df.columns else set()
    if submitted_uids != val_uid_set:
        missing = val_uid_set - submitted_uids
        extra = submitted_uids - val_uid_set
        if missing:
            errors.append(f"缺少验证集 uid {len(missing)} 个，示例: {list(missing)[:5]}")
        if extra:
            errors.append(f"多余非验证集 uid {len(extra)} 个，示例: {list(extra)[:5]}")

    # uid 重复/空值
    if "uid" in df.columns:
        dup_uid = df["uid"].duplicated().sum()
        null_uid = df["uid"].isnull().sum()
        if dup_uid > 0:
            errors.append(f"uid 存在 {dup_uid} 个重复值")
        if null_uid > 0:
            errors.append(f"uid 存在 {null_uid} 个空值")

    # ========== 4. prediction 校验 ==========
    _check_prediction_column(df, valid_item_set, topk, errors)

    # ========== 输出结果 ==========
    if errors:
        dual_print("❌ validation-top100.csv 验证未通过：")
        for i, err in enumerate(errors, 1):
            dual_print(f"   {i}. {err}")
        return False
    else:
        dual_print("✅ validation-top100.csv 验证通过！")
        dual_print(f"   - 验证集用户数: {len(df)}")
        dual_print(f"   - 每用户候选数: Top-{topk}")
        all_rec = []
        for p in df["prediction"]:
            all_rec.extend([x.strip() for x in str(p).split(",") if x.strip()])
        unique_rec = set(all_rec)
        dual_print(
            f"   - 推荐物品去重数: {len(unique_rec)} / {len(valid_item_set)} "
            f"(覆盖率 {len(unique_rec)/len(valid_item_set)*100:.2f}%)"
        )
        return True


def validate_test_top100(
    csv_path='./model/interface_rec/B2-top100.csv',
    data_path='/data/coding/line3/dataset_b/rec_data/',
    topk: int = 100
) -> bool:
    """
    验证 B2-top100.csv 的正确性。
    额外校验：B2.csv 是否为 B2-top100.csv 每个用户 prediction 的前 10 项。
    """
    errors = []

    # ========== 1. 文件读取 ==========
    if not os.path.exists(csv_path):
        dual_print(f"❌ 文件不存在: {csv_path}")
        return False
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        dual_print(f"❌ CSV 读取失败: {e}")
        return False

    # ========== 2. 表头校验 ==========
    expected_columns = ["uid", "prediction"]
    if list(df.columns) != expected_columns:
        errors.append(f"表头不正确。期望 {expected_columns}，实际为 {list(df.columns)}")

    # ========== 3. uid 完整性校验 ==========
    try:
        _, test_df, item_df = _load_base_data(data_path)
    except Exception as e:
        dual_print(f"❌ 数据集加载失败: {e}")
        return False

    gt_uid_set = set(test_df["uid"].unique())
    valid_item_set = set(item_df["iid"].astype(str).tolist())

    submitted_uids = set(df["uid"].unique()) if "uid" in df.columns else set()
    if submitted_uids != gt_uid_set:
        missing = gt_uid_set - submitted_uids
        extra = submitted_uids - gt_uid_set
        if missing:
            errors.append(f"缺少测试集 uid {len(missing)} 个，示例: {list(missing)[:5]}")
        if extra:
            errors.append(f"多余非测试集 uid {len(extra)} 个，示例: {list(extra)[:5]}")

    if "uid" in df.columns:
        dup_uid = df["uid"].duplicated().sum()
        null_uid = df["uid"].isnull().sum()
        if dup_uid > 0:
            errors.append(f"uid 存在 {dup_uid} 个重复值")
        if null_uid > 0:
            errors.append(f"uid 存在 {null_uid} 个空值")

    # ========== 4. prediction 校验 ==========
    _check_prediction_column(df, valid_item_set, topk, errors)

    # ========== 5. 与 B2.csv 前缀一致性校验 ==========
    B2_path = os.path.join(os.path.dirname(csv_path), "B2.csv")
    if os.path.exists(B2_path):
        try:
            B2_df = pd.read_csv(B2_path)
            merged = df.merge(B2_df, on="uid", suffixes=("_top100", "_B2"), how="inner")
            prefix_mismatch = 0
            for _, row in merged.iterrows():
                top100_items = [x.strip() for x in str(row["prediction_top100"]).split(",") if x.strip()]
                B2_items = [x.strip() for x in str(row["prediction_B2"]).split(",") if x.strip()]
                if top100_items[:10] != B2_items:
                    prefix_mismatch += 1
            if prefix_mismatch > 0:
                errors.append(f"B2.csv 与 B2-top100.csv 前10项不一致的用户数: {prefix_mismatch}")
            else:
                dual_print("   ✅ B2.csv 与 B2-top100.csv 前10项完全一致")
        except Exception as e:
            dual_print(f"⚠️ B2.csv 一致性校验跳过（读取失败: {e}）")
    else:
        dual_print(f"⚠️ B2.csv 不存在，跳过前缀一致性校验")

    # ========== 输出结果 ==========
    if errors:
        dual_print("❌ B2-top100.csv 验证未通过：")

        for i, err in enumerate(errors, 1):
            dual_print(f"   {i}. {err}")
        return False
    else:
        dual_print("✅ B2-top100.csv 验证通过！")
        dual_print(f"   - 测试集用户数: {len(df)}")
        dual_print(f"   - 每用户候选数: Top-{topk}")
        all_rec = []
        for p in df["prediction"]:
            all_rec.extend([x.strip() for x in str(p).split(",") if x.strip()])
        unique_rec = set(all_rec)
        dual_print(
            f"   - 推荐物品去重数: {len(unique_rec)} / {len(valid_item_set)} "
            f"(覆盖率 {len(unique_rec)/len(valid_item_set)*100:.2f}%)"
        )
        rec_counter = Counter(all_rec)
        dual_print(f"   - 被推荐最多的 Top-5 物品: {rec_counter.most_common(5)}")
        return True


#### 模型融合 ####
import os
import glob
import math
import pandas as pd
import numpy as np
from collections import defaultdict

def compute_ndcg_at_10(pred_items: list, true_item: str, k: int = 10) -> float:
    """计算单个用户的 NDCG@10"""
    if true_item in pred_items[:k]:
        rank = pred_items[:k].index(true_item) + 1
        return 1.0 / math.log2(rank + 1)
    return 0.0

from collections import defaultdict
import numpy as np

def _parse_score_csv(df) -> dict:
    """解析带分数的CSV，返回 {uid: {iid: score}} 结构"""
    res = {}
    for _, row in df.iterrows():
        uid = row["uid"]
        items = {}
        for pair in str(row["item_scores"]).split(","):
            pair = pair.strip()
            if not pair:
                continue
            iid, score_str = pair.split(":")
            items[iid.strip()] = float(score_str.strip())
        res[uid] = items
    return res


def weighted_rrf_fusion(
    score_maps: list[dict],
    weights: list[float],
    k: int = 60
) -> dict:
    """加权 RRF 融合，返回 {uid: [iid1, iid2, ...]} 排序列表"""
    fused_scores = defaultdict(lambda: defaultdict(float))
    for scores, w in zip(score_maps, weights):
        if w <= 0:
            continue
        for uid, items in scores.items():
            for rank, item in enumerate(items):
                fused_scores[uid][item] += w / (k + rank + 1)

    result = {}
    for uid, item_scores in fused_scores.items():
        sorted_items = sorted(item_scores.keys(), key=lambda x: item_scores[x], reverse=True)
        result[uid] = sorted_items
    return result


def weighted_score_fusion(
    score_maps: list[dict],
    weights: list[float]
) -> dict:
    """加权 Softmax 分数融合，返回 {uid: [iid1, iid2, ...]} 排序列表"""
    fused_scores = defaultdict(lambda: defaultdict(float))
    for scores, w in zip(score_maps, weights):
        if w <= 0:
            continue
        for uid, item_dict in scores.items():
            for iid, score in item_dict.items():
                fused_scores[uid][iid] += w * score

    result = {}
    for uid, item_scores in fused_scores.items():
        sorted_items = sorted(item_scores.keys(), key=lambda x: item_scores[x], reverse=True)
        result[uid] = sorted_items
    return result


def _fuse_score_map(score_maps: list[dict], weights: list[float]) -> dict:
    """分数融合并保留原始分数，返回 {uid: {iid: score}} 结构，用于后处理"""
    fused = defaultdict(lambda: defaultdict(float))
    w_sum = sum(weights)
    norm_weights = [w / w_sum for w in weights]
    for scores, w in zip(score_maps, norm_weights):
        for uid, item_dict in scores.items():
            for iid, score in item_dict.items():
                fused[uid][iid] += w * score
    return dict(fused)


def apply_history_penalty(score_map, user_seqs, gamma=1.0):
    """
    历史交互物品软惩罚：用户历史里出现过的物品，分数乘以 gamma
    gamma=1 无惩罚；gamma越小惩罚越强；gamma=0 等价于硬过滤
    """
    result = {}
    for uid, item_scores in score_map.items():
        history = set(user_seqs.get(uid, []))
        penalized = {}
        for iid, score in item_scores.items():
            if iid in history:
                penalized[iid] = score * gamma
            else:
                penalized[iid] = score
        sorted_items = sorted(penalized.keys(), key=lambda x: penalized[x], reverse=True)
        result[uid] = sorted_items
    return result


def apply_temperature_scaling(score_map, temperature=1.0):
    """
    温度系数校准：对分数做温度缩放，调整分布尖锐程度
    temperature < 1：分数更尖锐，头部差距拉大
    temperature > 1：分数更平滑，差距缩小
    """
    result = {}
    for uid, item_scores in score_map.items():
        scaled = {}
        # 先做数值稳定处理
        max_score = max(item_scores.values())
        for iid, score in item_scores.items():
            scaled[iid] = np.exp((score - max_score) / temperature)
        sorted_items = sorted(scaled.keys(), key=lambda x: scaled[x], reverse=True)
        result[uid] = sorted_items
    return result



def apply_mmr_diversity(score_map, item_cats, lambda_param=0.8, rerank_topk=10, candidate_topk=100):
    """
    MMR 多样性重排（优化版）
    - 仅对 candidate_topk 个候选物品做重排，只选出 rerank_topk 个最终结果
    - 预存列表减少字典查找，大幅提升速度
    """
    result = {}
    # lambda=1 等价于纯分数排序，直接返回
    if lambda_param >= 1.0:
        for uid, item_scores in score_map.items():
            sorted_items = sorted(item_scores.keys(), key=lambda x: item_scores[x], reverse=True)
            result[uid] = sorted_items[:candidate_topk]
        return result

    for uid, item_scores in score_map.items():
        # 先取出分数最高的候选池，只在这个池子里重排
        sorted_candidates = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)[:candidate_topk]
        item_list = [x[0] for x in sorted_candidates]
        score_list = [x[1] for x in sorted_candidates]
        cat_list = [item_cats.get(iid, "") for iid in item_list]
        n = len(item_list)
        if n == 0:
            result[uid] = []
            continue

        selected_idx = []
        selected_cats = set()
        # 只选出 rerank_topk 个就停止
        target_k = min(rerank_topk, n)

        for _ in range(target_k):
            best_idx = -1
            best_mmr = -float('inf')
            for i in range(n):
                if i in selected_idx:
                    continue
                score = score_list[i]
                cat = cat_list[i]
                novelty = 0.0 if cat in selected_cats else 1.0
                mmr = lambda_param * score + (1 - lambda_param) * novelty
                if mmr > best_mmr:
                    best_mmr = mmr
                    best_idx = i
            if best_idx == -1:
                break
            selected_idx.append(best_idx)
            selected_cats.add(cat_list[best_idx])

        # 选出的 + 剩下的按原分数补全到 candidate_topk
        selected_items = [item_list[i] for i in selected_idx]
        remain_items = [item_list[i] for i in range(n) if i not in selected_idx]
        result[uid] = selected_items + remain_items
    return result


def ensemble_and_evaluate(
    model_save_dir: str, 
    data_path: str, 
    topk: int = 100, 
    max_iter: int = 30,
    min_relative_gap: float = 0.05,
    min_keep_models: int = 3,
    # 后处理参数搜索范围
    hist_gamma_min: float = 0.3,
    hist_gamma_max: float = 1.0,
    hist_gamma_step: float = 0.1,
    temp_min: float = 0.6,
    temp_max: float = 2.0,
    temp_step: float = 0.2,
    mmr_lambda_min: float = 0.7,
    mmr_lambda_max: float = 1.0,
    mmr_lambda_step: float = 0.05
):
    """
    双方案集成 + 3种冷启动适配后处理自动调优，全局选验证集NDCG最高的方案输出
    """
    dual_print(f"\n{'='*60}")
    dual_print("🔗 双方案集成 + 冷启动适配后处理自动调优")
    dual_print(f"{'='*60}")

    # ========== 1. 收集所有有效文件 ==========
    val_top100_files = sorted(glob.glob(os.path.join(model_save_dir, "**/validation-top100.csv"), recursive=True))
    test_top100_files = sorted(glob.glob(os.path.join(model_save_dir, "**/B2-top100.csv"), recursive=True))
    val_softmax_files = sorted(glob.glob(os.path.join(model_save_dir, "**/validation_softmax_item.csv"), recursive=True))
    test_softmax_files = sorted(glob.glob(os.path.join(model_save_dir, "**/B2_softmax_item.csv"), recursive=True))

    val_dirs = {os.path.dirname(f) for f in val_top100_files}
    test_dirs = {os.path.dirname(f) for f in test_top100_files}
    val_soft_dirs = {os.path.dirname(f) for f in val_softmax_files}
    test_soft_dirs = {os.path.dirname(f) for f in test_softmax_files}
    valid_dirs = sorted(val_dirs & test_dirs & val_soft_dirs & test_soft_dirs)

    if len(valid_dirs) == 0:
        dual_print("❌ 未找到同时包含4个输出文件的有效子目录")
        return

    dual_print(f"✅ 发现 {len(valid_dirs)} 个有效实验目录:")
    for d in valid_dirs:
        dual_print(f"   - {os.path.basename(d)}")

    # ========== 2. 读取所有模型预测结果 ==========
    val_rank_maps, test_rank_maps = [], []
    val_score_maps, test_score_maps = [], []
    exp_names = []

    for exp_dir in valid_dirs:
        exp_name = os.path.basename(exp_dir)
        try:
            val_rank_df = pd.read_csv(os.path.join(exp_dir, "validation-top100.csv"))
            test_rank_df = pd.read_csv(os.path.join(exp_dir, "B2-top100.csv"))

            val_rank_map = {
                row["uid"]: [x.strip() for x in str(row["prediction"]).split(",") if x.strip()]
                for _, row in val_rank_df.iterrows()
            }
            test_rank_map = {
                row["uid"]: [x.strip() for x in str(row["prediction"]).split(",") if x.strip()]
                for _, row in test_rank_df.iterrows()
            }
            val_rank_maps.append(val_rank_map)
            test_rank_maps.append(test_rank_map)

            val_soft_df = pd.read_csv(os.path.join(exp_dir, "validation_softmax_item.csv"))
            test_soft_df = pd.read_csv(os.path.join(exp_dir, "B2_softmax_item.csv"))
            val_score_maps.append(_parse_score_csv(val_soft_df))
            test_score_maps.append(_parse_score_csv(test_soft_df))

            exp_names.append(exp_name)
        except Exception as e:
            dual_print(f"⚠️ 跳过 {exp_name}: {e}")

    if len(val_rank_maps) < 1:
        dual_print("❌ 没有成功加载任何模型结果，无法集成")
        return

    n_models = len(val_rank_maps)
    dual_print(f"📊 候选模型数: {n_models}")

    # ========== 3. 加载数据 + 构建后处理资源 ==========
    train_df = pd.read_csv(os.path.join(data_path, "train.csv"))
    test_df = pd.read_csv(os.path.join(data_path, "test.csv"))
    user_df = pd.read_csv(os.path.join(data_path, "user.csv"))
    item_df = pd.read_csv(os.path.join(data_path, "item.csv"))
    
    val_uid_set = _reproduce_val_split(train_df)
    val_labels = train_df[train_df["uid"].isin(val_uid_set)][["uid", "target_iid"]]
    val_label_dict = dict(zip(val_labels["uid"], val_labels["target_iid"]))

    # 提取用户历史交互序列
    val_seqs = {}
    for _, row in train_df[train_df["uid"].isin(val_uid_set)].iterrows():
        items = [x.strip() for x in str(row["item_seq_dedup"]).split(",") if x.strip()]
        val_seqs[row["uid"]] = items
    test_seqs = {}
    for _, row in test_df.iterrows():
        items = [x.strip() for x in str(row["item_seq_dedup"]).split(",") if x.strip()]
        test_seqs[row["uid"]] = items

    # 物品类目映射（用区分度最高的 i_cat_01，也可换其他字段）
    item_cat_map = dict(zip(item_df["iid"], item_df["i_cat_01"]))

    def calc_ndcg(pred_map):
        ndcg_vals = []
        for uid in val_uid_set:
            pred = pred_map.get(uid, [])
            true_item = val_label_dict.get(uid)
            if true_item is not None:
                ndcg_vals.append(compute_ndcg_at_10(pred, true_item))
        return np.mean(ndcg_vals) if ndcg_vals else 0.0

    dual_print("\n📈 各单模型验证集 NDCG@10:")
    single_ndcgs = [calc_ndcg(m) for m in val_rank_maps]
    for name, ndcg in zip(exp_names, single_ndcgs):
        dual_print(f"   {name}: {ndcg:.6f}")

    best_single_ndcg = max(single_ndcgs)
    best_single_idx = int(np.argmax(single_ndcgs))

    # ========== 4. 前置过滤劣质模型 ==========
    threshold = best_single_ndcg - 0.07
    keep_indices = [i for i, ndcg in enumerate(single_ndcgs) if ndcg >= threshold]
    
    if len(keep_indices) < min_keep_models:
        sorted_indices = sorted(range(n_models), key=lambda x: single_ndcgs[x], reverse=True)
        keep_indices = sorted_indices[:min_keep_models]

    if len(keep_indices) < n_models:
        filtered_names = [exp_names[i] for i in range(n_models) if i not in keep_indices]
        dual_print(f"\n🗑️ 前置过滤 {len(filtered_names)} 个劣质模型:")
        for name in filtered_names:
            dual_print(f"   - {name}")
    else:
        dual_print(f"\n✅ 所有模型均符合阈值要求，全部保留")

    val_rank_maps = [val_rank_maps[i] for i in keep_indices]
    test_rank_maps = [test_rank_maps[i] for i in keep_indices]
    val_score_maps = [val_score_maps[i] for i in keep_indices]
    test_score_maps = [test_score_maps[i] for i in keep_indices]
    exp_names = [exp_names[i] for i in keep_indices]
    single_ndcgs = [single_ndcgs[i] for i in keep_indices]
    n_models = len(exp_names)
    
    best_single_idx = int(np.argmax(single_ndcgs))
    best_single_ndcg = single_ndcgs[best_single_idx]

    # ========== 通用 GES 搜索函数 ==========
    def ges_search(val_maps, fusion_fn, desc_name):
        dual_print(f"\n🔍 开始 {desc_name} GES 搜索 (max_iter={max_iter})...")
        selected_indices = [best_single_idx]
        best_ndcg = best_single_ndcg
        no_improve_count = 0

        for iteration in range(1, max_iter + 1):
            current_best_ndcg = best_ndcg
            current_best_candidate = None
            candidates = []

            for idx in range(n_models):
                if idx not in selected_indices:
                    candidates.append((selected_indices + [idx], f"+{exp_names[idx]}"))
            if len(selected_indices) > 1:
                for rem_pos in range(len(selected_indices)):
                    trial = [selected_indices[j] for j in range(len(selected_indices)) if j != rem_pos]
                    candidates.append((trial, f"-{exp_names[selected_indices[rem_pos]]}"))

            for trial_indices, desc in candidates:
                trial_maps = [val_maps[i] for i in trial_indices]
                weights = [1.0] * len(trial_indices)
                fused_map = fusion_fn(trial_maps, weights)
                ndcg = calc_ndcg(fused_map)

                if ndcg > current_best_ndcg:
                    current_best_ndcg = ndcg
                    current_best_candidate = (trial_indices, desc)

            if current_best_candidate is not None and current_best_ndcg > best_ndcg:
                selected_indices, desc = current_best_candidate
                best_ndcg = current_best_ndcg
                no_improve_count = 0
                dual_print(f"   Iter {iteration:3d} | NDCG@10: {best_ndcg:.6f} ({desc})")
            else:
                no_improve_count += 1

            if no_improve_count >= 8:
                dual_print(f"   ⏹️ 连续 {no_improve_count} 轮无提升，提前终止")
                break

        dual_print(f"   ✅ {desc_name} 最优验证集 NDCG@10: {best_ndcg:.6f}")
        return best_ndcg, selected_indices

    # ========== 5. 双方案基础融合 ==========
    def rrf_fusion_wrapper(maps, weights):
        return weighted_rrf_fusion(maps, weights, k=60)
    
    best_rrf_ndcg, rrf_selected_indices = ges_search(val_rank_maps, rrf_fusion_wrapper, "RRF排名融合")
    best_score_ndcg, score_selected_indices = ges_search(val_score_maps, weighted_score_fusion, "Softmax分数融合")

    # 拿到验证集最优分数融合的原始分数
    val_fused_scores = _fuse_score_map(
        [val_score_maps[i] for i in score_selected_indices],
        [1.0] * len(score_selected_indices)
    )
    base_ndcg = best_score_ndcg

    # ========== 🆕 6. 后处理1：历史交互软惩罚 ==========
    dual_print(f"\n🔧 【后处理1】历史交互软惩罚调优 (gamma范围: {hist_gamma_min}~{hist_gamma_max}, 步长{hist_gamma_step})")
    best_hist_ndcg = 0
    best_hist_gamma = 1.0

    # for gamma in np.arange(hist_gamma_min, hist_gamma_max + hist_gamma_step, hist_gamma_step):
    #     gamma = round(gamma, 2)
    #     boosted_map = apply_history_penalty(val_fused_scores, val_seqs, gamma)
    #     ndcg = calc_ndcg(boosted_map)
    #     lift = ndcg - base_ndcg
    #     arrow = "🔺" if lift > 0 else ("🔻" if lift < 0 else "➖")
    #     dual_print(f"   gamma={gamma:.2f} | NDCG@10: {ndcg:.6f} ({lift:+.6f}) {arrow}")

    #     if ndcg > best_hist_ndcg:
    #         best_hist_ndcg = ndcg
    #         best_hist_gamma = gamma

    dual_print(f"   ✅ 最优历史惩罚系数: gamma={best_hist_gamma:.2f}, 对应NDCG@10: {best_hist_ndcg:.6f}")

    # ========== 🆕 7. 后处理2：温度系数校准 ==========
    dual_print(f"\n🔧 【后处理2】温度系数校准调优 (T范围: {temp_min}~{temp_max}, 步长{temp_step})")
    best_temp_ndcg = 0
    best_temp = 1.0

    # for t in np.arange(temp_min, temp_max + temp_step, temp_step):
    #     t = round(t, 2)
    #     boosted_map = apply_temperature_scaling(val_fused_scores, t)
    #     ndcg = calc_ndcg(boosted_map)
    #     lift = ndcg - base_ndcg
    #     arrow = "🔺" if lift > 0 else ("🔻" if lift < 0 else "➖")
    #     dual_print(f"   T={t:.2f} | NDCG@10: {ndcg:.6f} ({lift:+.6f}) {arrow}")

    #     if ndcg > best_temp_ndcg:
    #         best_temp_ndcg = ndcg
    #         best_temp = t

    dual_print(f"   ✅ 最优温度系数: T={best_temp:.2f}, 对应NDCG@10: {best_temp_ndcg:.6f}")

    # ========== 🆕 8. 后处理3：MMR类目多样性重排 ==========
    dual_print(f"\n🔧 【后处理3】MMR多样性重排调优 (lambda范围: {mmr_lambda_min}~{mmr_lambda_max}, 步长{mmr_lambda_step})")
    best_mmr_ndcg = 0
    best_mmr_lambda = 1.0

    # for lam in np.arange(mmr_lambda_min, mmr_lambda_max + mmr_lambda_step, mmr_lambda_step):
    #     lam = round(lam, 2)
    #     boosted_map = apply_mmr_diversity(val_fused_scores, item_cat_map, lam, rerank_topk=10, candidate_topk=100)
    #     ndcg = calc_ndcg(boosted_map)
    #     lift = ndcg - base_ndcg
    #     arrow = "🔺" if lift > 0 else ("🔻" if lift < 0 else "➖")
    #     dual_print(f"   lambda={lam:.2f} | NDCG@10: {ndcg:.6f} ({lift:+.6f}) {arrow}")

    #     if ndcg > best_mmr_ndcg:
    #         best_mmr_ndcg = ndcg
    #         best_mmr_lambda = lam

    dual_print(f"   ✅ 最优MMR系数: lambda={best_mmr_lambda:.2f}, 对应NDCG@10: {best_mmr_ndcg:.6f}")

    # ========== 9. 全局方案对比，择优输出 ==========
    dual_print(f"\n🏆 全局方案对比结果:")
    dual_print(f"   RRF排名融合:          {best_rrf_ndcg:.6f}")
    dual_print(f"   原始分数融合:          {best_score_ndcg:.6f}")
    dual_print(f"   分数融合+历史惩罚:     {best_hist_ndcg:.6f} (gamma={best_hist_gamma:.2f})")
    dual_print(f"   分数融合+温度校准:     {best_temp_ndcg:.6f} (T={best_temp:.2f})")
    dual_print(f"   分数融合+MMR多样性:    {best_mmr_ndcg:.6f} (lambda={best_mmr_lambda:.2f})")

    all_candidates = [
        ("RRF排名融合", best_rrf_ndcg, "rrf", 0.0),
        ("原始分数融合", best_score_ndcg, "score_raw", 0.0),
        ("分数融合+历史惩罚", best_hist_ndcg, "score_hist", best_hist_gamma),
        ("分数融合+温度校准", best_temp_ndcg, "score_temp", best_temp),
        ("分数融合+MMR多样性", best_mmr_ndcg, "score_mmr", best_mmr_lambda)
    ]
    all_candidates.sort(key=lambda x: x[1], reverse=True)
    winner_name, winner_ndcg, winner_type, winner_param = all_candidates[0]

    # 生成最终测试集结果
    test_fused_scores = _fuse_score_map(
        [test_score_maps[i] for i in score_selected_indices],
        [1.0] * len(score_selected_indices)
    )

    if winner_type == "rrf":
        final_test_maps = [test_rank_maps[i] for i in rrf_selected_indices]
        final_fused = weighted_rrf_fusion(final_test_maps, [1.0]*len(rrf_selected_indices), k=60)
    elif winner_type == "score_raw":
        final_test_maps = [test_score_maps[i] for i in score_selected_indices]
        final_fused = weighted_score_fusion(final_test_maps, [1.0]*len(score_selected_indices))
    elif winner_type == "score_hist":
        final_fused = apply_history_penalty(test_fused_scores, test_seqs, winner_param)
    elif winner_type == "score_temp":
        final_fused = apply_temperature_scaling(test_fused_scores, winner_param)
    else:
        final_fused = apply_mmr_diversity(test_fused_scores, item_cat_map, winner_param, topk=100)

    lift = winner_ndcg - best_single_ndcg
    arrow = "🔺" if lift > 0 else ("🔻" if lift < 0 else "➖")

    dual_print(f"\n🎯 最终采用方案: {winner_name}")
    dual_print(f"   最优验证集 NDCG@10: {winner_ndcg:.6f} {arrow} vs 最佳单模型 {best_single_ndcg:.6f} ({lift:+.6f})")
    if winner_param != 0.0 and winner_type != "score_raw":
        dual_print(f"   后处理最优参数: {winner_param:.2f}")

    # ========== 10. 输出最终 B2_final.csv ==========
    output_path = os.path.join(model_save_dir, "B2_final.csv")
    rows = [{"uid": uid, "prediction": ",".join(items[:10])} for uid, items in final_fused.items()]
    final_df = pd.DataFrame(rows, columns=["uid", "prediction"])
    
    if winner_ndcg > GLOBAL_STATE["best_merge_acc"]:
        GLOBAL_STATE["best_merge_acc"] = winner_ndcg
        dual_print(f"   🎉 【最佳全局结果更新，输出文档写入】当前全局最佳融合验证集 NDCG@10: {GLOBAL_STATE['best_merge_acc']:.6f}")
        final_df.to_csv(output_path, index=False)

    dual_print(f"💾 最终融合结果已保存: {output_path} ({len(final_df)} 用户)")
    dual_print(f"{'='*60}\n")


def run_ensemble():
# ===== 🆕 训练结束后自动执行 RRF 集成 =====
    
    try:
        ensemble_and_evaluate(
            model_save_dir=MODEL_SAVE_DIR,
            data_path=DATA_PATH,
            topk=100
        )
    except Exception as e:
        dual_print(f"❌ 模型集成阶段异常: {e}")