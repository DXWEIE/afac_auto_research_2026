

api_key = 'sk-xxxx'

from collections import Counter
import csv
import datetime
import math
import time
import os
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


# ==================================================
# 全局配置与公共状态（后续可抽到单独config文件）
# ==================================================
# 模型与接口配置
default_model = "qwen3.6-plus"

global_class_number = 8
# 全局路径配置
global_history_acc_md_path = "./model_save_vb17/cls_history_acc_summary.md"   # 全局训练结果总结MD
global_data_inspect_md_path = "./model_save_vb17/cls_data_inspection_result.md" # 数据探查结果MD
base_template_path = "./cls_base_template.py"                 # 初始接口模板文件
model_save_dir = "./model_save_vb17/cls_model/"                 # 模型与代码保存目录
cls_data_base_dir = '/data/coding/line3/dataset_b/cls_data/'
fixed_data_path = "/data/coding/line3/dataset_b/cls_data/B1.npz"
MODEL_SAVE_DIR = "./model_save_vb17/cls_model/"
DATA_PATH = "/data/coding/line3/dataset_b/cls_data/"
NUM_CLASSES = global_class_number  # ← 修改类别数只需改这里

global_best_acc = 0
global_best_code = ""
global_best_code_path = ""
global_best_model_summary = ""

global_best_merge_acc = 0

import json
import os
from datetime import datetime

LOG_PATH = "./model_save_vb17/trajectory_B1.json"


import json
import os
from datetime import datetime

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


data_analyse_plan_template = """你是一个图神经网络模型训练与调优专家。你的目标是通过分析数据集特征，为后续模型优化提供依据。

### 建模任务
{task_description}

### 历史模型训练情况(如有)
{history_train_summary}

### 数据集基本信息
{dataset_description}

请你结合上述信息，制定一份针对性的数据探查方案，以明确数据分布特性并指导后续优化。探查方案必须包含以下核心维度：
1. 同质性分析：计算边同质性比率(edge homophily ratio)，判断图是同质还是异质；
2. 度数分布分析：统计节点度数的均值、中位数、最大值及长尾比例，识别是否存在超级枢纽节点，判断是否需要使用或调整归一化方式或添加自环；
3. 特征稀疏度分析：计算节点特征矩阵的非零元密度，评估是否需要进行特征填充、降维或切换对稀疏特征友好的模型；
4. 标签分布分析：检查训练集类别是否均衡，若存在严重长尾需规划重采样或损失函数加权策略。

请直接输出你打算进行的明确的数据探查方案，以纯文本的形式输出，简要明确，300字以内：
"""


cls_descrition_template = """
你是一个图神经网络模型训练专家，请按照以下要求完成代码编写。
"""

cls_data_explanation_template = """# 数据文件说明
文件夹为{data_base_dir}
该文件夹内包含节点分类数据集 `.npz` 文件，另外包含对应测试集提交模板 CSV 文件。所有节点编号均为从 `0` 开始的整数编号。

## 数据集
| 文件 | 节点数 | 特征维度 | 类别数 | 训练节点数 | 测试节点数 | 邻接矩阵非零项 | 特征矩阵非零项 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `B1.npz` | 7650 | 512 | 8 | 6120 | 1530 | 191154 | 3916800 |

## `.npz` 变量含义

| 变量名 | 含义 |
| --- | --- |
| `adj_data` | 图邻接矩阵的 CSR `data` 数组，表示非零边的权重。当前文件中边权均为 `1.0`。 |
| `adj_indices` | 图邻接矩阵的 CSR `indices` 数组，表示每个非零边所在的列编号，也就是目标节点编号。 |
| `adj_indptr` | 图邻接矩阵的 CSR `indptr` 数组，长度为 `节点数 + 1`。节点 `i` 的邻接非零项位于 `adj_data[adj_indptr[i]:adj_indptr[i+1]]` 和 `adj_indices[adj_indptr[i]:adj_indptr[i+1]]`。 |
| `adj_shape` | 图邻接矩阵形状，当前为 `(7650, 7650)`。请按文件中保存的邻接关系原样使用，不额外假设一定无向、对称或无自环。 |
| `attr_data` | 节点特征矩阵的 CSR `data` 数组，表示非零特征值。 |
| `attr_indices` | 节点特征矩阵的 CSR `indices` 数组，表示每个非零特征所在的特征列编号。 |
| `attr_indptr` | 节点特征矩阵的 CSR `indptr` 数组，长度为 `节点数 + 1`。节点 `i` 的特征非零项位于 `attr_data[attr_indptr[i]:attr_indptr[i+1]]` 和 `attr_indices[attr_indptr[i]:attr_indptr[i+1]]`。 |
| `attr_shape` | 节点特征矩阵形状，当前为 `(7650, 512)`。 |
| `labels` | 节点标签数组，长度为 `7650`。`train_idx` 对应位置为公开标签，取值范围是 `0` 到 `7`；`test_idx` 对应位置在公开文件中统一置为 `-1`，表示测试标签隐藏。 |
| `train_idx` | 可用于训练/验证的节点编号数组。训练和验证节点已合并提供。 |
| `test_idx` | 测试节点编号数组。选手需要对这些节点预测标签，并按提交模板中的 `test_idx` 提交对应 `label`。 |
可以使用如下方式还原邻接矩阵和特征矩阵：

```python
import numpy as np
from scipy.sparse import csr_matrix

data_dir = '{data_base_dir}'
data = np.load(data_dir+"B1.npz")

adj = csr_matrix(
    (data["adj_data"], data["adj_indices"], data["adj_indptr"]),
    shape=tuple(data["adj_shape"]),
)

features = csr_matrix(
    (data["attr_data"], data["attr_indices"], data["attr_indptr"]),
    shape=tuple(data["attr_shape"]),
)

labels = data["labels"]
train_idx = data["train_idx"]
test_idx = data["test_idx"]
```

数据集示例说明：

示意数据集包含 4 个节点（节点编号为 0,1,2,3）和 4 条边，类别数为 2，节点特征维度为 4。其 .npz 文件中各字段可示意如下：

{

   "adj_data": [1.0, 1.0, 1.0, 1.0],

   "adj_indices": [1, 2, 3, 0],

   "adj_indptr": [0, 1, 2, 3, 4],

   "adj_shape": [4, 4],

   "attr_data": [1.0, 0.5, 1.0, 0.2, 1.0, 1.0, 1.0, 1.0],

   "attr_indices": [0, 2, 1, 3, 0, 1, 2, 3],

   "attr_indptr": [0, 2, 4, 6, 8],

   "attr_shape": [4, 4],

   "labels": [0, 1, -1, -1],

   "train_idx": [0, 1],

   "test_idx": [2, 3]

}

最终测试集输出文件命名为B1.csv，必须包含表头，且仅允许包含两列，例如：

test_idx,label

18,3

19,7

其中：test_idx：测试节点编号，必须与数据集保持一致。label：选手预测的类别编号，必须为合法整数类别编号。
""".replace("{data_base_dir}", cls_data_base_dir)

data_meta_info = """## 数据集
| 文件 | 节点数 | 特征维度 | 类别数 | 训练节点数 | 测试节点数 | 邻接矩阵非零项 | 特征矩阵非零项 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `B1.npz` | 7650 | 512 | 8 | 6120 | 1530 | 191154 | 3916800 |

## `.npz` 变量含义

| 变量名 | 含义 |
| --- | --- |
| `adj_data` | 图邻接矩阵的 CSR `data` 数组，表示非零边的权重。当前文件中边权均为 `1.0`。 |
| `adj_indices` | 图邻接矩阵的 CSR `indices` 数组，表示每个非零边所在的列编号，也就是目标节点编号。 |
| `adj_indptr` | 图邻接矩阵的 CSR `indptr` 数组，长度为 `节点数 + 1`。节点 `i` 的邻接非零项位于 `adj_data[adj_indptr[i]:adj_indptr[i+1]]` 和 `adj_indices[adj_indptr[i]:adj_indptr[i+1]]`。 |
| `adj_shape` | 图邻接矩阵形状，当前为 `(7650, 7650)`。请按文件中保存的邻接关系原样使用，不额外假设一定无向、对称或无自环。 |
| `attr_data` | 节点特征矩阵的 CSR `data` 数组，表示非零特征值。 |
| `attr_indices` | 节点特征矩阵的 CSR `indices` 数组，表示每个非零特征所在的特征列编号。 |
| `attr_indptr` | 节点特征矩阵的 CSR `indptr` 数组，长度为 `节点数 + 1`。节点 `i` 的特征非零项位于 `attr_data[attr_indptr[i]:attr_indptr[i+1]]` 和 `attr_indices[attr_indptr[i]:attr_indptr[i+1]]`。 |
| `attr_shape` | 节点特征矩阵形状，当前为 `(7650, 512)`。 |
| `labels` | 节点标签数组，长度为 `7650`。`train_idx` 对应位置为公开标签，取值范围是 `0` 到 `7`；`test_idx` 对应位置在公开文件中统一置为 `-1`，表示测试标签隐藏。 |
| `train_idx` | 可用于训练/验证的节点编号数组。训练和验证节点已合并提供。 |
| `test_idx` | 测试节点编号数组。选手需要对这些节点预测标签，并按提交模板中的 `test_idx` 提交对应 `label`。 |                                                                                          |

可以使用如下方式还原邻接矩阵和特征矩阵：

```python
import numpy as np
from scipy.sparse import csr_matrix

data_dir = '{data_base_dir}'
data = np.load(data_dir+"B1.npz")

adj = csr_matrix(
    (data["adj_data"], data["adj_indices"], data["adj_indptr"]),
    shape=tuple(data["adj_shape"]),
)

features = csr_matrix(
    (data["attr_data"], data["attr_indices"], data["attr_indptr"]),
    shape=tuple(data["attr_shape"]),
)

labels = data["labels"]
train_idx = data["train_idx"]
test_idx = data["test_idx"]
```""".replace("{data_base_dir}", cls_data_base_dir)


data_analyse_template = """你是一个图神经网络模型训练与调优专家。你的目标是通过分析数据集特征，为后续模型优化提供依据。

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
2. 在划分完训练集和验证集之后的代码里面，将训练集规模缩小（使训练集仅使用 1000 条数据，注意有可能前半部分和后半代码都要修改，防止取数报索引越界和device不一致错误；禁止缩小测试集和验证集大小，禁止缩小会输出的csv这部分大小，后续有校验）
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
3. 验证集 ACC > 0.07（冒烟测试数据量极小，仅为排除模型完全失效）
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
  "best_acc": "<float> 最佳验证集ACC",
  "model_and_params": "<string> 简述模型类型及核心配置。包括：1.算法/架构名称(架构名，以及检查是否使用了BN/LayerNorm等归一化层，是否使用了残差连接，是否使用了多头注意力机制，是否使用了跳跃连接等，是否使用了Jumping Knowledge机制，是否使用了DropPath/DropEdge/DropNode等)；2.关键超参(学习率/n_estimators/batch_size/hidden_size/head数/模型层数)；3.核心操作(归一化预处理/正则化/特征工程/数据增强/损失函数设计等)。用精炼的自然语言描述即可。",
  "training_diagnosis": "<string> 训练状态诊断。观察loss/metric变化趋势，判断是否过拟合、欠拟合、震荡或提前停止，是否学习率设置不合理导致学的很慢或者后期震荡，以及对比train与val的差距给出明确结论。",
  "conclusion_and_next": "<string> 100字以内的总结论(优/良/差/失败)，简要明确地提示下一轮需要调整的方向。"
}}

请开始输出：
"""


# train_history_analyse_template = """你是一个图神经网络模型训练与调优专家。你的目标是通过分析历史训练记录和数据集特征，持续提升模型在验证集上的ACC。

# ### 当前状态信息
# 数据集探查信息：
# {data_inspect_result}

# 所有训练尝试：
# {history_train_summary}


# ### 探索方向和思路
# 1. **优先排查特征预处理瓶颈**：原始稀疏高维特征直接输入GNN会导致注意力计算失准、模型学习困难，可能是低ACC的首要诱因。若验证集ACC显著偏低（如<0.6），优先检查是否做了特征标准化+SVD/PCA降维去噪；如果尝试了这些方法仍然无效，可以考虑去除一些标准化和正则化方式看看是否有效果。
# 2. **严格控制正则强度**：禁止特征丢弃、边丢弃、普通dropout多重正则高强度叠加使用，中小数据集下极易造成严重欠拟合，避免过度压制模型容量。
# 3. 实验少于3次时，优先尝试不同架构变体（GATv1、GATv2、MixHop、JKNet等），快速定位适配性最优的基础模型；超过3次后可转向细粒度超参调优。
# 4. 校验图结构完整性：检查边进行有向还是无向化处理，自环配置是否合理，邻居聚合信息是否不全。同时检查邻接矩阵归一化方式，可对比对称归一化、仅加自环的效果差异。
# 5. 模型容量与训练轮次匹配：若ACC低（低于0.75）且无明显过拟合趋势，优先排查是否epoch不足、模型层数太少、head不足、隐藏维度严重不足；epoch上限控制在400以内，配合早停机制。
# 6. 模型架构微调：检查是否使用了BN/LayerNorm等归一化层，是否使用了残差连接，是否使用了多头注意力机制，是否使用了跳跃连接等，是否使用了Jumping Knowledge机制，是否使用了DropPath/DropEdge/DropNode等图结构正则化方法，是否使用了特征降维或特征增强方法。根据验证集ACC变化趋势，适当调整这些结构。
# 7. 配合调整损失函数（交叉熵/Focal Loss/label smoothing）、学习率、权重衰减等超参，避免高学习率+大权重衰减导致训练震荡。
# 8. 如果连续几轮调整效果反倒严重变差，先回溯历史上最佳的情况，分析其特征与当前实验的差异，做对应调整。


# ### 任务要求
# 请基于上述信息：
# 1. 详细分析当前验证集性能瓶颈的核心原因，覆盖特征预处理、正则强度、模型容量、图结构、超参等维度；
# 2. 写明本次具体调优方案，明确模型架构、模型层数、预处理策略、超参数调整、正则配置等细节；
# 3. 简明扼要，400字以内，纯文本输出。

# 请开始分析并直接输出文本：
# """

train_history_analyse_template = """你是图神经网络模型训练与调优专家，专注图节点分类，目标持续拉高验证集ACC。

### 当前状态信息
数据集探查信息：
{data_inspect_result}

所有训练尝试：
{history_train_summary}

### 探索方向和思路
1. **优先排查特征预处理瓶颈**：原始稀疏高维特征易造成注意力失效、收敛困难。ACC低于0.6优先校验标准化+PCA/SVD降维；若预处理后效果下跌，可尝试关闭标准化、减弱特征正则。
2. **分层管控正则强度（区分特征/结构/路径正则）**
    特征：普通Dropout；图结构：DropEdge、DropNode；深度正则：DropPath。
    中小稠密数据集禁止三类正则高强度叠加，极易欠拟合压制注意力表达；仅保留单一种类轻量正则，推理禁用拓扑扰动TTA，仅可使用Dropout TTA。
3. **架构迭代优先级**：实验不足5轮优先横向对比模型（GATv2/MixHop/JKNet）；5轮以上聚焦细粒度结构与超参调优。
4. **图结构校验**：核对边无向化、自环配置、邻接归一化（对称归一/仅自环两种方案对照），拓扑是本数据集核心判别特征，不可随意扰动。
5. **模型容量与训练轮次匹配**：没有严重过拟合时可以提升层数、隐藏维度、注意力头数；epoch上限450，搭配早停。
6. **归一化、深度架构专项调优**
    归一化层：对比BatchNorm(BN)、LayerNorm(LN)效果，深层GATv2优先LN，浅层可尝试BN；
    深度连接：启用残差块缓解梯度消失，可堆叠3~4层深层GAT，搭配JK跳跃连接(Jumping Knowledge)聚合各层特征；
    深度正则：DropPath随机丢弃层路径做轻量化正则，需调低丢弃概率，DropPath丢弃率0.1以内，不和高强度DropEdge叠加。
7. **损失与优化器超参**：可选CE/标签平滑，严控学习率与权重衰减组合，高lr+大weight_decay易训练震荡；如果类别极度不平衡，逆频加权后导致验证集ACC下滑，优先保持无类别加权
8. **连续多轮精度下滑策略**：回溯历史最优实验，对比预处理、归一化、JK、DropPath、正则、lr等差异，回退有效配置再小幅迭代。

### 任务要求
1. 定位验证集ACC瓶颈，覆盖特征预处理、正则组合、归一化(BN/LN)、JK跳跃、DropPath正则、模型容量、图结构、超参全维度；
2. 输出落地调优方案，明确架构、层数、归一化选择、JK开关、DropPath丢弃率、预处理、正则、超参细节；
3. 全文400字以内，纯文本简洁输出。

请开始分析并直接输出文本：
"""


train_code_gen_template = """
你需要完成的任务为:
{task_description}

数据集信息为：
{dataset_description}

为此需要帮我完成一份完整的划分训练集验证集、构建模型、使用训练集训练模型，使用验证集验证ACC、对测试集进行预测并输出B1.csv到文件夹为{model_save_dir}下的代码

特别的，训练集/验证集划分方式为：
1. 加载 B1.npz 中预定义的 train_idx 和 test_idx 作为固定的训练集与测试集节点索引；
2. 验证集的构建方式：从 train_idx 中随机抽取 20% 的节点作为验证集（val_idx），剩余 80% 作为实际训练集（actual_train_idx）。抽取时必须使用种子 666 确保可复现；

可以参考的逻辑为：
def _reproduce_val_split(train_idx: np.ndarray) -> tuple:
    rng = np.random.RandomState(666)
    shuffled_idx = rng.permutation(train_idx)
    n_actual_train = math.ceil(len(shuffled_idx) * 0.8)
    actual_train_idx = shuffled_idx[:n_actual_train]
    val_idx = shuffled_idx[n_actual_train:]
    return actual_train_idx, val_idx

测试集输出文件命名为B1.csv，必须包含表头，且仅允许包含两列，例如：
test_idx,label
18,3
19,7
其中：test_idx：测试节点编号，必须与数据集保持一致。label：选手预测的类别编号，必须为合法整数类别编号。

为了便于调试，你可以比如每隔10个epoch打印一次验证集ACC，便于后续排查是否过拟合、欠拟合等等。
训练过程中，你需要print验证集上准确率，并且在每次新出现效果最好的模型时，立即保存效果最好的模型(该模型命名为best_model.pth，模型保存路径为{model_save_dir})，然后使用这个模型完成测试集推理，输出B1.csv到文件夹为{model_save_dir}下（如果这个文件存在，直接覆盖）。
同时每次新出现效果最好的模型时，还必须输出测试集每个类别的类别的softmax分数候选文件 B1-softmax.csv 和验证集候选文件 validation-softmax.csv以便于后续集成使用。

validation-softmax.csv和B1-softmax.csv格式要求：
- 两个文件表头统一为：test_idx,class_0,class_1,class_2,class_3,class_4,class_5,class_6,class_7
- validation-softmax.csv 的 test_idx 列存放验证集节点编号，B1-softmax.csv 的 test_idx 列存放测试集节点编号
- 每行包括idx以及对应的8个类别的softmax分数，用英文逗号连接，test_idx 必须为整数类型，softmax分数必须是浮点数，保留4位小数，类别列按编号升序排列

【索引对齐铁律 · 必须严格遵守】
保存CSV时，必须直接用 val_idx / test_idx 整数数组索引模型输出，保证行顺序与ID顺序一一对应；
禁止用布尔掩码(val_mask/test_mask)提取输出后再拼接打乱的索引数组，否则ID与概率会错位。
参考写法（核心只看这两行）：
    val_probs = F.softmax(out[val_idx], dim=1).cpu().numpy()    # 验证集：用 val_idx 直接取
    test_probs = F.softmax(out[test_idx], dim=1).cpu().numpy()  # 测试集：用 test_idx 直接取
训练时计算ACC可以继续使用布尔掩码，不影响准确率。


【测试集TTA增强 · 默认启用】
1. 仅测试集最终输出启用TTA（测试时增强），基于Dropout多次推理取平均，免费提升泛化性能；验证集始终使用纯eval模式，保证与训练过程口径一致。
2. 实现方式：训练全部结束、加载最优模型后，开启模型train模式启用Dropout，重复推理40-50次，对softmax概率取平均后生成最终测试集文件。
3. validation-softmax.csv 不做TTA，保持与训练验证ACC同口径，用于后续集成评估。

为了提升训练效率和效果，你只能使用一种模型(不考虑多模型融合)，并且注意准确率和效率（显存20GB以内）。你可以使用torch_geometric库构建模型(例如GAT、GATv2、MixHop和各种变体等)。
在模型准确率持续上升的情况下，即使过拟合程度有略微提升，也可以尝试加大一些epoch和early stopping的patience。如果都还没有发生过拟合，甚至比较欠拟合，更可以大胆的上调epoch和模型复杂度。初期epoch可以在100-200左右，后期可以epoch上调到300-500。patience可以设置大一些，比如30-50。

之前你做了一些尝试，有前景的方向为：
{optimize_direction}

当前代码【重点参考格式】:
{reference_code}

上一轮报错信息(如果有):
{last_error}

# Constraint
要完成这个任务，你需要分析瓶颈和提升方向，完成对当前代码的调整和优化，直接输出可执行的python代码，不要解释。尤其注意数据集导入和最后预测的代码的逻辑准确性，如果之前出错，注意不要重犯错误。
请开始输出：
"""


train_code_gen_edit_based_template = """
# Role
你是一个图神经网络算法代码修复与优化专家。你的任务是基于上一轮的报错信息或新的优化方向，对现有训练代码进行【最小化增量修改】。

# Goal
请根据以下输入信息，精准定位问题并完成代码修复/优化：
1. 若存在报错：优先修复该报错，并检查是否还有其他错误，确保代码可执行、不重犯同类错误。
2. 若存在优化方向：在保持代码可运行的前提下，落实该优化方向（如调整模型结构、模型复杂度、超参、损失函数等）。
3. 严禁破坏以下核心业务约束（即使报错与此相关，也应以修复而非删除的方式处理）：
   - 加载 B1.npz 中预定义的 train_idx 和 test_idx 作为固定的训练集与测试集节点索引；
   -  验证集的构建方式：从 train_idx 中随机抽取 20% 的节点作为验证集（val_idx），剩余 80% 作为实际训练集（actual_train_idx）。抽取时必须使用种子 666 确保可复现；

# 划分训练集验证集的参考逻辑不能修改
可以参考的逻辑为：
def _reproduce_val_split(train_idx: np.ndarray) -> tuple:
    rng = np.random.RandomState(666)
    shuffled_idx = rng.permutation(train_idx)
    n_actual_train = math.ceil(len(shuffled_idx) * 0.8)
    actual_train_idx = shuffled_idx[:n_actual_train]
    val_idx = shuffled_idx[n_actual_train:]
    return actual_train_idx, val_idx

测试集输出文件命名为B1.csv，必须包含表头，且仅允许包含两列，例如：
test_idx,label
18,3
19,7
其中：test_idx：测试节点编号，必须与数据集保持一致。label：选手预测的类别编号，必须为合法整数类别编号。

保存softmax的两个CSV时，必须直接用 val_idx / test_idx 整数数组索引模型输出，保证行顺序与ID顺序一一对应；
禁止用布尔掩码(val_mask/test_mask)提取输出后再拼接打乱的索引数组，否则ID与概率会错位。
参考写法（核心只看这两行）：
    val_probs = F.softmax(out[val_idx], dim=1).cpu().numpy()    # 验证集：用 val_idx 直接取
    test_probs = F.softmax(out[test_idx], dim=1).cpu().numpy()  # 测试集：用 test_idx 直接取
训练时计算ACC可以继续使用布尔掩码，不影响准确率。对于验证集不要使用TTA，对于测试集启用TTA。

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


def dual_print(*args, log_file="cls_output.log", mode="a", encoding="utf-8", **kwargs):
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
    csv_path='./model/interface_cls/B1.csv',
    data_path='/data/coding/line3/dataset_b/cls_data/B1.npz',
    num_classes: int = global_class_number
) -> bool:
    """
    验证 B1.csv 提交文件格式是否正确。

    Args:
        csv_path:    提交文件路径，如 "./model/interface_cls/B1.csv"
        data_path:   原始数据集路径，如 "/data/coding/line3/dataset_b/cls_data/B1.npz"
        num_classes: 类别总数（0 ~ num_classes-1）

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
    expected_columns = ["test_idx", "label"]
    if list(df.columns) != expected_columns:
        errors.append(
            f"表头不正确。期望 {expected_columns}，实际为 {list(df.columns)}"
        )

    if df.shape[1] != 2:
        errors.append(f"列数应为 2，实际为 {df.shape[1]}")

    # ========== 4. 加载数据集获取标准 test_idx ==========
    try:
        data = np.load(data_path, allow_pickle=True)
        gt_test_idx = np.sort(data["test_idx"])
    except Exception as e:
        dual_print(f"❌ 数据集加载失败: {e}")
        return False

    # ========== 5. 检查 test_idx ==========
    if "test_idx" in df.columns:
        submitted_idx = np.sort(df["test_idx"].values)

        # 5.1 数量一致
        if len(submitted_idx) != len(gt_test_idx):
            errors.append(
                f"test_idx 数量不一致。期望 {len(gt_test_idx)}，实际 {len(submitted_idx)}"
            )
        else:
            # 5.2 内容完全匹配
            if not np.array_equal(submitted_idx, gt_test_idx):
                missing = np.setdiff1d(gt_test_idx, submitted_idx)
                extra = np.setdiff1d(submitted_idx, gt_test_idx)
                if len(missing) > 0:
                    errors.append(f"缺少 test_idx: {missing[:10]}{'...' if len(missing) > 10 else ''}")
                if len(extra) > 0:
                    errors.append(f"多余 test_idx: {extra[:10]}{'...' if len(extra) > 10 else ''}")

        # 5.3 test_idx 应为整数
        if not pd.api.types.is_integer_dtype(df["test_idx"]):
            errors.append(f"test_idx 必须为整数类型，实际为 {df['test_idx'].dtype}")

        # 5.4 不能有重复
        dup_count = df["test_idx"].duplicated().sum()
        if dup_count > 0:
            errors.append(f"test_idx 存在 {dup_count} 个重复值")

    # ========== 6. 检查 label ==========
    if "label" in df.columns:
        # 6.1 整数类型
        if not pd.api.types.is_integer_dtype(df["label"]):
            errors.append(f"label 必须为整数类型，实际为 {df['label'].dtype}")
        else:
            # 6.2 合法范围 [0, num_classes - 1]
            min_label = df["label"].min()
            max_label = df["label"].max()
            if min_label < 0 or max_label >= num_classes:
                errors.append(
                    f"label 超出合法范围 [0, {num_classes - 1}]，"
                    f"实际范围 [{min_label}, {max_label}]"
                )

        # 6.3 不能有空值
        null_count = df["label"].isnull().sum()
        if null_count > 0:
            errors.append(f"label 存在 {null_count} 个空值")

    # ========== 输出结果 ==========
    if errors:
        dual_print("❌ 验证未通过，发现以下问题：")
        for i, err in enumerate(errors, 1):
            dual_print(f"   {i}. {err}")
        return False
    else:
        dual_print("✅ 验证通过！B1.csv 格式完全正确。")
        dual_print(f"   - 样本数: {len(df)}")
        dual_print(f"   - test_idx 范围: [{df['test_idx'].min()}, {df['test_idx'].max()}]")
        dual_print(f"   - label 分布: {dict(df['label'].value_counts().sort_index())}")
        return True
    

def validate_softmax_csv(
    csv_path: str,
    data_path: str = '/data/coding/line3/dataset_b/cls_data/B1.npz',
    num_classes: int = global_class_number,
    is_validation: bool = False
) -> bool:
    """
    验证 B1-softmax.csv 或 validation-softmax.csv 格式是否正确。

    Args:
        csv_path:     softmax文件路径
        data_path:    原始数据集路径 (用于获取标准idx)
        num_classes:  类别总数（0 ~ num_classes-1），表头应为 class_0 到 class_{num_classes-1}
        is_validation: True表示校验validation-softmax.csv(使用train_idx中的val_idx),
                       False表示校验B1-softmax.csv(使用test_idx)

    Returns:
        True 表示验证通过，False 表示存在错误
    """
    errors = []
    file_type = "validation-softmax" if is_validation else "B1-softmax"

    # ========== 1. 文件是否存在 ==========
    if not os.path.exists(csv_path):
        dual_print(f"❌ [{file_type}] 文件不存在: {csv_path}")
        return False

    # ========== 2. 读取 CSV ==========
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        dual_print(f"❌ [{file_type}] CSV 读取失败: {e}")
        return False

    # ========== 3. 检查表头和列数 ==========
    expected_columns = ["test_idx"] + [f"class_{i}" for i in range(num_classes)]
    if list(df.columns) != expected_columns:
        errors.append(
            f"表头不正确。期望 {expected_columns}，实际为 {list(df.columns)}"
        )

    expected_col_count = 1 + num_classes
    if df.shape[1] != expected_col_count:
        errors.append(f"列数应为 {expected_col_count}，实际为 {df.shape[1]}")

    # ========== 4. 加载数据集获取标准 idx ==========
    try:
        data = np.load(data_path, allow_pickle=True)
        if is_validation:
            # 复现验证集划分逻辑
            train_idx = data["train_idx"]
            rng = np.random.RandomState(666)
            shuffled_idx = rng.permutation(train_idx)
            n_actual_train = math.ceil(len(shuffled_idx) * 0.8)
            gt_idx = np.sort(shuffled_idx[n_actual_train:])
        else:
            gt_idx = np.sort(data["test_idx"])
    except Exception as e:
        dual_print(f"❌ [{file_type}] 数据集加载失败: {e}")
        return False

    # ========== 5. 检查 test_idx ==========
    if "test_idx" in df.columns:
        submitted_idx = np.sort(df["test_idx"].values)

        # 5.1 数量一致
        if len(submitted_idx) != len(gt_idx):
            errors.append(
                f"idx 数量不一致。期望 {len(gt_idx)}，实际 {len(submitted_idx)}"
            )
        else:
            # 5.2 内容完全匹配
            if not np.array_equal(submitted_idx, gt_idx):
                missing = np.setdiff1d(gt_idx, submitted_idx)
                extra = np.setdiff1d(submitted_idx, gt_idx)
                if len(missing) > 0:
                    errors.append(f"缺少 idx: {missing[:10]}{'...' if len(missing) > 10 else ''}")
                if len(extra) > 0:
                    errors.append(f"多余 idx: {extra[:10]}{'...' if len(extra) > 10 else ''}")

        # 5.3 test_idx 应为整数
        if not pd.api.types.is_integer_dtype(df["test_idx"]):
            errors.append(f"test_idx 必须为整数类型，实际为 {df['test_idx'].dtype}")

        # 5.4 不能有重复
        dup_count = df["test_idx"].duplicated().sum()
        if dup_count > 0:
            errors.append(f"test_idx 存在 {dup_count} 个重复值")

    # ========== 6. 检查 softmax 分数列 ==========
    class_cols = [f"class_{i}" for i in range(num_classes)]
    existing_class_cols = [c for c in class_cols if c in df.columns]

    if existing_class_cols:
        softmax_values = df[existing_class_cols].values

        # 6.1 必须为浮点类型
        if not np.issubdtype(softmax_values.dtype, np.floating):
            errors.append(f"softmax分数必须为浮点数类型，实际为 {softmax_values.dtype}")

        # 6.2 不能有空值
        null_count = df[existing_class_cols].isnull().sum().sum()
        if null_count > 0:
            errors.append(f"softmax分数存在 {null_count} 个空值")

        # 6.3 值域应在 [0, 1] 之间
        valid_mask = ~np.isnan(softmax_values)
        if valid_mask.any():
            min_val = np.nanmin(softmax_values)
            max_val = np.nanmax(softmax_values)
            if min_val < 0 or max_val > 1.0:
                errors.append(
                    f"softmax分数超出合法范围 [0, 1]，"
                    f"实际范围 [{min_val:.6f}, {max_val:.6f}]"
                )

        # 6.4 每行softmax之和应约等于1.0 (容差1e-3)
        row_sums = np.nansum(softmax_values, axis=1)
        invalid_sum_mask = np.abs(row_sums - 1.0) > 0.1
        invalid_sum_count = int(invalid_sum_mask.sum())
        if invalid_sum_count > 0:
            worst_idx = np.argmax(np.abs(row_sums - 1.0))
            errors.append(
                f"有 {invalid_sum_count} 行softmax分数之和不等于1.0，"
                f"最大偏差行(test_idx={df.iloc[worst_idx]['test_idx']}): sum={row_sums[worst_idx]:.6f}"
            )

        # 6.5 检查小数位数（最多4位）
        # 将浮点数转为字符串检查精度
        sample_vals = softmax_values[valid_mask][:100]  # 抽样检查
        for val in sample_vals:
            s = f"{val:.10f}".rstrip('0')
            decimal_part = s.split('.')[-1] if '.' in s else ''
            if len(decimal_part) > 4:
                errors.append(f"softmax分数精度超过4位小数，示例: {val}")
                break

    # ========== 输出结果 ==========
    if errors:
        dual_print(f"❌ [{file_type}] 验证未通过，发现以下问题：")
        for i, err in enumerate(errors, 1):
            dual_print(f"   {i}. {err}")
        return False
    else:
        dual_print(f"✅ [{file_type}] 验证通过！格式完全正确。")
        dual_print(f"   - 样本数: {len(df)}")
        dual_print(f"   - idx 范围: [{df['test_idx'].min()}, {df['test_idx'].max()}]")
        dual_print(f"   - 类别数: {num_classes}")
        if existing_class_cols:
            avg_probs = df[existing_class_cols].mean(axis=0)
            # 修正：使用字符串列名访问 Series 的值
            prob_strs = [f"class_{i}: {avg_probs[f'class_{i}']:.4f}" for i in range(num_classes)]
            dual_print(f"   - 各类平均概率: {{{', '.join(prob_strs)}}}")
        return True


def run_smoking_test(data_path, script_path, submission_csv_path,num_classes):
    # script_path = './interface_gen_cls_smoke.py'  # ← 替换为你的冒烟测试脚本路径
    # submission_csv_path = './model/interface_cls/B1.csv'
    # data_path='/data/coding/line3/dataset/cls_data/B1.npz'
    # num_classes=10
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
                    return False,'❌ 训练过程出错，代码有误' + '\n报错信息为：' + result.stderr + "\n训练日志为："+train_log + "\n" + data_meta_info
                else:
                    return False,'❌ 训练过程出错，代码有误' + '\n报错信息为：' + result.stderr + "\n训练日志为："+train_log
        
        # 判断运行是否正常，用flash模型判断即可
        smoking_test_judge_prompt = smoking_test_judge_template.format(train_log=train_log)
        judge_result = call_llm(smoking_test_judge_prompt)
        if judge_result == "正常":
            dual_print("✅ 冒烟测试运行通过")
        else:
            dual_print("❌ 冒烟测试运行未通过")
            return False, "❌ 冒烟测试运行未通过，训练过程日志为"+"\n"+train_log
        if os.path.exists(os.path.join(submission_csv_path,'B1.csv')):
            dual_print(f"✅ 测试集结果已输出: {os.path.join(submission_csv_path,'B1.csv')}")
        else:
            dual_print("❌ 测试集结果未输出，异常")
            return False, f"❌ 测试集结果未输出在规定路径{os.path.join(submission_csv_path,'B1.csv')}，异常"
        
        test_ok = validate_submission(csv_path=os.path.join(submission_csv_path,'B1.csv'), data_path=fixed_data_path, num_classes=num_classes)
        if test_ok:
            dual_print("✅ 测试集结果格式验证通过")
        else:
            return False, "❌ 测试集结果格式验证未通过，异常"
        
        err_hint = """validation-softmax.csv和B1-softmax.csv格式要求：
        - 两个文件表头统一为：test_idx,class_0,class_1,class_2,class_3,class_4,class_5,class_6,class_7
        - validation-softmax.csv 的 test_idx 列存放验证集节点编号，B1-softmax.csv 的 test_idx 列存放测试集节点编号
        - 每行包括idx以及对应的8个类别的softmax分数，用英文逗号连接，test_idx 必须为整数类型，softmax分数必须是浮点数，保留4位小数，类别列按编号升序排列

        【索引对齐铁律 · 必须严格遵守】
        保存CSV时，必须直接用 val_idx / test_idx 整数数组索引模型输出，保证行顺序与ID顺序一一对应；
        禁止用布尔掩码(val_mask/test_mask)提取输出后再拼接打乱的索引数组，否则ID与概率会错位。
        参考写法（核心只看这两行）：
            val_probs = F.softmax(out[val_idx], dim=1).cpu().numpy()    # 验证集：用 val_idx 直接取
            test_probs = F.softmax(out[test_idx], dim=1).cpu().numpy()  # 测试集：用 test_idx 直接取
        训练时计算ACC可以继续使用布尔掩码，不影响准确率。
        """
        valid_softmax_ok = validate_softmax_csv(
                csv_path=os.path.join(submission_csv_path,'validation-softmax.csv'),
                data_path=fixed_data_path,
                num_classes=global_class_number,
                is_validation=True
            )
        if valid_softmax_ok:
            dual_print("✅ 验证集softmax结果格式验证通过")
        else:
            return False, "❌ 验证集softmax结果格式验证通不过" + err_hint
        
        test_softmax_ok = validate_softmax_csv(
                csv_path=os.path.join(submission_csv_path,'B1-softmax.csv'),
                data_path=fixed_data_path,
                num_classes=global_class_number,
                is_validation=False
            )
        if test_softmax_ok:
            dual_print("✅ 测试集softmax结果格式验证通过")
        else:
            return False, "❌ 测试集softmax结果格式验证通不过" + err_hint
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
    1. 判断训练是否正常（在epoch数超过100的情况下看ACC是否正常）

    如果当前还没有完成100个epoch，输出 wait  (还需要观察更多输出)
    如果已经训练到了超过100个epoch，检查当前ACC是否大于0.6，如果ACC还很低，并且从增长趋势上看不出有明显增长趋势，才输出 bad
    如果当前验证集ACC超过0.9，100%的可能为指标计算出错和代码有bug，输出bad
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
        kill_msg = "模型训练没有效果，请调整(如果准确率超过0.9，为准确率计算过程有错误，检查是否数据泄漏；准确率太低可能是存在bug)，训练日志为："+'\n'.join(stdout_lines)
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
    
    if not validate_submission(csv_path=os.path.join(model_save_dir, "B1.csv"), data_path=fixed_data_path, num_classes=global_class_number):
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
    
#### 模型融合 ####
import warnings
warnings.filterwarnings("ignore")
import os
import math
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import csr_matrix, diags, eye
from sklearn.semi_supervised import LabelSpreading


# ========================
# 工具函数
# ========================

def _reproduce_val_split(train_idx: np.ndarray) -> tuple:
    rng = np.random.RandomState(666)
    shuffled_idx = rng.permutation(train_idx)
    n_actual_train = math.ceil(len(shuffled_idx) * 0.8)
    actual_train_idx = shuffled_idx[:n_actual_train]
    val_idx = shuffled_idx[n_actual_train:]
    return actual_train_idx, val_idx


def build_normalized_adj(adj, num_nodes):
    """构建对称归一化邻接矩阵 D^{-1/2} A D^{-1/2}"""
    A = adj.copy()
    if (A - A.T).nnz > 0:
        A = (A + A.T).tocsr()
        A[A > 1] = 1
    A = A + eye(num_nodes, format='csr')
    deg = np.array(A.sum(axis=1)).flatten()
    deg_inv_sqrt = np.zeros_like(deg, dtype=np.float64)
    mask = deg > 0
    deg_inv_sqrt[mask] = 1.0 / np.sqrt(deg[mask])
    D_inv_sqrt = diags(deg_inv_sqrt, format='csr')
    S = D_inv_sqrt @ A @ D_inv_sqrt
    return S, A


def correct_and_smooth(base_probs, S_norm, train_mask, labels_onehot,
                       num_classes, correct_alpha=0.5, smooth_alpha=0.5,
                       num_iters=50):
    """Correct and Smooth 后处理"""
    num_nodes = base_probs.shape[0]
    I = eye(num_nodes, format='csr')

    # Stage 1: Correct
    residuals = np.zeros((num_nodes, num_classes), dtype=np.float64)
    residuals[train_mask] = labels_onehot[train_mask] - base_probs[train_mask]

    S_correct = correct_alpha * S_norm + (1 - correct_alpha) * I
    Z = residuals.copy()
    for _ in range(num_iters):
        Z_new = S_correct @ Z
        Z_new[train_mask] = residuals[train_mask]
        if np.max(np.abs(Z_new - Z)) < 1e-6:
            Z = Z_new
            break
        Z = Z_new

    corrected = base_probs + Z
    corrected = np.clip(corrected, 1e-10, 1.0)
    corrected /= corrected.sum(axis=1, keepdims=True)

    # Stage 2: Smooth
    S_smooth = smooth_alpha * S_norm + (1 - smooth_alpha) * I
    Y = corrected.copy()
    for _ in range(num_iters):
        Y_new = S_smooth @ Y
        Y_new[train_mask] = labels_onehot[train_mask]
        Y_new = np.clip(Y_new, 1e-10, 1.0)
        Y_new /= Y_new.sum(axis=1, keepdims=True)
        if np.max(np.abs(Y_new - Y)) < 1e-6:
            Y = Y_new
            break
        Y = Y_new

    return Y


def label_propagation_postprocess(base_probs, S_norm, train_mask, labels_onehot,
                                  num_classes, alpha=0.5, num_iters=100):
    """标准 Label Propagation 后处理"""
    num_nodes = base_probs.shape[0]
    I = eye(num_nodes, format='csr')

    Y = base_probs.copy()
    Y[train_mask] = labels_onehot[train_mask]

    S_lp = alpha * S_norm + (1 - alpha) * I
    for _ in range(num_iters):
        Y_new = S_lp @ Y
        Y_new[train_mask] = labels_onehot[train_mask]
        Y_new = np.clip(Y_new, 1e-10, 1.0)
        Y_new /= Y_new.sum(axis=1, keepdims=True)
        if np.max(np.abs(Y_new - Y)) < 1e-6:
            Y = Y_new
            break
        Y = Y_new

    return Y


def feature_based_label_spreading(features, train_mask_arr, labels_arr,
                                  num_classes, num_nodes, n_neighbors=15, alpha=0.3):
    """基于特征的 Label Spreading"""
    ls_labels = np.full(num_nodes, -1, dtype=int)
    ls_labels[train_mask_arr] = labels_arr[train_mask_arr]

    ls = LabelSpreading(kernel='knn', n_neighbors=n_neighbors,
                        alpha=alpha, max_iter=100)
    ls.fit(features, ls_labels)
    ls_probs = ls.label_distributions_

    # 对齐列顺序
    ls_probs_aligned = np.full((num_nodes, num_classes), 1.0 / num_classes)
    for i, cls in enumerate(ls.classes_):
        ls_probs_aligned[:, int(cls)] = ls_probs[:, i]

    return ls_probs_aligned


def self_training_pseudo_labels(base_probs, train_mask, labels, num_classes,
                                confidence_threshold=0.9, max_rounds=5):
    """自训练伪标签"""
    expanded_mask = train_mask.copy()
    all_labels = np.zeros((len(labels), num_classes), dtype=np.float64)
    all_labels[train_mask] = np.eye(num_classes)[labels[train_mask]]

    for round_i in range(max_rounds):
        preds = np.argmax(base_probs, axis=1)
        confs = np.max(base_probs, axis=1)

        unlabeled_mask = ~expanded_mask
        high_conf = unlabeled_mask & (confs > confidence_threshold)
        new_nodes = np.where(high_conf)[0]
        if len(new_nodes) == 0:
            break

        expanded_mask[new_nodes] = True
        all_labels[new_nodes] = base_probs[new_nodes]

    return expanded_mask, all_labels


# ========================
# 主集成函数（修复版）
# ========================

def ensemble_with_postprocess_v3(
    model_save_dir: str,
    data_path: str,
    num_classes: int = global_class_number,
    max_ges_iter: int = 50
):
    """
    修复版 v3：
    - 验证集评估时：后处理只用 actual_train 标签（val 标签不可见）
    - 测试集输出时：后处理用 actual_train + val 标签（全部已知信息）
    """
    global global_best_merge_acc
    print(f"\n{'='*70}")
    print("🔗 多模型集成 + 多后处理方案 (修复版 v3 - 标签隔离)")
    print(f"{'='*70}")

    # ========== 1. 收集所有有效实验 ==========
    val_softmax_files = sorted(glob.glob(
        os.path.join(model_save_dir, "**/validation-softmax.csv"), recursive=True))
    test_softmax_files = sorted(glob.glob(
        os.path.join(model_save_dir, "**/B1-softmax.csv"), recursive=True))

    val_dirs = {os.path.dirname(f) for f in val_softmax_files}
    test_dirs = {os.path.dirname(f) for f in test_softmax_files}
    valid_dirs = sorted(val_dirs & test_dirs)

    if not valid_dirs:
        print("❌ 未找到有效的实验目录")
        return

    class_cols = [f"class_{i}" for i in range(num_classes)]
    val_probs_list, test_probs_list, exp_names = [], [], []
    full_probs_list = []
    ref_test_idx, ref_val_idx = None, None

    for exp_dir in valid_dirs:
        exp_name = os.path.relpath(exp_dir, model_save_dir)
        try:
            val_df = pd.read_csv(os.path.join(exp_dir, "validation-softmax.csv"))
            test_df = pd.read_csv(os.path.join(exp_dir, "B1-softmax.csv"))

            val_df = val_df.sort_values("test_idx").reset_index(drop=True)
            test_df = test_df.sort_values("test_idx").reset_index(drop=True)

            val_probs_list.append(val_df[class_cols].values.astype(np.float64))
            test_probs_list.append(test_df[class_cols].values.astype(np.float64))
            exp_names.append(exp_name)

            # 尝试加载全图预测
            full_path = os.path.join(exp_dir, "full-softmax.csv")
            if os.path.exists(full_path):
                full_df = pd.read_csv(full_path).sort_values("test_idx").reset_index(drop=True)
                full_probs_list.append(full_df[class_cols].values.astype(np.float64))
            else:
                full_probs_list.append(None)

            if ref_test_idx is None:
                ref_test_idx = test_df["test_idx"].values
                ref_val_idx = val_df["test_idx"].values

            print(f"   ✅ {exp_name}")
        except Exception as e:
            print(f"   ❌ {exp_name}: {e}")

    n_models = len(val_probs_list)
    has_full_probs = any(f is not None for f in full_probs_list)
    print(f"\n模型数量: {n_models}, 有全图预测: {has_full_probs}")

    # ========== 2. 加载数据 ==========
    raw = np.load(os.path.join(data_path, "B1.npz"), allow_pickle=True)
    train_idx = raw["train_idx"]
    test_idx = raw["test_idx"]
    labels = raw["labels"]
    adj = csr_matrix(
        (raw["adj_data"], raw["adj_indices"], raw["adj_indptr"]),
        shape=tuple(raw["adj_shape"]))
    features = csr_matrix(
        (raw["attr_data"], raw["attr_indices"], raw["attr_indptr"]),
        shape=tuple(raw["attr_shape"]))

    actual_train_idx, val_idx = _reproduce_val_split(train_idx)
    num_nodes = adj.shape[0]

    # 构建归一化邻接矩阵
    S_norm, A_sym = build_normalized_adj(adj, num_nodes)

    features_dense = features.toarray().astype(np.float32)

    # ==============================================================
    # 【核心修复】定义两个不同的 mask，严格隔离标签可见范围
    # ==============================================================

    # mask_val: 只包含 actual_train（用于验证集评估，val 标签不可见）
    mask_val = np.zeros(num_nodes, dtype=bool)
    mask_val[actual_train_idx.astype(int)] = True

    # mask_test: 包含 actual_train + val（用于测试集输出，所有已知标签可用）
    mask_test = np.zeros(num_nodes, dtype=bool)
    mask_test[actual_train_idx.astype(int)] = True
    mask_test[val_idx.astype(int)] = True

    # onehot 标签矩阵
    labels_onehot = np.zeros((num_nodes, num_classes), dtype=np.float64)
    for idx in actual_train_idx:
        labels_onehot[int(idx), int(labels[idx])] = 1.0
    for idx in val_idx:
        labels_onehot[int(idx), int(labels[idx])] = 1.0

    # 验证集真实标签（仅用于评估 ACC，不参与后处理拟合）
    val_labels = np.array([int(labels[idx]) for idx in ref_val_idx])

    print(f"\n标签隔离信息:")
    print(f"   mask_val (验证集后处理): {mask_val.sum()} 个已知标签 (仅 actual_train)")
    print(f"   mask_test (测试集后处理): {mask_test.sum()} 个已知标签 (actual_train + val)")

    # ========== 3. 构建全图概率矩阵 ==========
    def get_full_probs(m_idx, mode='val'):
        """
        构建全图概率矩阵
        mode='val':  训练节点用 soft label (val 标签不可见，所以 val 节点用 GNN 预测)
        mode='test': 训练节点+验证节点都用 soft label (所有已知标签可见)
        """
        if full_probs_list[m_idx] is not None:
            Y = full_probs_list[m_idx].copy()
        else:
            Y = np.full((num_nodes, num_classes), 1.0 / num_classes, dtype=np.float64)

            # 训练节点：用 soft label
            for idx in actual_train_idx:
                idx = int(idx)
                Y[idx] = np.full(num_classes, 0.1 / num_classes)
                Y[idx, int(labels[idx])] = 0.9 + 0.1 / num_classes

            if mode == 'test':
                # 测试模式：验证节点也用 soft label
                for idx in val_idx:
                    idx = int(idx)
                    Y[idx] = np.full(num_classes, 0.1 / num_classes)
                    Y[idx, int(labels[idx])] = 0.9 + 0.1 / num_classes

            # 用邻居投票填充未覆盖的节点
            for idx in range(num_nodes):
                if idx not in ref_test_idx and idx not in ref_val_idx:
                    is_train = mask_val[idx] if mode == 'val' else mask_test[idx]
                    if not is_train and full_probs_list[m_idx] is None:
                        neighbors = A_sym[idx].nonzero()[1]
                        if len(neighbors) > 0:
                            neighbor_probs = Y[neighbors].mean(axis=0)
                            if neighbor_probs.sum() > 0:
                                Y[idx] = neighbor_probs / neighbor_probs.sum()

        # 填入 GNN 预测（覆盖上面的初始化）
        for i, idx in enumerate(ref_val_idx):
            Y[int(idx)] = val_probs_list[m_idx][i]
        for i, idx in enumerate(ref_test_idx):
            Y[int(idx)] = test_probs_list[m_idx][i]

        return Y

    # ========== 4. GES 搜索 ==========
    def run_ges_search(probs_list, labels_gt, names, max_iter):
        n = len(probs_list)
        single_accs = [np.mean(np.argmax(p, axis=1) == labels_gt) for p in probs_list]
        best_single_idx = int(np.argmax(single_accs))

        selected_indices = [best_single_idx]
        selected_weights = [1.0]
        best_fused_acc = single_accs[best_single_idx]
        no_improve = 0

        for iteration in range(1, max_iter + 1):
            current_best = best_fused_acc
            current_candidate = None
            candidates = []

            # 添加新模型
            for idx in range(n):
                if idx not in selected_indices:
                    candidates.append((selected_indices + [idx],
                                       selected_weights + [1.0], f"+{names[idx]}"))

            # 调整权重
            for w_idx in range(len(selected_weights)):
                for delta in [-0.2, -0.1, 0.1, 0.2]:
                    new_w = max(0.05, selected_weights[w_idx] + delta)
                    tw = list(selected_weights)
                    tw[w_idx] = new_w
                    candidates.append((list(selected_indices), tw, f"w{delta:+.1f}"))

            # 移除模型
            if len(selected_indices) > 1:
                for rem_pos in range(len(selected_indices)):
                    ti = [j for p, j in enumerate(selected_indices) if p != rem_pos]
                    tw = [w for p, w in enumerate(selected_weights) if p != rem_pos]
                    candidates.append((ti, tw, f"-{names[selected_indices[rem_pos]]}"))

            for ti, tw, desc in candidates:
                w = np.array(tw, dtype=np.float64)
                w /= w.sum()
                fused = np.average([probs_list[i] for i in ti], axis=0, weights=w)
                acc = np.mean(np.argmax(fused, axis=1) == labels_gt)
                if acc > current_best:
                    current_best = acc
                    current_candidate = (ti, tw, desc)

            if current_candidate and current_best > best_fused_acc:
                selected_indices, selected_weights, _ = current_candidate
                best_fused_acc = current_best
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= 15:
                    break

        return best_fused_acc, selected_indices, selected_weights, single_accs

    # ========== 5. 定义后处理方案（接收 mask 参数）==========

    def method_base(Y_full, mask, onehot):
        return Y_full

    def method_cs(Y_full, mask, onehot):
        return correct_and_smooth(Y_full, S_norm, mask, onehot,
                                  num_classes, correct_alpha=0.5, smooth_alpha=0.5)

    def method_lp(Y_full, mask, onehot):
        return label_propagation_postprocess(Y_full, S_norm, mask, onehot,
                                             num_classes, alpha=0.7)

    def method_fls(Y_full, mask, onehot):
        return feature_based_label_spreading(features_dense, mask, labels,
                                             num_classes, num_nodes, n_neighbors=15, alpha=0.3)

    def method_self_train_cs(Y_full, mask, onehot):
        expanded_mask, expanded_onehot = self_training_pseudo_labels(
            Y_full, mask, labels, num_classes, confidence_threshold=0.85, max_rounds=5)
        # 合并扩展标签和真实标签 onehot
        merged_onehot = onehot.copy()
        merged_onehot[expanded_mask] = expanded_onehot[expanded_mask]
        return correct_and_smooth(Y_full, S_norm, expanded_mask, merged_onehot,
                                  num_classes, correct_alpha=0.5, smooth_alpha=0.5)

    def method_hybrid(Y_full, mask, onehot):
        lp_result = label_propagation_postprocess(Y_full, S_norm, mask, onehot,
                                                  num_classes, alpha=0.5)
        fls_result = feature_based_label_spreading(features_dense, mask, labels,
                                                   num_classes, num_nodes, n_neighbors=20, alpha=0.2)
        return 0.5 * lp_result + 0.5 * fls_result

    def method_multiscale_lp(Y_full, mask, onehot):
        results = []
        for alpha in [0.3, 0.5, 0.7, 0.9]:
            r = label_propagation_postprocess(Y_full, S_norm, mask, onehot,
                                              num_classes, alpha=alpha, num_iters=50)
            results.append(r)
        return np.mean(results, axis=0)

    def method_cs_tuned(Y_full, mask, onehot):
        best_acc = 0
        best_result = Y_full
        for ca in [0.3, 0.5, 0.7]:
            for sa in [0.3, 0.5, 0.7]:
                r = correct_and_smooth(Y_full, S_norm, mask, onehot,
                                       num_classes, correct_alpha=ca, smooth_alpha=sa,
                                       num_iters=30)
                # 用 val 评估（此时 mask 是 mask_val，val 标签不可见，公平）
                val_r = np.array([r[int(idx)] for idx in ref_val_idx])
                acc = np.mean(np.argmax(val_r, axis=1) == val_labels)
                if acc > best_acc:
                    best_acc = acc
                    best_result = r
        return best_result

    def method_dist_align_tuned(Y_full, mask, onehot):
        """分布对齐 (先验校准) 后处理，基于验证集自适应搜索最佳 alpha"""
        # 利用当前可见的真实标签分布作为先验
        train_counts = onehot[mask].sum(axis=0) + 1e-8
        train_prior = train_counts / train_counts.sum()
        pred_prior = Y_full.mean(axis=0) + 1e-8
        
        best_result = Y_full.copy()
        # 初始 base acc
        val_r_base = np.array([best_result[int(idx)] for idx in ref_val_idx])
        best_acc = np.mean(np.argmax(val_r_base, axis=1) == val_labels)
        
        for alpha in np.arange(0.1, 1.2, 0.1):
            adj_factor = (train_prior / pred_prior) ** alpha
            Y_adj = Y_full * adj_factor
            Y_adj /= Y_adj.sum(axis=1, keepdims=True)
            
            val_r = np.array([Y_adj[int(idx)] for idx in ref_val_idx])
            acc = np.mean(np.argmax(val_r, axis=1) == val_labels)
            
            if acc > best_acc:
                best_acc = acc
                best_result = Y_adj
                
        return best_result

    methods = {
        "Base (原始)": method_base,
        "DistAlign Tuned": method_dist_align_tuned,
        #"Correct&Smooth": method_cs,
        #"LP (图结构)": method_lp,
        #"Feature-LS": method_fls,
        "SelfTrain+C&S": method_self_train_cs,
        #"Hybrid LP+FLS": method_hybrid,
        #"MultiScale-LP": method_multiscale_lp,
        #"C&S Tuned": method_cs_tuned,
    }

    # ================================================================
    # 6. 第一轮：验证集评估（标签隔离，val 标签不可见）
    #    用 mask_val 做后处理，评估验证集 ACC，选出最佳方案
    # ================================================================
    print(f"\n{'─'*60}")
    print("📊 Step 1: 验证集评估 (mask_val: 仅 actual_train 标签可见)")
    print(f"{'─'*60}")

    # 基础集成（无后处理）
    base_acc, base_sel, base_w, base_single = run_ges_search(
        val_probs_list, val_labels, exp_names, max_ges_iter)
    print(f"\n基础集成 Val ACC: {base_acc:.4f}")
    for idx, w in zip(base_sel, np.array(base_w) / sum(base_w)):
        print(f"   {exp_names[idx]}: w={w:.3f} (single={base_single[idx]:.4f})")

    # 对比所有后处理方案
    results_val = {}

    for method_name, method_func in methods.items():
        print(f"\n🌀 {method_name}")

        val_post_list = []
        for m_idx in range(n_models):
            # ============================================================
            # 【关键】验证集后处理使用 mask_val（val 标签不可见）
            # ============================================================
            Y_full = get_full_probs(m_idx, mode='val')
            Y_post = method_func(Y_full, mask_val, labels_onehot)

            # 提取验证集概率
            val_post = np.array([Y_post[int(idx)] for idx in ref_val_idx])
            val_post_list.append(val_post)

            single_acc = np.mean(np.argmax(val_post, axis=1) == val_labels)
            print(f"   {exp_names[m_idx]}: {single_acc:.4f}")

        # GES 集成（在验证集上）
        best_acc, sel_idx, sel_w, single_accs = run_ges_search(
            val_post_list, val_labels, exp_names, max_ges_iter)

        print(f"   => 集成 Val ACC: {best_acc:.4f} (选中 {len(sel_idx)} 个模型)")

        results_val[method_name] = {
            "acc": best_acc,
            "sel_idx": sel_idx,
            "sel_w": sel_w,
            "single_accs": single_accs
        }

    # 选出验证集最佳方案
    best_method = max(results_val, key=lambda k: results_val[k]["acc"])
    best_val_res = results_val[best_method]

    print(f"\n{'='*70}")
    print(f"🏆 验证集最佳方案: {best_method} (Val ACC: {best_val_res['acc']:.4f})")
    print(f"{'='*70}")

    # 打印全部方案对比
    print(f"\n{'─'*60}")
    print("📊 验证集全部方案对比 (公平评估，val 标签不可见):")
    print(f"{'─'*60}")
    for name, res in sorted(results_val.items(), key=lambda x: -x[1]["acc"]):
        print(f"   {name:25s}: Val ACC = {res['acc']:.4f}")

    # ================================================================
    # 7. 第二轮：测试集输出（所有已知标签都可用）
    #    用选出的最佳方案 + mask_test 重新做后处理，生成最终提交
    # ================================================================
    print(f"\n{'─'*60}")
    print(f"📊 Step 2: 测试集输出 (mask_test: actual_train + val 标签全部可见)")
    print(f"   使用方案: {best_method}")
    print(f"{'─'*60}")

    best_method_func = methods[best_method]
    final_sel = best_val_res["sel_idx"]
    final_w = np.array(best_val_res["sel_w"], dtype=np.float64)
    final_w /= final_w.sum()

    test_post_list = []
    for m_idx in final_sel:
        # ============================================================
        # 【关键】测试集后处理使用 mask_test（所有已知标签都可用）
        # ============================================================
        Y_full = get_full_probs(m_idx, mode='test')
        Y_post = best_method_func(Y_full, mask_test, labels_onehot)

        # 提取测试集概率
        test_post = np.array([Y_post[int(idx)] for idx in ref_test_idx])
        test_post_list.append(test_post)

        print(f"   ✅ {exp_names[m_idx]}")

    # 用验证集选出的权重融合测试集概率
    fused_test = np.average(test_post_list, axis=0, weights=final_w)

    # 输出
    final_preds = np.argmax(fused_test, axis=1)
    output_path = os.path.join(model_save_dir, "B1_final_ensemble_v3.csv")

    final_df = pd.DataFrame({
        "test_idx": ref_test_idx,
        "label": final_preds.astype(int)
    })
    if best_val_res['acc']>global_best_merge_acc:
        global_best_merge_acc = best_val_res['acc']
        final_df.to_csv(output_path, index=False)
        dual_print(f"\n💾 【新的最佳提交结果生成】已保存: {output_path} ")

    unique, counts = np.unique(final_preds, return_counts=True)
    dist = ", ".join(f"{int(u)}:{int(c)}" for u, c in zip(unique, counts))
    print(f"\n💾 已保存: {output_path} ({len(final_df)} 样本)")
    print(f"   类别分布: {{{dist}}}")

    # 同时输出 softmax
    softmax_path = os.path.join(model_save_dir, "B1_final_ensemble_v3_softmax.csv")
    softmax_df = pd.DataFrame(fused_test, columns=class_cols)
    softmax_df.insert(0, 'test_idx', ref_test_idx)
    for c in class_cols:
        softmax_df[c] = softmax_df[c].map(lambda x: f"{x:.4f}")
    
    softmax_df.to_csv(softmax_path, index=False)
    print(f"💾 Softmax 已保存: {softmax_path}")

    print(f"\n{'='*70}")
    print(f"完成！最终方案: {best_method}")
    print(f"  验证集 ACC (公平): {best_val_res['acc']:.4f}")
    print(f"  测试集已输出 (使用了更多标签的后处理)")
    print(f"{'='*70}")


def run_ensemble():
    """训练结束后自动执行 Softmax 集成"""

    try:
        ensemble_with_postprocess_v3(
            model_save_dir=MODEL_SAVE_DIR,
            data_path=DATA_PATH,
            num_classes=NUM_CLASSES
        )
    except Exception as e:
        dual_print(f"❌ 模型集成阶段异常: {e}")
        

