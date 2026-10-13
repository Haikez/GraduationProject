# SCF-Mamba 网络结构规格书（画图用）
### Architecture Specification for Diagram Drawing — SCF-Mamba (Split-stream Cross-attention Fusion Mamba)

> 依据代码：`BasicTS/src/basicts/models/SCFMamba/` 中的 `arch/scf_mamba_arch.py`、`arch/sci.py`、`arch/node_mamba.py`、`arch/cross_attention.py`、`arch/embedding.py`、`arch/torch_mamba.py`、`config/scf_mamba_config.py`、`data.py`、`train.py`
> 参数量公式已与单元测试 `tests/basicts_test/test_scf_mamba.py::test_public_config_shapes_and_parameter_counts` 中 8 组期望值**逐一精确核对一致**（见第 6 节），因此本文件中的结构关系（含"每个 SCIBlock 只实例化 1 个 NodeMamba、共 3 个实例、12 次调用"）可视为已核实。

---

## 0. 一页速览

- 输入：历史流量 `X: [B, 96, N]`（已按训练集逐节点标准化）+ 历史时间戳 `S: [B, 96, 2]`（第 0 列 = 日内位置/288，第 1 列 = 七日周期/7）。
- 输出：`Y_hat: [B, H, N]`（标准化空间，之后由 TaskFlow 反标准化）。
- 主干：**两级 SCI 交互树**（`T=96 → 48+48 → 24+24`），每级"合并前"调用**同一个**双向交叉注意力模块（共 3 次）。
- 状态空间：**NodeMamba** 里的双向 Mamba 沿**节点轴（长度 N）**扫描，不是沿时间轴；每个 NodeMamba 由 2 个 Mamba-1 块（正向 / 反向）组成。
- 无 TCN 分支、无时间门控融合、无实例图 / 模式库图分支（那属于另一套 MambaTCNCA 原型）。
- 预测头：`Conv1d(96 → H, kernel=1, bias=False)`，等价于"每个节点共享同一个 `H×96` 线性映射"。

### 符号

| 符号 | 含义 | 取值 |
| --- | --- | --- |
| `B` | batch size | 默认 64 |
| `T` | 历史长度 `input_len` | **固定 96**（不满足直接报错） |
| `N` | 节点数 `num_features` | ≥16；PEMS08 = 170，PEMS04 = 307 |
| `H` | 预测步长 `output_len` | 默认 12（已验证 12/24/48/96） |
| `L` | 当前层支路长度 | 一级 48（=T/2），二级 24（=T/4） |
| `L_e, L_o` | 偶 / 奇支路长度 | `ceil(L/2), floor(L/2)`；T=96 时均为 L/2 |
| `h` | 卷积支路隐藏宽度 | `h = floor(N/16)`（N=170→10；N=307→19） |
| `d` | NodeMamba 内部宽度 | 固定 128 |
| `c` | 日历条件向量 | 固定 64 维 |

---

## 1. 模块层级树（含实例数 / 调用次数 / 属性路径）

```
SCFMamba                                   [B,96,N] + [B,96,2] → [B,H,N]
├── time_embedding : CalendarEmbedding                       1 个实例, 1 次调用
├── backbone : SCIBackbone
│   ├── position_frequencies (buffer, 无可学习参数)            ——
│   ├── tree : SCITree(depth=1, history_len=96)               1 个实例, 1 次调用
│   │   ├── interaction : SCIBlock(T=96)                      1 个实例, 1 次调用
│   │   │   ├── scale_odd  : ConvStack                        1 个实例, 1 次调用
│   │   │   ├── scale_even : ConvStack                        1 个实例, 1 次调用
│   │   │   ├── update_odd : ConvStack                        1 个实例, 1 次调用
│   │   │   ├── update_even: ConvStack                        1 个实例, 1 次调用
│   │   │   └── node_mamba : NodeMamba(L=48)                   1 个实例, 4 次调用 ★共享
│   │   │       ├── input_proj  Linear(48→128)
│   │   │       ├── node_embedding  Parameter [1,N,128]
│   │   │       ├── time_proj   Linear(192→128)
│   │   │       ├── encoder : BidirectionalMamba              1 个实例, 4 次调用
│   │   │       │   ├── forward_mamba  : Mamba(128, d_state=32, d_conv=2, expand=1)
│   │   │       │   ├── backward_mamba : Mamba(128, d_state=32, d_conv=2, expand=1)
│   │   │       │   ├── mamba_norm LayerNorm(128), ffn_norm LayerNorm(128)
│   │   │       │   └── ffn_in / ffn_out : Conv1d(128,128,1) + GELU + Dropout(0.1)
│   │   │       ├── output_norm LayerNorm(128)
│   │   │       ├── output_proj Linear(128→48)
│   │   │       └── gate  Parameter 标量（初始 0）
│   │   ├── even : SCITree(depth=0, history_len=48)            1 个实例, 1 次调用
│   │   │   └── interaction : SCIBlock(T=48)
│   │   │       ├── scale_odd / scale_even / update_odd / update_even : ConvStack  ×4
│   │   │       └── node_mamba : NodeMamba(L=24)               1 个实例, 4 次调用 ★共享
│   │   └── odd  : SCITree(depth=0, history_len=48)            1 个实例, 1 次调用
│   │       └── interaction : SCIBlock(T=48)  （结构同上，独立参数）
│   └── head : ForecastHead  Conv1d(96→H, k=1, bias=False)     1 个实例, 1 次调用
└── fusion : CrossSequenceAttention(N, 96)                    1 个实例, 3 次调用 ★全局共享
```

★ **共享关系（画图时最容易画错的地方）**

| 模块 | 实例数 | 前向调用次数 | 调用位置 |
| --- | --- | --- | --- |
| `NodeMamba`（一级，L=48） | 1 | **4** | `scale_odd`、`scale_even`、`update_even`、`update_odd` 四条支路各一次 |
| `NodeMamba`（偶支，L=24） | 1 | **4** | 同上（在 even 子树的 SCIBlock 内） |
| `NodeMamba`（奇支，L=24） | 1 | **4** | 同上（在 odd 子树的 SCIBlock 内） |
| `CrossSequenceAttention` | 1 | **3** | ①偶子树合并前（L=24）②奇子树合并前（L=24）③根级合并前（L=48） |
| `Mamba`（单块） | 6 | **24** | 3 个 BidirectionalMamba × 2（forward / backward）× 4 次 NodeMamba 调用 |

> 单元测试直接断言了注意力的 3 次调用规模为 `[24, 24, 48]`（`test_shared_attention_three_temporal_sites`）。

---

## 2. 逐层张量追踪表（含数值示例 N=170, B=64, H=12）

约定：`[B, T, N]` 为**时间主序**；`[B, N, L]` 为**节点主序**（SCIBlock / NodeMamba 内部使用的布局）。

| # | 模块（English） | 代码属性路径 | 输入形状 | 输出形状 | 备注 |
| --- | --- | --- | --- | --- | --- |
| 0 | Input traffic window | `forward(inputs, inputs_timestamps)` | — | `X [B,96,N]` | 已标准化，仅流量通道 |
| 0' | Input timestamps | 同上 | — | `S [B,96,2]` | 形状校验：必须 = `(*X.shape[:2], 2)` |
| 1 | **CalendarEmbedding** | `time_embedding` | `S [B,96,2]` | `c [B,64]` | 只取 `S[:, -1]`：`round(·×[288,7])` → `[B,2]` → 查表 288×32 与 7×32 → 拼接 |
| 2 | **Node-wise Position Encoding**（buffer，无参数） | `backbone.position_frequencies` | — | `PE [96,N]` | `freq [ceil(N/2)]`；`angles [96,ceil(N/2)]`；`cat(sin,cos)` 后截取前 N 列 |
| 3 | **Residual input** | `encoded = X + PE[None]` | `X [B,96,N]`, `PE [96,N]` | `Z0 [B,96,N]` | 广播到 batch；原输入不被修改（`.clone()`） |
| 4 | **SCITree (root)** | `backbone.tree` | `Z0 [B,96,N]`, `c [B,64]` | `Z4 [B,96,N]` | 内部先 split 再递归再合并 |
| 4.1 | → **SCI Interaction Block (level 1)** | `tree.interaction` | `Z0 [B,96,N]` | `E1 [B,48,N]`, `O1 [B,48,N]` | 见 2.1 详解 |
| 4.2 | → **Node-wise Position Encoding**（默认输出即根级融合使用） | `fusion.position_encoding[:, idx]` | — | `[1,96,64]` | 由 `fusion` 内部持有 |
| 4.3 | → **SCI Interaction Block (level 2, even, T=48)** | `tree.even.interaction` | `E1 [B,48,N]` | `[B,24,N]` ×2 | 内部 NodeMamba(L=24) 被调用 4 次 |
| 4.4 | → **CrossSequenceAttention (call ①)** | `fusion` | `[B,24,N]` ×2 | `[B,24,N]` ×2 | even 子树「合并前」；位置索引 (0,4,…)/(2,6,…) |
| 4.5 | → **Interleaved Stream Merge** (even child) | `SCITree.forward` 末行 | `[B,24,N]` ×2 | `E1' [B,48,N]` | `stack(dim=2).flatten(1,2)`：索引 2k=even[k], 2k+1=odd[k] |
| 4.6 | → **SCI Interaction Block (level 2, odd, T=48)** + **Fusion (call ②)** + **Merge** | `tree.odd.*` | `O1 [B,48,N]` | `O1' [B,48,N]` | 与 4.3–4.5 同构，但**参数独立**；Fusion 位置索引 (1,5,…)/(3,7,…) |
| 4.7 | → **CrossSequenceAttention (call ③, root)** | `fusion` | `E1' [B,48,N]`, `O1' [B,48,N]` | `[B,48,N]` ×2 | 同一实例的第 3 次调用；位置索引 (0,2,…)/(1,3,…) |
| 4.8 | → **Interleaved Stream Merge** (root) | `SCITree.forward` 末行 | `[B,48,N]` ×2 | `Z4 [B,96,N]` | 还原原始时间顺序（2k=even, 2k+1=odd） |
| 5 | **Residual add** | `features + encoded` | `Z4 [B,96,N]`, `Z0 [B,96,N]` | `[B,96,N]` | 逐元素相加 |
| 6 | **Forecast Head** | `backbone.head` | `[B,96,N]` | `Y_hat [B,H,N]` | `Conv1d(96→H, k=1, bias=False)`，通道维=时间、长度维=节点 → 每节点共享 `W∈R^{H×96}` |

### 2.1 SCI Interaction Block 内部（以一级 T=96 为例，其余级同构）

前置：`history [B,96,N]`；`h = floor(N/16)`；`L = T/2 = 48`。

| 子步骤 | 运算 | 输入 | 输出 |
| --- | --- | --- | --- |
| a | 奇偶分流：`even = history[:, ::2].transpose(1,2)`；`odd = history[:, 1::2].transpose(1,2)` | `[B,96,N]` | `even, odd: [B,N,48]` |
| b | `ConvStack scale_odd(even)` | `[B,N,48]` | `[B,N,48]`（中间：`[B,N,54]→[B,h,50]→[B,h,50]→[B,N,48]`） |
| c | `delta_odd = node_mamba(·, c)` | `[B,N,48]`, `c [B,64]` | `[B,N,48]` |
| d | `odd_scaled = odd * exp(delta_odd)` | ×2 `[B,N,48]` | `[B,N,48]` |
| e | `ConvStack scale_even(odd)` → `delta_even = node_mamba(·, c)` → `even_scaled = even * exp(delta_even)` | `[B,N,48]` | `even_scaled [B,N,48]` |
| f | `even_update = node_mamba(update_even(odd_scaled), c)` | `[B,N,48]` | `[B,N,48]` |
| g | `odd_update = node_mamba(update_odd(even_scaled), c)` | `[B,N,48]` | `[B,N,48]` |
| h | 输出（转回时间主序） | — | `(even_scaled + even_update).transpose(1,2) → [B,48,N]`；`(odd_scaled − odd_update).transpose(1,2) → [B,48,N]` |

> 注意：`node_mamba` 是**同一个实例**被调用 4 次；4 条卷积支路（`scale_odd / scale_even / update_odd / update_even`）是**4 个独立实例**。
> 修正号：偶支用 **加**、奇支用 **减**（`even_scaled + even_update`，`odd_scaled − odd_update`）。

### 2.2 三处融合调用的层级与位置索引（T=96）

| 融合调用 | 所在位置 | 偶支形状 | 奇支形状 | 偶支位置索引（绝对） | 奇支位置索引（绝对） |
| --- | --- | --- | --- | --- | --- |
| ① | even 子树（depth=0）合并前 | `[B,24,N]` | `[B,24,N]` | `(0,4,8,…,92)` | `(2,6,10,…,94)` |
| ② | odd 子树（depth=0）合并前 | `[B,24,N]` | `[B,24,N]` | `(1,5,9,…,93)` | `(3,7,11,…,95)` |
| ③ | 根（depth=1）合并前 | `[B,48,N]` | `[B,48,N]` | `(0,2,4,…,94)` | `(1,3,5,…,95)` |

---

## 3. 模块逐一定义（输入 / 输出 / 内部维度）

### 3.1 CalendarEmbedding（`arch/embedding.py`）

| 项 | 内容 |
| --- | --- |
| 输入 | `timestamps [B,96,2]`（float；`inputs_timestamps.to(inputs)` 后送入） |
| 内部 | `idx = round(timestamps[:, -1] * tensor([288,7])).long() → [B,2]`；越界即抛 `ValueError` |
| 可学习参数 | `time_of_day [288,32]`、`day_of_cycle [7,32]`（xavier_uniform） |
| 输出 | `c = cat(time_of_day[idx0], day_of_cycle[idx1]) → [B,64]` |
| 画图要点 | 只依赖**最后一个历史时刻**的日历信息，是"每样本一个"的全局条件向量，虚线注入每个 NodeMamba；不参与时间维计算 |

### 3.2 Node-wise Sinusoidal Position Encoding（`SCIBackbone`，无可学习参数）

| 项 | 内容 |
| --- | --- |
| 输入 | 无（构造函数中生成 buffer `position_frequencies [ceil(N/2)]`） |
| 内部 | `positions = arange(96) [96]`；`angles = positions[:,None] * freq[None,:] → [96, ceil(N/2)]`；`encoding = cat(sin, cos, dim=1) → [96, 2*ceil(N/2)]`，截取前 N 列 |
| 输出 | `PE [96,N]`，与 `X [B,96,N]` 逐元素相加（广播到 B） |
| 画图要点 | 编码维度是**节点维**（每个节点一个固定的正弦指纹），不是时间维位置编码；`T=96` 时 N 为偶数则刚好占满 |

### 3.3 SCITree（两级交互树）（`arch/sci.py`）

| 项 | 内容 |
| --- | --- |
| `SCITree(history_len, num_nodes, depth=1)` | 含 1 个 `SCIBlock(history_len)`；`depth>0` 时再建 `even`/`odd` 两个 `SCITree(history_len//2, depth-1)` |
| 输入 | `history [B,T,N]`、`calendar [B,64]`、`fusion`、`positions: tuple[int]` |
| 输出 | `[B,T,N]`（`stack(even, odd, dim=2).flatten(1,2)` 交错还原） |
| 调用顺序 | ① `interaction` → ② 递归 even/odd 子树 → ③ `fusion`（每级合并前） → ④ 交错合并 |
| depth | 代码硬编码 `depth=1`（未开放配置） |

### 3.4 ConvStack（`SCIBlock` 内 4 条，名字 `scale_odd / scale_even / update_odd / update_even`）

| 项 | 内容 |
| --- | --- |
| 输入 | `[B, N, L]`（节点主序；L = T/2） |
| 层序 | `ReplicationPad1d((3,3))` → `Conv1d(N → h, k=5)` → `LeakyReLU(0.01)` → `Dropout(0.5)` → `Conv1d(h → N, k=3)` → `Tanh` |
| 中间维度 | `[B,N,L]` → `[B,N,L+6]` → `[B,h,L+2]` → `[B,h,L+2]` → `[B,N,L]` → `[B,N,L]`（k=5、pad=3 两侧 → 长度保持 L） |
| 输出 | `[B, N, L]`，值域 (−1,1)（Tanh） |
| 画图要点 | 卷积沿**时间维**；使用**对称复制填充**（非因果），因为整段窗口都是"已观测历史"；隐藏宽度 h = floor(N/16) 很窄，属于轻量支路 |

### 3.5 NodeMamba（`arch/node_mamba.py`）

| 项 | 内容 |
| --- | --- |
| 输入 | `features [B, N, L]`（L = 该级 T/2：一级 48、二级 24）、`calendar [B,64]` |
| 步骤 1 | `input_proj: Linear(L → 128)` → `[B,N,128]`；`Dropout(0.1)` 后 **加** `node_embedding [1,N,128]`（xavier）→ `[B,N,128]` |
| 步骤 2 | `calendar.unsqueeze(1).expand(-1, N, -1) → [B,N,64]`；`cat((embedded, cal), -1) → [B,N,192]`；`time_proj: Linear(192 → 128)` → `[B,N,128]` |
| 步骤 3 | `encoder: BidirectionalMamba` → `[B,N,128]`（沿节点轴扫描，见 3.6） |
| 步骤 4 | `output_norm: LayerNorm(128)` → `[B,N,128]`；`output_proj: Linear(128 → L)` → `[B,N,L]` = `correction` |
| 步骤 5 | `return features + tanh(gate) * correction` → `[B,N,L]`（`gate` 为标量，初始 0 ⇒ 初始时为恒等映射） |
| 训练期 | 当 `activation_checkpoint=True` 且 `model.training` 时，步骤 1–4 用 `torch.utils.checkpoint` 重算以省显存 |
| 画图要点 | 输入输出的形状完全相同（`[B,N,L]→[B,N,L]`），是"节点轴上的双向记忆 + 日历条件调制"的残差块 |

### 3.6 BidirectionalMamba（`arch/node_mamba.py`）

| 项 | 内容 |
| --- | --- |
| 输入 | `features [B, N, 128]`（序列维 = 节点维，长度 N） |
| 正向 | `forward_mamba(features)` → `[B,N,128]`（沿节点轴 k=1..N 扫描） |
| 反向 | `backward_mamba(features.flip(1)).flip(1)` → `[B,N,128]`（沿节点轴逆序扫描后再翻回） |
| 融合 | `residual = mamba_norm(forward + backward)` → `[B,N,128]`（LayerNorm） |
| 前馈 | `FFN`：`transpose(1,2) → [B,128,N]` → `Conv1d(128,128,1)` → `GELU` → `Dropout(0.1)` → `Conv1d(128,128,1)` → `transpose(1,2) → [B,N,128]`（再 Dropout(0.1)） |
| 输出 | `ffn_norm(hidden + residual)` → `[B,N,128]` |
| 画图要点 | 两个 Mamba 分支是**两个独立实例**（`forward_mamba` / `backward_mamba`），参数不共享；扫描方向是节点维 |

### 3.7 Selective SSM（Mamba-1 块，`arch/torch_mamba.py` 或 `mamba_ssm.Mamba`）

| 项 | 内容 |
| --- | --- |
| 构造参数 | `d_model=128, d_state=32, d_conv=2, expand=1`（`d_inner = 128`，`dt_rank = ceil(128/16) = 8`） |
| 输入 / 输出 | `[B, N, 128] → [B, N, 128]` |
| 内部形状 | `in_proj: Linear(128 → 256)` → 切分为 `signal, gate: [B,N,128]`；`conv1d(depthwise, k=2)` → `silu`；`x_proj: Linear(128 → 8+64)` → `dt [B,N,8]`, `B_t, C_t [B,N,32]`；`dt_proj: Linear(8 → 128)` + softplus → `Δ [B,N,128]`；状态 `h [B,N,128,32]`（Python 循环沿长度 N 推进）；输出 `(C_t ⊙ h).sum(-1) + D ⊙ signal`，再乘 `silu(gate)`，最后 `out_proj: Linear(128 → 128)` |
| 递推 | `h_k = exp(Δ_k ⊙ A) ⊙ h_{k−1} + Δ_k ⊙ (B_k x_k)`，`A = −exp(A_log)` |
| 后端 | `mamba_backend="official"`（CUDA mamba_ssm）或 `"torch"`（纯 PyTorch CPU 校验，键名与 Mamba-1 对齐）；两者参数量与形状一致 |
| 画图要点 | 画成"SSM 扫描"箭头即可，箭头方向标注**沿节点轴**；`d_state=32` 为每通道状态维 |

### 3.8 CrossSequenceAttention（`arch/cross_attention.py`，全局共享 1 个实例）

| 项 | 内容 |
| --- | --- |
| 构造 | `CrossSequenceAttention(num_nodes=N, history_len=96, seed=54321)`；宽度 `width=64`，4 头（每头 16 维） |
| 输入 | `even [B,L_e,N]`、`odd [B,L_o,N]`、`even_positions`、`odd_positions`（元组，元素为 0..95 的绝对时间索引） |
| 嵌入 | `input_proj: Linear(N → 64)` → `[B,L,64]`；加 `position_encoding[:, positions] → [L,64]`（buffer `[1,96,64]`，正弦）；`LayerNorm(64)` → `[B,L,64]` |
| 交叉注意力 | `MHAttention(64, heads=4, dropout=0.1)`：`Q=E_feat, K=V=O_feat` → `[B,L_e,64]`；`Q=O_feat, K=V=E_feat` → `[B,L_o,64]`（`need_weights=False`） |
| 输出投影 | `output_proj: Linear(64 → N)` → `[B,L,N]` |
| 残差门控 | `even + tanh(even_gate) * ΔE`、`odd + tanh(odd_gate) * ΔO`（两个标量门，初始 0 ⇒ 初始恒等） |
| 输出 | `[B,L_e,N]`、`[B,L_o,N]`（形状与输入一致） |
| 随机流 | 注意力 dropout 使用**独立 RNG 流**（`fork_rng` + seed 54321），其状态通过 `get_extra_state/set_extra_state` 存入 checkpoint |
| 画图要点 | token = **时间步**，token 的特征维 = **节点数 N**；注意力沿时间轴交换信息，不逐节点独立；三处调用共享同一套参数与同一张位置编码表 |

### 3.9 Interleaved Stream Merge（`SCITree.forward` 末行）

| 项 | 内容 |
| --- | --- |
| 输入 | `even [B,L,N]`、`odd [B,L,N]` |
| 运算 | `torch.stack((even, odd), dim=2)` → `[B,L,N,2]`；`.flatten(1,2)` → `[B,2L,N]` |
| 语义 | 按原时间索引交错还原：输出第 `2k` 步 = `even[k]`，第 `2k+1` 步 = `odd[k]` |
| 输出 | 一级：`[B,48,N]→[B,96,N]`；二级：`[B,24,N]→[B,48,N]` |

### 3.10 Forecast Head（`SCIBackbone.head`）

| 项 | 内容 |
| --- | --- |
| 输入 | `features + encoded → [B,96,N]` |
| 运算 | `Conv1d(in_channels=96, out_channels=H, kernel_size=1, bias=False)`：把**时间维当通道**、**节点维当长度** → 对每个节点施加同一个 `W ∈ R^{H×96}` |
| 输出 | `Y_hat [B,H,N]` |
| 参数量 | `96 × H`（H=12→1152，24→2304，48→4608，96→9216） |
| 画图要点 | 可画成"逐节点共享的线性预测头（96 → H）"，方向为时间→未来步，节点维保持不变 |

---

## 4. 端到端形状汇总（可直接标注在图上）

```
X [B,96,N] ──►(+PE[96,N])──► X' [B,96,N]
S [B,96,2] ──► CalendarEmbedding ──► c [B,64]  ┈┈(虚线条件)┈┈► 每个 NodeMamba

X' ──► SCIBlock(T=96) ──► E1 [B,48,N] , O1 [B,48,N]
        E1 ──► SCIBlock(T=48) ──► [B,24,N],[B,24,N] ──► Fusion① ──► [B,24,N],[B,24,N] ──► Interleave ──► [B,48,N]
        O1 ──► SCIBlock(T=48) ──► [B,24,N],[B,24,N] ──► Fusion② ──► [B,24,N],[B,24,N] ──► Interleave ──► [B,48,N]
        [B,48,N],[B,48,N] ──► Fusion③ ──► [B,48,N],[B,48,N] ──► Interleave ──► Z4 [B,96,N]

Z4 + X' ──► ForecastHead Conv1d(96→H, k=1) ──► Y_hat [B,H,N]

其中 SCIBlock(T) 内部：
  history [B,T,N] ─► even/odd [B,N,T/2] ─► 4×ConvStack ─► 4×NodeMamba(共享) ─► [B,T/2,N] ×2
  尺度调制： O' = O ⊙ exp(NM(Conv_scaleOdd(E))) ; E' = E ⊙ exp(NM(Conv_scaleEven(O)))
  增量更新： E'' = E' + NM(Conv_updateEven(O')) ; O'' = O' − NM(Conv_updateOdd(E'))
```

---

## 5. 参数量分解（公式 + 已核对数值）

公式（N = 节点数，H = 预测步长，`h = floor(N/16)`）：

| 模块 | 参数量公式 | N=170 | N=307 |
| --- | --- | --- | --- |
| CalendarEmbedding | `288×32 + 7×32 = 9440` | 9,440 | 9,440 |
| Forecast Head | `96 × H` | 1,152 (H=12) | 1,152 (H=12) |
| ConvStack ×1 | `(5Nh + h) + (3Nh + N) = 8Nh + h + N` | 13,780 | 46,990 |
| ConvStack ×12（3 个 SCIBlock × 4） | `12 × (8Nh + h + N)` | 165,360 | 563,880 |
| Mamba-1 块 ×1 | in_proj 32,768 + conv 384 + x_proj 9,216 + dt_proj 1,152 + A_log 4,096 + D 128 + out_proj 16,384 = **64,128** | 64,128 | 64,128 |
| BidirectionalMamba（=2 Mamba + FFN + 2 LN） | `2×64,128 + 2×(128×128+128) + 2×(2×128) = 161,792` | 161,792 | 161,792 |
| NodeMamba(L) | `161,792 + 24,704(time_proj) + 128N(node_emb) + (128L+128) + (128L+L) + 256(LN) + 1(gate)` | L=48: 220,977；L=24: 214,809 | L=48: 238,769；L=24: 232,601 |
| NodeMamba ×3（一级 1 + 二级 2） | `NM48 + 2×NM24` | 650,595 | 703,971 |
| SCIBlock(T=96) 合计 | `4×ConvStack + NM48` | 276,097 | 426,473 |
| SCIBlock(T=48) ×2 合计 | `2 × (4×ConvStack + NM24)` | 539,858 | 840,610 |
| CrossSequenceAttention | `(64N+64) + 128 + 16,640 + (64N+N) + 2 = 129N + 16,834` | 38,764 | 56,437 |
| **总计（H=12）** | — | **865,311** | **1,334,112** |
| 总计（H=24 / 48 / 96） | 每个 +12 步增加 1,152 | 866,463 / 868,767 / 873,375 | 1,335,264 / 1,337,568 / 1,342,176 |

> 以上 8 个总数与单元测试 `test_public_config_shapes_and_parameter_counts` 的期望值**完全一致**，可用于校验重绘/复现的模型是否与代码一致。

---

## 6. 画图节点-边清单（Diagram-ready）

### 节点（建议英文标签 → 形状标注）

| ID | Label（建议写在图上） | In shape | Out shape | 颜色建议 |
| --- | --- | --- | --- | --- |
| IN | Traffic Window `X` / Timestamps `S` | — | `[B,96,N]` / `[B,96,2]` | 灰 |
| CAL | Calendar Embedding | `[B,96,2]` | `[B,64]` | 青 |
| PE | Node-wise Position Encoding | — | `[96,N]` | 灰 |
| ADD1 | ⊕ (add position encoding) | `[B,96,N] + [96,N]` | `[B,96,N]` | — |
| SCI1 | SCI Interaction Block (Level 1, T=96) | `[B,96,N]` | `[B,48,N]×2` | 蓝 |
| BR-E | Branch: Even stream (T=48) | `[B,48,N]` | `[B,48,N]` | 蓝 |
| BR-O | Branch: Odd stream (T=48) | `[B,48,N]` | `[B,48,N]` | 蓝 |
| SCI2E | SCI Interaction Block (Level 2, Even, T=48) | `[B,48,N]` | `[B,24,N]×2` | 蓝 |
| SCI2O | SCI Interaction Block (Level 2, Odd, T=48) | `[B,48,N]` | `[B,24,N]×2` | 蓝 |
| FUSE | Cross-Sequence Attention Fusion（共享，3 处） | `[B,L,N]×2` | `[B,L,N]×2` | 紫 |
| MRG | Interleaved Stream Merge | `[B,L,N]×2` | `[B,2L,N]` | 绿 |
| ADD2 | ⊕ (residual add) | `[B,96,N]×2` | `[B,96,N]` | — |
| HEAD | Forecast Head (per-node linear, 96→H) | `[B,96,N]` | `[B,H,N]` | 黄 |
| OUT | Prediction `Y_hat` | — | `[B,H,N]` | 灰 |

### 节点内部的子层（若要画模块放大图）

| 父节点 | 子层（英文） | 形状 |
| --- | --- | --- |
| SCI1 / SCI2E / SCI2O | Even-Odd Split | `[B,T,N] → [B,N,T/2]×2` |
| 同上 | ConvStack ×4（scale_odd, scale_even, update_odd, update_even） | `[B,N,L] → [B,N,L]`（内部 `[B,h,L+2]`） |
| 同上 | NodeMamba ×4（1 个实例） | `[B,N,L] → [B,N,L]` |
| NodeMamba | Linear(L→128) → +NodeEmbedding(N×128) → Concat(calendar 64) → Linear(192→128) | `[B,N,128]` |
| NodeMamba | Bidirectional SSM（forward + backward，沿节点轴） | `[B,N,128] → [B,N,128]` |
| NodeMamba | LayerNorm → Linear(128→L) → gated residual | `[B,N,L]` |
| FUSE | Linear(N→64) + Sinusoidal PE → 4-head MHAttention（双向） → Linear(64→N) → tanh-gated residual | `[B,L,64] → [B,L,N]` |
| HEAD | 1×1 Conv1d (96→H, bias=False) | `[B,96,N] → [B,H,N]` |

### 边（数据流）

| From → To | 形状 | 线型 |
| --- | --- | --- |
| IN → ADD1 | `[B,96,N]` | 实线 |
| PE → ADD1 | `[96,N]` | 实线 |
| IN(S) → CAL | `[B,96,2]` | 实线 |
| CAL → SCI1.node_mamba / SCI2*.node_mamba | `[B,64]` | **虚线**（条件注入，共 3 处 NodeMamba） |
| ADD1 → SCI1 | `[B,96,N]` | 实线 |
| SCI1 → BR-E / BR-O | `[B,48,N]` | 实线 |
| BR-E → SCI2E → FUSE① → MRG → `[B,48,N]` | — | 实线（子图内循环） |
| BR-O → SCI2O → FUSE② → MRG → `[B,48,N]` | — | 实线 |
| （两子树输出）→ FUSE③ → MRG → `[B,96,N]` | — | 实线 |
| ADD1 → ADD2 | `[B,96,N]` | **虚线**（残差，跨越主干） |
| MRG(root) → ADD2 | `[B,96,N]` | 实线 |
| ADD2 → HEAD → OUT | `[B,96,N] → [B,H,N]` | 实线 |

---

## 7. Mermaid 备用流程图（可直接粘进 Markdown）

```mermaid
flowchart LR
  X["Traffic Window X<br/>[B,96,N]"] --> ADD(("+"))
  PE["Node-wise Position Encoding<br/>[96,N]"] --> ADD
  S["Timestamps S<br/>[B,96,2]"] --> CAL["Calendar Embedding<br/>[B,64]"]
  CAL -.->|condition c [B,64]| NM
  ADD --> SCI1["SCI Interaction Block (L1, T=96)"]
  SCI1 --> E1["Even stream [B,48,N]"]
  SCI1 --> O1["Odd stream [B,48,N]"]
  E1 --> SCI2E["SCI Block (L2 Even, T=48)"] --> F1["Cross-Sequence Attention (shared)"]
  F1 --> M1["Interleaved Merge [B,48,N]"]
  O1 --> SCI2O["SCI Block (L2 Odd, T=48)"] --> F2["Cross-Sequence Attention (shared)"]
  F2 --> M2["Interleaved Merge [B,48,N]"]
  M1 --> F3["Cross-Sequence Attention (shared)"]
  M2 --> F3
  F3 --> M3["Interleaved Merge [B,96,N]"]
  M3 --> ADD2["+ residual"]
  ADD -.-> ADD2
  ADD2 --> HEAD["Forecast Head: Conv1d 96→H"] --> OUT["Prediction [B,H,N]"]
  NM["NodeMamba (bidirectional SSM over node axis)<br/>[B,N,L] → [B,N,L]"]
  SCI1 -.->|4 calls| NM
  SCI2E -.->|4 calls| NM
  SCI2O -.->|4 calls| NM
```

---

## 8. 机器可读附录（JSON）

```json
{
  "model": "SCFMamba",
  "config": { "input_len": 96, "output_len": "H", "num_features": "N >= 16",
              "mamba_backend": ["official", "torch"], "activation_checkpoint": true },
  "symbols": { "B": "batch", "T": 96, "N": "nodes", "H": "horizon",
               "L": "stream length (48 at level1, 24 at level2)",
               "h": "floor(N/16)", "d_model": 128, "d_state": 32, "width": 64 },
  "call_counts": { "CalendarEmbedding": 1, "SCIBlock": 3, "ConvStack": 12,
                   "NodeMamba_instances": 3, "NodeMamba_calls": 12,
                   "Mamba_instances": 6, "Mamba_calls": 24,
                   "CrossSequenceAttention_instances": 1, "CrossSequenceAttention_calls": 3,
                   "ForecastHead": 1 },
  "fusion_call_sites": [
    { "site": "even_subtree", "even_shape": "[B,24,N]", "odd_shape": "[B,24,N]",
      "even_positions": "0,4,...,92", "odd_positions": "2,6,...,94" },
    { "site": "odd_subtree",  "even_shape": "[B,24,N]", "odd_shape": "[B,24,N]",
      "even_positions": "1,5,...,93", "odd_positions": "3,7,...,95" },
    { "site": "root",         "even_shape": "[B,48,N]", "odd_shape": "[B,48,N]",
      "even_positions": "0,2,...,94", "odd_positions": "1,3,...,95" }
  ],
  "tensor_trace": [
    ["input", "X", "[B,96,N]"], ["input", "S", "[B,96,2]"],
    ["CalendarEmbedding", "c", "[B,64]"],
    ["PositionEncoding", "PE", "[96,N]"],
    ["add", "Z0", "[B,96,N]"],
    ["SCIBlock_L1", "E1,O1", "[B,48,N]"],
    ["SCIBlock_L2_even", "even2,odd2", "[B,24,N]"],
    ["Fusion_1", "even2',odd2'", "[B,24,N]"],
    ["Merge_even", "E1'", "[B,48,N]"],
    ["SCIBlock_L2_odd", "even2,odd2", "[B,24,N]"],
    ["Fusion_2", "even2',odd2'", "[B,24,N]"],
    ["Merge_odd", "O1'", "[B,48,N]"],
    ["Fusion_3", "E1'',O1''", "[B,48,N]"],
    ["Merge_root", "Z4", "[B,96,N]"],
    ["residual", "Z4 + Z0", "[B,96,N]"],
    ["ForecastHead", "Y_hat", "[B,H,N]"]
  ],
  "parameter_counts": { "N170_H12": 865311, "N170_H96": 873375,
                        "N307_H12": 1334112, "N307_H96": 1342176 }
}
```

---

## 9. 绘制注意点（避免画错的 12 条）

1. **Mamba 扫描方向是节点轴**（长度 N），不是时间轴；图上的 SSM 箭头应沿"节点"方向。
2. 每个 `SCIBlock` 只有 **1 个 NodeMamba 实例**，却被调用 **4 次**；图上这 4 条支路应指向同一个方框（或标注 "shared, ×4"）。
3. **交叉注意力只有 1 个实例、被调用 3 次**（两级子树各 1 次 + 根级 1 次），参数与位置编码表共享。
4. 每条支路的卷积是 **4 个独立 ConvStack**（`scale_odd / scale_even / update_odd / update_even`），参数不共享。
5. 更新符号不同：偶支 `+`，奇支 `−`。
6. 位置编码有**两套**：主干输入上的**节点维**正弦编码（无参数），以及交叉注意力内部的**时间维**正弦编码（buffer `[1,96,64]`，按绝对时间索引切片）。
7. 日历嵌入只取**最后一个**历史时间戳，得到每样本一个 `[B,64]` 条件向量（画成虚线注入）。
8. 时间戳不作为序列特征进入主干，只用于日历嵌入。
9. 主干残差是"`tree(encoded) + encoded`"，即残差项是**加过位置编码的输入**，不是原始 X。
10. 三个标量门（每个 NodeMamba 的 `gate`、注意力模块的 `even_gate/odd_gate`）**初始为 0**，初值时两条残差支路为恒等映射，可在图上标注 "gated residual (init 0)"。
11. 前馈用 `Conv1d(128,128,1)` 实现（等价于在节点轴上共享的 1×1 线性层），不是标准 Linear。
12. 输出头是 `Conv1d(96→H, k=1, bias=False)`：**无偏置**，且各节点共享同一组权重。

---

## 10. 与初稿第 3 章的差异（一句话）

初稿 3.3–3.7 节与图 3-1（mermaid）描述的是 **TCN + 输入选择性状态递推 + 时间门控融合 + 实例图/模式库图** 的原型（对应 BasicTS 里的 MambaTCNCA）；本文件对应的是 `SCFMamba` 目录的真实实现（**SCI 奇偶分流 + 节点轴双向 Mamba + 每级合并前的共享双向交叉注意力**，无 TCN、无时间门控、无图分支）。画图与正文务必与最终选定的模型一致。
