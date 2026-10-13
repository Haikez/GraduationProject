# SCF-Mamba 论文框架图（依据代码绘制）

> 绘制依据：`BasicTS/src/basicts/models/SCFMamba/`（`arch/scf_mamba_arch.py`、`arch/sci.py`、`arch/node_mamba.py`、`arch/cross_attention.py`、`arch/embedding.py`、`config/scf_mamba_config.py`、`README.md`）
> 生成方式：由 DSH 会话按代码结构程序化生成矢量图，几何与文字均可在 SVG 中直接编辑。

## 一、文件清单

| 文件 | 内容 | 画布尺寸 |
| --- | --- | --- |
| `SCFMamba_framework.svg` | 图 1：模型总体结构与数据流 | 1220 × 620 |
| `SCFMamba_modules.svg` | 图 2：关键模块细节 (b)(c)(d) | 1260 × 1110 |
| `SCFMamba_full.svg` | 图 1 + 图 2 合并版（单张长图） | 1260 × 1760 |
| `SCFMamba_figures.html` | 浏览器预览页，可直接 Ctrl+P 打印成 A4 横向 PDF（每页一图） | — |
| `SCFMamba_figures_tikz.tex` | LaTeX（xelatex + ctex / xeCJK）TikZ 版本，含两幅图 | — |
| `README_框架图.md` | 本说明 | — |

## 二、建议图注

**图 3-x　SCF-Mamba 模型总体结构。** 模型以 96 步历史流量窗口与历史时间戳为输入：流量经节点维正弦位置编码后进入主干，时间戳经日历嵌入得到条件向量 `c`。主干为二级 SCI 交互树，在每一级上对奇、偶子序列分别做卷积交互与 NodeMamba 调制，并在每级合并之前调用参数共享的双向交叉注意力完成子序列间的信息交换；主干输出与位置编码后的输入做残差相加，最后由逐节点共享的 1×1 卷积预测头一次性输出 H 步结果。

**图 3-y　SCF-Mamba 关键模块结构。** (b) SCI 交互块：奇偶分流、4 条卷积支路以及尺度调制与增量更新；(c) NodeMamba 块：沿节点轴的双向选择性状态空间（SSM），受日历条件调制并以门控残差输出；(d) 共享双向交叉注意力融合模块：偶、奇子序列互为查询与键值，输出经 tanh 门控残差后按原始时间索引交错还原。

## 三、图中模块 ↔ 代码对应

| 图中模块 | 代码位置 | 关键实现 |
| --- | --- | --- |
| 输入与校验 | `scf_mamba_arch.py` | `input_len` 固定为 96，`num_features ≥ 16`；输入 `[B,96,N]`、时间戳 `[B,96,2]` |
| 节点维正弦位置编码 | `sci.py::SCIBackbone` | `scales=ceil(N/2)`，sin/cos 拼接为 `96×N`，逐元素加到输入上（`encoded = history + encoding`） |
| 日历嵌入 | `embedding.py` | 只取**最后一个**时间戳：日内位置/288 → 32 维、七日周期/7 → 32 维，共 64 维 |
| SCI 交互块 | `sci.py::SCIBlock` | 奇偶分流；4 条卷积支路（`ReplicationPad1d(3,3) → Conv1d(5) → LeakyReLU → Dropout(0.5) → Conv1d(3) → Tanh`，隐藏宽度 `⌊N/16⌋`）；`odd*exp(NM(...))`、`even*exp(NM(...))`、`even+NM(...)`、`odd-NM(...)` |
| NodeMamba 块 | `node_mamba.py` | `Linear(L→128)` + 可学习节点嵌入 `N×128` → 拼接 64 维日历 → `Linear(192→128)` → 双向 Mamba（沿**节点轴**扫描）→ LayerNorm + 1×1 Conv 前馈 → `Linear(128→L)`；输出为 `F + tanh(g)·ΔF`，`g` 初始为 0 |
| 状态递推 | `torch_mamba.py`（torch 后端）/ `mamba_ssm.Mamba`（official 后端） | 官方后端调用 `Mamba(d_model=128, d_state=32, d_conv=2, expand=1)` |
| 共享双向交叉注意力 | `cross_attention.py` | `Linear(N→64)` + 正弦位置编码 → 4 头 `MultiheadAttention`（E 查询 O、O 查询 E）→ `Linear(64→N)` → `E + tanh(even_gate)·ΔE`、`O + tanh(odd_gate)·ΔO`；注意力独立 dropout 随机流（seed 54321），并随 `state_dict` 保存 |
| 交错合并 | `sci.py::SCITree.forward` | `torch.stack((even, odd), dim=2).flatten(1,2)`，按原时间索引还原为 T 步 |
| 残差与预测头 | `sci.py::SCIBackbone.forward` | `self.head = Conv1d(input_len, output_len, 1, bias=False)`，输入为 `features + encoded`，各节点共享权重 |

## 四、调用次数与形状（与图一致的技术细节）

- 主干为 `SCITree(depth=1)`：一级 SCI 块（T = 96）→ 偶/奇两个二级 SCI 块（T = 48）→ 根级合并。
- **NodeMamba 共调用 12 次**（一级 4 次 + 二级 2×4 次），每次包含正向与反向两条 Mamba 分支，即 24 次状态扫描。
- **共享交叉注意力共调用 3 次**（两个二级块内部各 1 次 + 根级 1 次），全部复用同一个 `CrossSequenceAttention` 实例。
- 前向形状：`[B,96,N] → 主干 [B,96,N] → +encoded → 预测头 → [B,H,N]`；一级块内部每支路 `96 → 48`，二级块内部每支路 `48 → 24`。
- NodeMamba 内部 `L = T/2`（即该级支路长度），`input_proj: Linear(L→128)`、`output_proj: Linear(128→L)`。

## 五、插入论文的方法

1. **Word / WPS**：插入 → 图片 → 选择 `SCFMamba_framework.svg` 与 `SCFMamba_modules.svg`。Word 2016 及以上、WPS 新版支持 SVG 矢量，分辨率无损；如期刊只收位图，可在浏览器中放大后截图，或用 Inkscape 导出 PNG（建议 600 dpi）。
2. **LaTeX**：直接使用 `SCFMamba_figures_tikz.tex`（用 xelatex 编译，中文需 ctex 或 xeCJK，颜色定义写在文件头部注释里）。也可以先转换：
       inkscape SCFMamba_framework.svg --export-type=pdf
   然后 \includegraphics 引入。
3. **PPT / Visio**：可直接把 SVG 拖入，文字与方框均为可编辑对象。
4. **字号提示**：两幅图的画布宽度约为 1220–1260 px。按整幅 16 cm 宽插入时，正文级文字约 6.7 pt（接近六号），标题约 7.4 pt；若希望更大，建议把图 2 的三块面板 (b)(c)(d) 拆成三张单图，或把图 2 单独占一页。

## 六、配色（与图中图例一致）

| 用途 | 填充 | 描边 |
| --- | --- | --- |
| 输入 / 输出 | `#F2F2F2` | `#808080` |
| 时间卷积 / 主干 | `#DEEBF7` | `#2E75B6` |
| 节点状态空间（SSM） | `#FCE4D6` | `#C55A11` |
| 交叉注意力 | `#E8E1F2` | `#7030A0` |
| 门控 / 融合 | `#E2EFDA` | `#548235` |
| 预测头 | `#FFF2CC` | `#BF9000` |
| 日历嵌入 / 条件支路 | `#DAEEF3` | `#31859B` |

## 七、重要提醒：与初稿第 3 章的模型不一致

初稿 `硕士学位论文初稿.md` 中的图 3-1（mermaid 流程图）以及 3.3–3.7 节描述的是另一个原型：**因果空洞卷积（TCN）+ 输入选择性状态递推 + 时间门控融合 + 样本相关图与模式库图的门控融合**，这对应 BasicTS 里的 `MambaTCNCA` 系列代码。

而 `src/basicts/models/SCFMamba/` 下的代码是 **Split-stream Cross-attention Fusion Mamba**：SCI 奇偶分流 + 沿节点轴的双向 Mamba + 每级合并前的共享双向交叉注意力；它**没有** TCN 分支、没有时间门控融合、也没有实例图/模式库图分支。

因此：

- 本框架图只对应 SCF-Mamba（`SCFMamba` 目录）的实际计算路径；
- 若论文第 3 章仍按初稿的“TCN + 状态递推 + 图融合”叙述，请**不要**直接插入本图，需先统一正文与代码，或另绘对应 MambaTCNCA 的结构图；
- 若论文最终以 SCF-Mamba 为主线，建议同步改写 3.1–3.7 节（把图 3-1 的 mermaid 换成 `SCFMamba_framework.svg`，模块公式改为本说明第三节所列实现）。
