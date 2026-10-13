# MambaTCNCA 训练与实验方案（BasicTS 当前版）

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-09-29
- Verification Status: UNVERIFIED（代码与本地数据已检查；训练尚未执行）
- Version Label: code_plan_v1

## 1. 研究目标与实验边界

以 BasicTS 中的 STIDGCN 派生版 MambaTCNCA 为对象，回答三个问题：完整模型相对 STIDGCN 和简单基线的误差与资源成本如何；TCN、Mamba 分支、跨序列注意力各自是否带来稳定变化；这些变化能否在 PEMS08 与 PEMS04 上重现。主要指标预先定为测试集全 12 步 MAE，次要指标为 RMSE、MAPE 和第 3、6、12 步 MAE。所有模型使用同一原始数据、时间切分、目标、缺失值规则与评估程序。

当前 `torch` 后端是非融合的 PyTorch Mamba-1 式递推，`official` 后端调用 [`mamba_ssm.Mamba`](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py)。两种后端是不同实验条件，不能混合汇总；本方案先固定 `torch` 后端。如果论文题目或结论明确声称使用官方 Mamba，则另外开展 `official` 后端复现与计时。

## 2. 开跑前的必要检查

1. **数据**：本地 `BasicTS/datasets/PEMS08` 有已处理数据和 `meta.json`；本地尚无 `BasicTS/datasets/PEMS04`。先取得 PEMS04 原始数据并按现有脚本处理，记录来源、版本、校验值、张量形状、时间范围、零值比例。两数据集现有规则均为按时间 6:2:2 切分、仅用训练集拟合标准化参数、零值掩码为 0。
2. **配置锁定**：统一历史 12 步、预测 12 步、batch 16、最多 300 轮、Ranger（学习率 0.001、权重衰减 0.0001）、梯度裁剪 5、`torch` 后端、相同 GPU。记录 Python、PyTorch、CUDA、GPU 型号、代码提交和未提交差异。不要在首轮结果后按测试集改超参数。
3. **测试隔离**：当前 MambaTCNCA 训练配置设 `test_interval=args.epochs`，会在末轮训练时访问测试集，又在训练结束后评估最佳验证检查点。正式运行前把训练期测试间隔设为大于最大轮数，只在训练结束后评估验证 MAE 最优检查点。把 `save_results` 设为 `True`，保留预测张量以复算指标、画图和做配对比较。
4. **基线对齐**：当前 STIDGCN 训练入口没有 `--seed` 参数，batch 默认 64，测试频率也不同。正式对照前统一种子、batch、数据与测试隔离规则。STID 模型代码存在，但没有用于 PEMS04/08 的现成训练入口；若纳入主表，应先补齐并核验相同评价口径。历史值持平预测可作为几乎不耗训练时长的简单基线。
5. **先做短检查**：在 PEMS08 上运行完整模型 `--steps 2 --batch-size 2`，只验证前向、反向、损失为有限数和显存占用；此模式不做验证、测试或检查点选择，不能当成结果。再用 3–5 个完整轮次估算每轮时间、验证开销和显存；按实际测量修正下表的时间预算。
6. **缩短无效训练**：先看预实验的验证曲线，再统一预设训练上限和早停规则。当前入口没有启用 `EarlyStopping`；若加入，建议以验证 MAE 连续 20 轮无改善为停止条件，所有变体和直接基线都用同一规则，并记录实际停止轮次。不要在看到测试指标后为某一组单独改早停或训练轮数。

## 3. 实验矩阵与预算

当前 CLI 已有四个可直接选择的变体；四组都保留原 STIDGCN 的动态图、时间嵌入与预测头。

| 名称 | 时间分支 | 奇偶交互 | 对应问题 |
|---|---|---|---|
| `full` | TCN + Mamba | 跨序列注意力 | 完整模型 |
| `without_mamba` | TCN | 跨序列注意力 | Mamba 分支贡献 |
| `without_tcn` | Mamba | 跨序列注意力 | TCN 分支贡献 |
| `without_ca` | TCN + Mamba | 原 STIDGCN Hadamard | 注意力贡献 |
| STIDGCN | 原始时间卷积 | 原始 Hadamard | 最直接结构基线 |
| STID | 简洁 MLP | 无上述交互 | 简单强基线，需补训练入口 |

推荐随机种子固定为 `0, 1, 2`。把每次完整训练暂按 4 小时估算；STID、STIDGCN、不同变体及 PEMS04 的实际耗时可能不同，下表只是排程基准，另外预留约 20% 的故障与重跑时间。

| 可用训练时间 | 具体矩阵 | 名义耗时 | 可以支持的结论 |
|---|---|---:|---|
| 约 40 小时 | 两数据集 ×（四变体 + STIDGCN）× 种子 0 | 10 次 × 4 小时 | 初筛与可行性；单种子差异不得写成稳定提升 |
| 约 80 小时（推荐） | 上述 10 次；STID 两数据集各 1 次；`full` 和 STIDGCN 在两数据集各补种子 1、2 | 20 次 × 4 小时 | 完整模型与直接基线可报告三种子均值及离散度；消融仍属探索性 |
| 约 120 小时 | 两数据集 ×（四变体 + STIDGCN）× 三种子 | 30 次 × 4 小时 | 核心模型与全部现成消融可做三种子比较；STID 另加约 8–24 小时 |

若必须把 STID 也纳入两数据集、三种子的完整主表，六种模型共 36 次，按每次 4 小时为约 144 小时，另加缓冲。时间不足时优先保证两数据集、直接结构基线和主消融；不要用多组低轮数训练替代同一规则下的完整训练。

执行顺序：PEMS08 上先跑 `full` 与 STIDGCN 种子 0，确认数据、指标与时间；再跑 PEMS08 三个消融；PEMS04 数据准备好后重复；最后按照预算补种子和 STID。若第一批发现输出异常、指标口径不一致或训练未收敛，暂停后续长跑并修正共同配置，已受影响的运行整组重做。

## 4. 命令示例与记录

在 `BasicTS` 目录运行，正式长跑应在第 2 节的测试隔离和结果保存设置完成之后：

```bash
PYTHONPATH=src python -m basicts.models.MambaTCNCA.train --dataset PEMS08 --variant full --mamba-backend torch --epochs 300 --batch-size 16 --seed 0 --gpus 0 --checkpoint-dir checkpoints/MambaTCNCADG/PEMS08_full_torch_s0
```

短检查命令：

```bash
PYTHONPATH=src python -m basicts.models.MambaTCNCA.train --dataset PEMS08 --variant full --steps 2 --batch-size 2 --seed 0 --gpus 0
```

每次运行使用独立检查点目录，保存完整配置、随机种子、逐轮训练与验证曲线、最佳验证轮次、最佳检查点、一次性测试指标、预测与真值、总耗时、单轮耗时、峰值显存和参数量。每次运行设置约 6 小时的故障告警阈值，而非到点自动判定失败；若硬件/数据不同，阈值应按短检查重估。

## 5. 结果分析与停止规则

先按预先确定的验证 MAE 选检查点，训练期间不根据测试结果选模型。正式测试后统一计算全 12 步及第 3、6、12 步 MAE、RMSE、MAPE。各变体相对 `full` 计算同数据集、同种子的配对差值，并报告绝对差和百分比；三种子结果给出均值、标准差及每个种子原值。若只做一个种子，直接标注“探索性”。如需区间估计，应按连续时间块而非相互重叠的单个窗口重采样。

以真实训练耗时、吞吐、峰值显存和推理时延与 MAE 同表呈现。若全模型没有稳定优于 STIDGCN 或 STID，也保留真实负结果和资源代价。`torch` 后端的速度结论仅适用于该后端；官方后端另列。

当前 CLI **没有去图融合变体**，而且树结构固定输入 12 步，`--output-len` 只改变预测长度。因此论文初稿中的“+图融合”逐级消融表，以及历史 24/48 步训练表，不能直接用现有四变体结果填充。若这些是论文必需主张，需先实现并验证独立开关/可变历史树，再额外排入训练预算；否则应同步修改论文中的对应研究问题、方法描述和空表。
