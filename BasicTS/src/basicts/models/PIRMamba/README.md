# PIR-Mamba

PEMS08 最终 V-12 模型的 BasicTS 实现：输入 96 步、输出 12 步、170 个节点，837,813 个参数。
本次为结构整理和命名重构，保留已有模型的计算、权重与实验协议。

## 文件结构

```text
PIRMamba/
├── __init__.py                  # 导出 PIRMamba、PIRMambaConfig
├── arch/
│   ├── pir_mamba_arch.py        # 模型入口与显式数据流
│   ├── sci.py                   # SCI 分解、交互、重组和预测投影
│   ├── mamba.py                 # 节点轴双向 Mamba 与门控残差
│   ├── embedding.py             # STID 风格周期时间嵌入
│   ├── attention.py             # 历史模式检索与时间交叉注意力
│   └── torch_mamba.py           # CPU 检查后端
├── config/pir_mamba_config.py   # BasicTSModelConfig
├── utils/checkpoint.py          # 权重加载与旧 V 权重名称迁移
├── assets/pattern_keys.npy      # 固定的训练集模式库
├── data.py                     # 保持 V 协议的数据集和 TaskFlow
├── train.py                    # BasicTSLauncher 训练配置
└── LICENSE_SCINet              # SCINet 原 Apache-2.0 许可
```

训练入口在 `examples/forecasting/pir_mamba_pems08.py`，评估入口在
`scripts/forecasting/evaluate_pir_mamba.py`，测试在 `tests/basicts_test/test_pir_mamba.py`。
权重放在 `checkpoints/PIRMamba/PEMS08_10_96_12/`；历史日志、旧版源码和核对记录均移出模型目录。

## 架构

历史流量经过 SCI 二层递归树；每个 SCI 交互块的四条路径共享一个轻量双向节点 Mamba。
节点嵌入与日内、七日周期嵌入融合后送入 Mamba，经门控残差回注 SCI，最后投影为 12 步预测。
另一条分支对历史的因果三步窗口检索固定模式库，得到延迟感知模式特征。
时间交叉注意力使用初始预测作为 Query、历史流量作为 Key、检索特征作为 Value，
通过门控残差修正预测；没有额外的模式特征节点矩阵注入。

```python
from basicts.models.PIRMamba import PIRMamba, PIRMambaConfig
from basicts.models.PIRMamba.utils import load_checkpoint

model = PIRMamba(PIRMambaConfig()).cuda().eval()
load_checkpoint(model)
prediction = model(inputs, inputs_timestamps)
# inputs: [B, 96, 170]，已归一化的流量
# inputs_timestamps: [B, 96, 2]，日内/七日周期位置
# prediction: [B, 12, 170]，归一化预测
```

时间索引沿用原实验：从最后一个历史时间戳计算 `round(time_of_day*288)` 和
`round(day_of_cycle*7)`。第二列是既有预处理定义的七日周期位置，未重新解释为真实星期标签。
不再支持把时间戳拼入流量末尾的旧包装输入；BasicTS 分别传入流量和时间戳。

## 运行

在 BasicTS 根目录、已安装 BasicTS 依赖的环境中执行：

```bash
# 查看配置，不启动训练
PYTHONPATH=src python examples/forecasting/pir_mamba_pems08.py --print-config

# 训练：96 -> 12、batch=64、10 轮、seed=4321
PYTHONPATH=src python examples/forecasting/pir_mamba_pems08.py --data-dir datasets/PEMS08 --epochs 10 --batch-size 64 --gpus 0

# 默认加载迁移后的最终 V 权重，评估完整测试集
PYTHONPATH=src python scripts/forecasting/evaluate_pir_mamba.py --data-dir datasets/PEMS08

# 无 GPU 的结构检查，使用显式 PyTorch 后端
PYTHONPATH=src python scripts/forecasting/evaluate_pir_mamba.py --device cpu --backend torch --max-windows 2

PYTHONPATH=src python tests/basicts_test/test_pir_mamba.py -v
```

正式实验使用官方 `mamba_ssm.Mamba` CUDA 后端。CPU 后端用于检查，不宣称与 CUDA 浮点结果逐值相同。
依赖沿用服务器当前已验证的 BasicTS/PyTorch/mamba-ssm 环境，不在模型目录维护另一套安装清单。

训练数据使用训练集均值/标准差归一化，各 split 独立以 stride=1 切窗。
训练损失为归一化流量的无掩码 L1；最佳权重按原尺度、无掩码验证 MAE 选择。
评估脚本按原协议屏蔽原尺度零值，全局累计 MAE/RMSE/MAPE，并报告 horizon 3、6、12。
这两个 MAE 口径用途不同，比较历史实验时应使用评估脚本。

默认权重文件为 `PIRMamba_best_val_MAE_unmasked.pt`，使用 BasicTS 的 `model_state_dict` 字段。
迁移只重命名张量，保留全部 186 个张量的数值、形状和精度。
此文件用于评估或以已有权重微调，不包含优化器/调度器状态，不能当作完整断点恢复文件。
旧 V `.bin` 仍可通过 `load_checkpoint(model, path)` 严格迁移加载。

## 来源与验证

模型来自既有 V-12 实验。SCI 结构改写自 SCINet
`02e6b0af2d58243de09aaa1eac3840237b659847`；季节 Mamba 分支来自既有 DST-Mamba 实验
（参考版本 `c0fbb62426f3da73f05f073d8b219ab4ad82345e`），时间嵌入参考 STID，
延迟感知模式检索沿用本项目已训练的 DFT 分支。改写说明：删除未启用分支与包装类，
将运行时 hook 改为显式调用，统一参数和模块命名，未更改最终 V 结构。

完整测试集 3,465 个窗口、7,068,600 个预测值与原 V 逐值一致；MAE 15.506606、
RMSE 24.581713、MAPE 10.044315%。核对脚本、JSON 和旧版来源记录位于项目根目录
`PIR-Mamba/scripts/`、`PIR-Mamba/verification/`、`PIR-Mamba/archive/`。
