# SCF-Mamba

Split-stream Cross-attention Fusion Mamba：SCI 奇偶分流、双向节点 Mamba，以及每层合并前的共享双向交叉注意力。对应既有 B/G5 模型，输入 `[B,96,N]` 和历史时间戳 `[B,96,2]`，输出归一化预测 `[B,H,N]`。

## 文件结构

```text
SCFMamba/
├── __init__.py                 # 导出 SCFMamba、SCFMambaConfig
├── config/scf_mamba_config.py   # BasicTSModelConfig
├── arch/
│   ├── scf_mamba_arch.py        # 模型入口
│   ├── sci.py                   # 分流、交互、逐层融合与预测头
│   ├── node_mamba.py            # 节点轴双向 Mamba
│   ├── cross_attention.py       # 奇偶序列交叉注意力
│   ├── embedding.py             # 日内和七日周期嵌入
│   └── torch_mamba.py           # CPU 检查后端
├── data.py                     # 训练集标准化、原始非零 mask、TaskFlow
├── train.py                    # BasicTS 配置、优化器和必要指标回调
├── utils/checkpoint.py          # 原生及旧 B 权重加载
└── LICENSE_SCINet              # SCI 来源许可
```

模型直接继承 `torch.nn.Module`，配置使用 BasicTS 的 `num_features` 表示节点数。不依赖 PIRMamba、消融包装类、模式文件或实验目录；不保留检索分支、其他注意力位置、实验诊断或远程调度代码。运行入口放在 BasicTS 的 `examples/forecasting/`，评估入口在 `scripts/forecasting/`，测试在 `tests/basicts_test/`。

## 使用

```python
from basicts.models.SCFMamba import SCFMamba, SCFMambaConfig

model = SCFMamba(SCFMambaConfig(num_features=307, output_len=24)).cuda()
prediction = model(inputs, inputs_timestamps)
```

`input_len=96`；`output_len` 可设为正整数，已验证 PEMS08/PEMS04 的 12、24、48、96 步。正式训练使用 `mamba_ssm` 的官方 CUDA 实现，`mamba_backend="torch"` 用于无 GPU 的结构和权重检查。时间戳第一列为日内位置/288，第二列沿用数据中的七日周期位置/7。

在 BasicTS 根目录执行：

```bash
PYTHONPATH=src python examples/forecasting/scf_mamba.py --print-config
PYTHONPATH=src python examples/forecasting/scf_mamba.py --dataset PEMS04 --horizon 24 --data-dir datasets/PEMS04
PYTHONPATH=src python scripts/forecasting/evaluate_scf_mamba.py --dataset PEMS04 --horizon 24 --checkpoint /path/to/SCFMamba_best_val_MAE_masked.pt --output metrics.json
PYTHONPATH=src python tests/basicts_test/test_scf_mamba.py -v
```

默认输入96、batch64、80轮、seed4321，FP32+TF32，AMP关闭。Adam学习率0.001、每轮乘0.95；主干权重衰减1e-5，注意力为0。采用按时间顺序预分割的数组，每个split独立stride1切窗；只用训练集逐节点均值/标准差归一化。训练是标准化空间masked MAE，验证和测试使用原尺度masked指标，mask始终取原始标签 `!=0`。全零目标batch跳过优化器更新。按验证masked MAE保存最佳权重，80轮后测试最佳模型。

BasicTS原生管理日志、配置、最佳/末轮权重、优化器及调度器。注意力独立dropout状态随原生模型state_dict保存。测试脚本完整累计误差和有效元素数，以 `sqrt(global MSE)` 计算RMSE，并输出全部H步聚合和逐步指标。CPU检查需显式加 `--gpus cpu --backend torch`；评估使用 `--device cpu --backend torch`。

## 已有权重

```python
from basicts.models.SCFMamba.utils import load_checkpoint
load_checkpoint(model, "/path/to/SCFMamba_best_val_MAE_masked.pt")
```

加载器严格转换原实验 B 模型的张量名称，并裁去注意力位置编码中未使用的末12项；其余张量数值、形状保持不变。旧文件按权重加载，不自动迁移旧实验优化器或随机状态。新原生检查点支持正常BasicTS恢复。

本次整理保留已训练模型的计算路径与预测结果。删除无效模块初始化、采用BasicTS原生采样后，从头训练的全局随机数消耗与旧实验快照不同，不保证训练过程逐位重现；历史实验源码、数据、权重和报告仍保存在 `training-runs/`。

SCI改写自SCINet（Apache-2.0）；节点Mamba和周期嵌入沿用本项目的已验证实现。迁移核对记录位于项目根目录 `reports/SCFMamba_BasicTS_Integration_20261013/`。
