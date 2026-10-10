# I-JEPA 从零实现练习

这套材料以 Assran 等人的 *Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture*（`../2301.08243v3.pdf`）为依据，结构参考 CS336 Assignment 1：先理解动机和张量约定，再逐件实现，最后训练、评估和做消融。

## 文件

- `ijepa_assignment.pdf`：适合顺序阅读的中文指导书。
- `ijepa_assignment.md`：指导书的可编辑源文。
- `tests/adapters.py`：测试与自己实现之间的接口；只在这里连接自己的代码。
- `tests/test_*.py`：逐阶段测试。**不要修改测试来使其通过。**
- `AGENTS.md`：给协助学习的 agent 的指导。

## 环境与运行

建议 Python 3.10+、PyTorch 2.x、pytest；测试只用 CPU，不需要下载数据。

```powershell
python -m pip install -e .
python -m pytest -q
python -m pytest -q tests/test_masking.py
python -m pytest -q tests/test_model.py
python -m pytest -q tests/test_objective.py
python -m pytest -q tests/test_training.py
```

刚开始时，测试中的 `NotImplementedError` 是预期结果。先在自己建立的 `ijepa_student/` 下实现，再让 `tests/adapters.py` 的各个函数**仅调用**自己的实现。测试给出的是功能契约，不规定你的模块划分。

## 复现的两层目标

1. **机制复现（本练习的必做部分）**：图像 patch、ViT、目标/上下文掩码、位置条件预测器、表征空间损失、EMA、训练与线性评估，全部能正确工作。可以用较小数据和模型完成。
2. **论文级结果（扩展目标）**：用论文附录的 ImageNet、模型规模、训练周期和算力配置比较数值。小模型通过测试或玩具数据损失下降，均不能声称复现了论文的 ImageNet 准确率。

详细要求、论文定位、问题与交付物见指导书。
