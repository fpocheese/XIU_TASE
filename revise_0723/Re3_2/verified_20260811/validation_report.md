# 数据与交付核验报告

## 输入保护

- 高亮稿（只读）：`new_highlight/main.tex`
- 本次核验前后 SHA-256：`df47702ba9eac78bda5f3c727a5a050d01448e7693ae99b5e53d2d1ef3fafee6`
- IDBO 实现：`idbo_paper/idbo_paper.py`
- IDBO 实现 SHA-256：`e0e2bf23320b44ed5c8f66458ea6365f02fa5b17c4e0cae4d1a60b354b21699f`
- 结论：高亮稿未被修改。

## 运行环境

- CPU：Intel Core i7-12700H
- Python：3.10.12
- NumPy：1.24.2
- Matplotlib：3.7.1

## 自动检查

1. `benchmark_idbo_complexity.py` 已通过 `py_compile`。
2. 四份 CSV 均非空，所有数值字段均为有限值，不含 NaN 或 Inf。
3. 延迟、拓扑和蜂群规模三组实验的 `fixed_point_verified` 均为 `True`。
4. PDF 为单页矢量图，PNG 已实际查看；坐标、子图标号和图例均未截断或遮挡。
5. `reviewer_response_3_2.tex` 已通过独立 LaTeX 语法编译。
6. `manuscript_revision_suggestion.tex` 已通过独立 LaTeX 语法编译；独立测试包装器中的 `eq:complexity` 未定义警告是预期的，因为该标签只存在于完整论文中。

## 数据性质边界

- `local_idbo_runtime.csv` 由当前 `idbo_paper.py` 真实运行产生。
- 三份 consensus CSV 由独立的、确定性的 Top-$L_{\max}$ 邻居传播基准产生，并逐次对照全局固定点；它们不是完整交战仿真，也不包含强化学习训练或制导轨迹。
- 固定点不变的结论仅适用于静态分配快照、连通图、有限延迟和消息最终送达。在线时变环境中仍须保证共识时间短于重新分配周期。
