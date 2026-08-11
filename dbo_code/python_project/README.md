# IDBO – 改进蜣螂优化算法在固定翼无人机任务分配中的应用

## 问题描述

20架固定翼无人机打击8个地面目标的任务分配问题。

### 代价函数考虑：
- **距离代价** (35%): 无人机到目标的欧氏距离
- **航向偏转代价** (20%): 当前航向与目标方位角的偏差
- **负载均衡代价** (25%): 各无人机任务分配的均衡性
- **威胁暴露代价** (20%): 飞行时间 × 威胁等级

### 算法对比
| 算法 | 说明 |
|------|------|
| **IDBO** | 改进DBO：伯努利混沌映射 + 可变螺旋 + Lévy飞行 + 非线性权重 |
| DBO | 原始蜣螂优化算法 |
| PSO | 粒子群优化 |
| GWO | 灰狼优化 |
| SSA | 麻雀搜索算法 |
| BOA | 蝴蝶优化算法 |

## 运行方法

```bash
conda activate assignment
python main.py --N 50 --iter 300 --runs 30
```

## 输出图表 (IEEE/Elsevier顶刊风格)

| 图表 | 说明 |
|------|------|
| `convergence.pdf` | 收敛曲线 |
| `boxplot.pdf` | 箱线图统计 |
| `bar_comparison.pdf` | 均值±标准差柱状图 |
| `violin.pdf` | 小提琴图 |
| `assignment_map.pdf` | IDBO最优分配方案地图 |
| `cost_breakdown.pdf` | 代价分量堆叠柱状图 |
| `radar_chart.pdf` | 多指标雷达图 |
| `time_comparison.pdf` | 计算时间对比 |
