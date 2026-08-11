# Partial-interceptor-failure trajectory candidates

本目录包含两个工况各5个真实固定策略推理回放。每个回放提供三维轨迹和水平面二维轨迹，共20个不同图件，并同时提供PNG（600 dpi）与PDF。所有回合均满足8个进攻飞行器被拦截。Case 1所选回合同时满足严格协同成功；Case 2来自资产安全且 $E_t\le45$ s 的成功子集，其中严格协同标志见 `trajectory_selection_and_validation.csv`，不得将普通拦截成功误写为严格协同成功。灰色叉号标记两架失效拦截器，颜色表示其分配目标组。

轨迹由同哈希策略权重、同哈希场景预设及原蒙特卡洛种子确定性重放得到；未训练模型，未修改轨迹坐标。`data/`保存逐回合NPZ和指标文件。
