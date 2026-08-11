# Case 1消融分析




## 2. 核心指标消融表

| Variant | Training AUC ($10^4$) $\uparrow$ | Final reward $\uparrow$ | ISR (\%) $\uparrow$ | $E_{co\text{-}time}$ (s) $\downarrow$ | $E_n$ (g) $\downarrow$ | $E_t$ (s) $\downarrow$ | $E_{miss}$ (m) $\downarrow$ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full ART-MAPPO | 7.354 | 8.958 | 98.7 | 0.0505 | 0.0718 | 28.843 | 2.997 |
| w/o trust-aware mechanism | 6.037 | 8.683 | 97.2 | 0.0545 | 0.0739 | 29.132 | 3.087 |
| w/o GRU temporal encoder | 6.388 | 8.134 | 95.8 | 0.0732 | 0.0782 | 30.574 | 3.297 |
| w/o attention--residual backbone | 6.651 | 8.353 | 96.1 | 0.0606 | 0.0847 | 29.853 | 3.507 |



## 3. Trust-aware mechanism：训练效率贡献

移除Trust后，Training AUC由7.354降至6.037，下降17.9\%，是三个消融中最大的AUC降幅；Final reward仅由8.958降至8.683，下降3.1\%。与此同时，ISR只下降1.5个百分点，四个连续测试指标仅退化1\%--8\%。

这种“全过程AUC明显下降、末段回报和测试性能仅轻度下降”的差异说明，Trust的主要贡献是帮助策略在训练过程中更高效地获得有用经验，而不是单纯决定最终策略上限。该结论与论文中Trust用于动态调节探索与利用的理论定位一致。

在不使用Policy entropy的情况下，严谨表述应为：

> Trust-aware modulation primarily improves cumulative learning efficiency, as its removal causes the largest AUC reduction while producing only a modest change in final return and nominal interception metrics.



## 4. GRU temporal encoder：时序策略质量与协同时间贡献

移除GRU后，Final reward从8.958降至8.134，下降9.2\%，为三个消融中最低；ISR下降2.9个百分点，同样为三个消融中最大。最具区分度的是$E_{co\text{-}time}$由0.0505 s增至0.0732 s，增加45\%，而$E_t$增加6\%。

这些指标形成清晰的功能证据链：缺少历史状态编码后，策略仅依靠当前瞬时观测更难捕捉目标机动和各拦截器time-to-go的演化，最终表现为末段策略回报下降、协同时间误差增加和完成拦截的时间变长。

不使用Critic loss时，严谨表述应为：

> The GRU temporal encoder contributes primarily to final policy quality and cooperative timing: removing it yields the lowest final return and the largest increase in time-to-go mismatch among the three controls.



## 5. Attention--residual backbone：控制质量与末端精度贡献

移除attention--residual后，Training AUC下降9.6\%，Final reward下降6.7\%，属于中等幅度的训练退化。更关键的是，$E_n$由0.0718 g增至0.0847 g，增加18\%；$E_{miss}$由2.997 m增至3.507 m，增加17\%。这两项均是三个消融中的最大退化，而$E_t$仅增加3.5\%。

该非均匀变化说明，attention--residual backbone的主要作用不是单纯缩短拦截时间，而是从LOS、相对运动和time-to-go等耦合状态中提取控制相关特征，进而降低控制修正需求并提高末端精度。

严谨表述应为：

> The attention--residual backbone primarily supports control-relevant representation and terminal precision, as its removal produces the largest increases in control effort and miss distance.



## 6. 三个模块的独立贡献

| Module | 核心判别指标 | 主要退化 | 支撑的功能结论 |
|---|---|---|---|
| Trust-aware mechanism | Training AUC相对Final return的差异 | AUC下降17.9\%，Final return仅下降3.1\% | 提高整个训练阶段的学习效率 |
| GRU temporal encoder | Final return、ISR、$E_{co\text{-}time}$、$E_t$ | Final return下降9.2\%，$E_{co\text{-}time}$增加45\% | 利用时间历史改善最终策略和同步拦截 |
| Attention--residual backbone | $E_n$、$E_{miss}$ | 分别增加18\%和17\% | 改善控制特征表示和末端精度 |

三个变体并不是所有指标同时按相同比例变差，而是分别在与模块理论功能最相关的指标上出现最大退化。这种模块特异性比“Full在所有列都最好”的简单排序更能回答审稿人关于independent contribution的意见。


