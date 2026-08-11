#!/usr/bin/env python3
"""
IEEE TASE Publication Quality Plots  V9
Multi-UAV Cooperative Interception — 每个工况×算法 = 一系列独立单幅图

V9 核心变化（相对 V8）：
  旧布局 (V8): 每个算法一张图 → [Evasive | Sinusoidal] (1×2 子图)
  新布局 (V9): 每个工况×算法 = 一幅独立的图, 全部单栏宽度 3.5"
               轨迹图、法向过载图等每个信息各一幅独立的图

  格式要求:
    ① 所有图均无标题 (no title)
    ② 图例宽度 = 单栏宽度, 横排铺满
    ③ 轨迹图带 Group 图例 + Defender/Attacker/Intercept
    ④ 法向过载图带 D-UAV 图例 + Limit 图例
    ⑤ 其余时间序列图带 D-UAV 图例

用法:
  conda run -n rlgpu python3 ieee_plot_v9_tase.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from pathlib import Path
from scipy.ndimage import uniform_filter1d
import warnings
warnings.filterwarnings('ignore')

# =====================================================================
#  IEEE TASE 格式
# =====================================================================
SINGLE_COL_WIDTH = 3.5       # inches  (单栏)
DOUBLE_COL_WIDTH = 7.16      # inches  (双栏)

plt.rcParams.update({
    'font.family':       'serif',
    'font.serif':        ['Times New Roman', 'DejaVu Serif'],
    'font.size':         10,
    'axes.labelsize':    11,
    'axes.titlesize':    11,
    'legend.fontsize':   8,
    'xtick.labelsize':   10,
    'ytick.labelsize':   10,
    'mathtext.fontset':  'stix',
    'lines.linewidth':   1.5,
    'lines.markersize':  5,
    'axes.linewidth':    0.8,
    'grid.linewidth':    0.3,
    'grid.linestyle':    '--',
    'grid.color':        '#CCCCCC',
    'grid.alpha':        0.5,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.direction':   'in',
    'ytick.direction':   'in',
    'legend.framealpha': 0.95,
    'legend.edgecolor':  '0.7',
    'legend.fancybox':   False,
    'legend.frameon':    True,
    'figure.dpi':        150,
    'savefig.dpi':       300,
    'savefig.bbox':      'tight',
    'savefig.pad_inches': 0.03,
    'text.usetex':       False,
})

# =====================================================================
#  颜色
# =====================================================================
# 用于轨迹图和组别划分的 8 种颜色
ACADEMIC_COLORS_8 = [
    '#0072B2',  # 1 深蓝
    '#D55E00',  # 2 深橙
    '#009E73',  # 3 翠绿
    '#CC3311',  # 4 砖红
    '#AA4499',  # 5 紫
    '#DDAA33',  # 6 金黄
    '#EE6677',  # 7 玫红
    '#88CCEE',  # 8 天蓝
]

# 新增：用于曲线图和 D-UAV 图例的 20 种独立高辨识度颜色 (基于经典 Tab20)
COLORS_20 = [
    '#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78',
    '#2ca02c', '#98df8a', '#d62728', '#ff9896',
    '#9467bd', '#c5b0d5', '#8c564b', '#c49c94',
    '#e377c2', '#f7b6d2', '#7f7f7f', '#c7c7c7',
    '#bcbd22', '#dbdb8d', '#17becf', '#9edae5'
]

COLOR_MAPPO = '#0072B2'
COLOR_PN    = '#D55E00'

MC_MAX_RUNS = 1000

# =====================================================================
#  工况显示名称
# =====================================================================
SCENARIO_DISP = {
    'nopn': 'Evasive',
    'sin':  'Sinusoidal',
}

# =====================================================================
#  局部放大区域配置
# =====================================================================
ZOOM_CFG = {
    'nopn': {
        'ny': {
            'xlim': (22, 30), 'ylim': (-0.4, 0.4),
            'loc': 'upper right', 'connect': (2, 3),
            'width': '38%', 'height': '38%',
        },
        'tgo': {
            'xlim': (20, 30), 'ylim': (-0.5, 3),
            'loc': 'upper right', 'connect': (2, 3),
            'width': '38%', 'height': '38%',
        },
        'tgo_error': {
            'xlim': (20, 30), 'ylim': (-0.5, 0.5),
            'loc': 'upper right', 'connect': (2, 3),
            'width': '38%', 'height': '38%',
        },
        'velocity': {
            'xlim': (0, 8), 'ylim': (20, 40),
            'loc': 'lower left', 'connect': (1, 4),
            'width': '38%', 'height': '35%',
        },
    },
    'sin': {
        'ny': {
            'xlim': (30, 38), 'ylim': (-0.4, 0.4),
            'loc': 'upper right', 'connect': (2, 3),
            'width': '38%', 'height': '38%',
        },
        'tgo': {
            'xlim': (28, 38), 'ylim': (-0.5, 3),
            'loc': 'upper right', 'connect': (2, 3),
            'width': '38%', 'height': '38%',
        },
        'tgo_error': {
            'xlim': (28, 38), 'ylim': (-0.5, 0.5),
            'loc': 'upper right', 'connect': (2, 3),
            'width': '38%', 'height': '38%',
        },
        'velocity': {
            'xlim': (0, 10), 'ylim': (20, 40),
            'loc': 'lower left', 'connect': (1, 4),
            'width': '38%', 'height': '35%',
        },
    },
}


# =====================================================================
#  数据加载器
# =====================================================================
class DataLoader:
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.dt = 0.05
        self.n_defenders  = 20
        self.n_attackers   = 8
        self.defender_x_cols    = list(range(0, 40, 2))
        self.defender_y_cols    = list(range(1, 40, 2))
        self.attacker_x_cols    = list(range(40, 56, 2))
        self.attacker_y_cols    = list(range(41, 56, 2))
        self.defender_nx_cols   = list(range(0, 40, 2))
        self.defender_ny_cols   = list(range(1, 40, 2))
        self.defender_tgo_cols  = list(range(0, 40, 2))
        self.defender_dist_cols = list(range(1, 40, 2))
        self.defender_V_cols    = list(range(0, 40, 2))
        self.defender_gamma_cols = list(range(1, 40, 2))

    def load_data(self, folder_name: str) -> dict:
        folder_path = self.base_path / folder_name
        data = {}
        for f in ['agentspos.txt', 'agentsall.txt',
                   'agentsvel.txt', 'agentstimetgo.txt']:
            fpath = folder_path / f
            if fpath.exists():
                with open(fpath, 'r') as fh:
                    lines = fh.readlines()
                max_cols = max(len(l.split()) for l in lines)
                arr = []
                for l in lines:
                    vals = [float(x) for x in l.split()]
                    while len(vals) < max_cols:
                        vals.append(vals[-1] if vals else 0)
                    arr.append(vals)
                data[f.replace('.txt', '')] = np.array(arr)
        return data

    def load_eval_data(self, eval_path: str, max_runs: int = MC_MAX_RUNS):
        p = Path(eval_path)
        if not p.exists():
            return None
        lines = p.read_text().strip().split('\n')
        arr = []
        for line in lines:
            sep = ',' if ',' in line else None
            vals = [float(x) for x in line.split(sep)]
            arr.append(vals)
        data = np.array(arr)
        valid = data[:, 0] <= 1.0
        data = data[valid]
        if len(data) > max_runs:
            data = data[:max_runs]
        return data

    def find_episode_end(self, pos_data):
        diff_d1 = np.abs(np.diff(pos_data[:, 0]))
        jump = np.where(diff_d1 > 100)[0]
        return jump[0] if len(jump) > 0 else pos_data.shape[0] - 1

    def find_repeat_start_rows(self, pos_data, episode_end):
        n, m = 0, episode_end + 1
        rsr = np.zeros(self.n_defenders, dtype=int)
        for i in range(self.n_defenders):
            col = self.defender_x_cols[i]
            diff = np.diff(pos_data[n:m, col])
            zi = np.where(diff == 0)[0]
            rsr[i] = (n + zi[0] + 1) if len(zi) > 0 else m - 1
        return rsr

    def get_plot_end_rows(self, rsr):
        return rsr - 2


class GroupedColors:
    def __init__(self, mapping=None):
        self.mapping = mapping or {d: d % 8 for d in range(20)}

    def get_defender_color(self, d):
        return ACADEMIC_COLORS_8[self.mapping.get(d, d % 8)]

    def get_attacker_color(self, a):
        return ACADEMIC_COLORS_8[a % 8]


def analyze_defender_target_mapping(loader, data, rsr):
    pos = data['agentspos']
    mapping = {}
    for d in range(loader.n_defenders):
        row = max(0, min(rsr[d] - 1, pos.shape[0] - 1))
        dx = pos[row, loader.defender_x_cols[d]]
        dy = pos[row, loader.defender_y_cols[d]]
        best_dist, best_a = float('inf'), 0
        for a in range(loader.n_attackers):
            ax_v = pos[row, loader.attacker_x_cols[a]]
            ay_v = pos[row, loader.attacker_y_cols[a]]
            dist = np.hypot(dx - ax_v, dy - ay_v)
            if dist < best_dist:
                best_dist, best_a = dist, a
        mapping[d] = best_a
    return mapping


def compute_impact_angles(loader, data, rsr, mapping):
    pos, vel = data['agentspos'], data['agentsvel']
    angles = {}
    for d in range(loader.n_defenders):
        row = max(1, min(rsr[d] - 1, pos.shape[0] - 2))
        a = mapping[d]
        dx = pos[row, loader.defender_x_cols[d]]
        dy = pos[row, loader.defender_y_cols[d]]
        ax_v = pos[row, loader.attacker_x_cols[a]]
        ay_v = pos[row, loader.attacker_y_cols[a]]
        los = np.arctan2(ay_v - dy, ax_v - dx)
        gamma_d = vel[row, loader.defender_gamma_cols[d]]
        imp = np.arctan2(np.sin(gamma_d - los), np.cos(gamma_d - los))
        angles[d] = {'impact_angle': imp, 'los_angle': los,
                      'gamma_d': gamma_d, 'target': a}
    return angles


def compute_statistics(loader, data, rsr, per, mapping):
    all_arr = data['agentsall']
    tgo_data = data['agentstimetgo']
    intercept_times = rsr * loader.dt
    final_ny, peak_ny = [], []
    for d in range(loader.n_defenders):
        end = max(1, per[d])
        ny = all_arr[:end, loader.defender_ny_cols[d]]
        if len(ny) > 0:
            peak_ny.append(np.max(np.abs(ny)))
            final_ny.append(np.mean(np.abs(ny[-min(10, len(ny)):])))
    tg = {}
    for d in range(loader.n_defenders):
        tg.setdefault(mapping[d], []).append(intercept_times[d])
    dt_vals = [max(v) - min(v) for v in tg.values()]
    final_dist = []
    for d in range(loader.n_defenders):
        end = max(1, per[d])
        dist = tgo_data[:end, loader.defender_dist_cols[d]]
        if len(dist) > 0:
            final_dist.append(dist[-1])
    return {
        'mean_final_ny':  np.mean(final_ny) if final_ny else 0,
        'peak_ny':        np.max(peak_ny)   if peak_ny else 0,
        'mean_peak_ny':   np.mean(peak_ny)  if peak_ny else 0,
        'delta_t_mean':   np.mean(dt_vals)  if dt_vals else 0,
        'delta_t_std':    np.std(dt_vals)    if dt_vals else 0,
        'delta_t_max':    np.max(dt_vals)    if dt_vals else 0,
        'mean_final_dist': np.mean(final_dist) if final_dist else 0,
    }


# =====================================================================
#  辅助函数
# =====================================================================
def _despine(ax):
    ax.spines['right'].set_visible(True)
    ax.spines['top'].set_visible(True)


def _save_fig(fig, out, stem):
    """统一保存 png (移除了 pdf 的保存)"""
    fig.savefig(out / f'{stem}.png')
    plt.close(fig)
    print(f"  ✓ {stem}")


def _add_zoom_inset_tgo(ax, prefix, plot_func_on_ax, xlim=None, ylim=None):
    """仅用于 tgo 图的末端局部放大，支持动态 x/y 范围"""
    cfg = ZOOM_CFG.get(prefix, {}).get('tgo', {})
    w = cfg.get('width', '38%')
    h = cfg.get('height', '35%')
    loc_val = cfg.get('loc', 'upper right')
    loc_map = {'upper right': 1, 'upper left': 2, 'lower left': 3,
               'lower right': 4, 'right': 5, 'center left': 6,
               'center right': 7, 'lower center': 8, 'upper center': 9,
               'center': 10}
    loc_num = loc_map.get(loc_val, 1) if isinstance(loc_val, str) else loc_val

    axins = inset_axes(ax, width=w, height=h, loc=loc_num, borderpad=0.8)
    plot_func_on_ax(axins)

    if xlim is None:
        xlim = cfg.get('xlim', ax.get_xlim())
    if ylim is None:
        ylim = cfg.get('ylim', ax.get_ylim())

    axins.set_xlim(xlim)
    axins.set_ylim(ylim)
    axins.tick_params(labelsize=6, direction='in')
    axins.grid(True, ls='--', lw=0.15, color='#CCC', alpha=0.4)
    for sp in axins.spines.values():
        sp.set_linewidth(0.6)
        sp.set_edgecolor('#666666')

    c1, c2 = cfg.get('connect', (2, 3))
    mark_inset(ax, axins, loc1=c1, loc2=c2,
               fc='none', ec='#666666', lw=0.6, ls='--')


def _legend_full_width(ax_or_fig, handles, labels, ncol, y_anchor=1.02,
                       fontsize=7, handlelength=1.5, columnspacing=0.8):
    """
    相对于绘图区（Axes）居中放置图例，避免因为纵坐标标签导致的视觉偏移。
    """
    if hasattr(ax_or_fig, 'get_axes') and len(ax_or_fig.axes) > 0:
        target_ax = ax_or_fig.axes[0]
    else:
        target_ax = ax_or_fig

    target_ax.legend(handles, labels, 
                     loc='lower center',             
                     bbox_to_anchor=(0.5, y_anchor), 
                     ncol=ncol, fontsize=fontsize, framealpha=0.95,
                     edgecolor='0.7', fancybox=False, frameon=True,
                     handlelength=handlelength,
                     columnspacing=columnspacing,
                     handletextpad=0.3,
                     borderpad=0.4)


# =====================================================================
#  图例构建器
# =====================================================================
def _build_group_legend(mapping):
    """
    轨迹图图例: Group A1 ~ A8 (按颜色) + Defender + Attacker + 区分起点的图例 + Intercept
    """
    handles, labels = [], []
    for t_id in sorted(set(mapping.values())):
        color = ACADEMIC_COLORS_8[t_id]
        handles.append(Line2D([0], [0], color=color, lw=2.0, ls='-'))
        labels.append(f'Group $A_{{{t_id+1}}}$')
        
    handles.append(Line2D([0], [0], color='gray', lw=1.5, ls='-'))
    labels.append('Defender')
    handles.append(Line2D([0], [0], color='gray', lw=1.5, ls='--'))
    labels.append('Attacker')
    
    # 防守方起点 (圆圈)
    handles.append(Line2D([0], [0], marker='o', color='w',
                          markerfacecolor='white', markeredgecolor='gray',
                          markersize=5, ls='None'))
    labels.append('Def Start')

    # 进攻方起点 (方块)
    handles.append(Line2D([0], [0], marker='s', color='w',
                          markerfacecolor='gray', markeredgecolor='gray',
                          markersize=5, ls='None'))
    labels.append('Att Start')

    # 拦截点
    handles.append(Line2D([0], [0], marker='*', color='w',
                          markerfacecolor='red', markeredgecolor='darkred',
                          markersize=8, ls='None'))
    labels.append('Intercept')
    
    return handles, labels


def _build_duav_legend(mapping, extra_handles=None, extra_labels=None):
    """
    D-UAV 图例: 强制使用 20 种不同的颜色
    """
    handles, labels = [], []
    for d in range(20):
        # 修改点：不再按 mapping 获取组别颜色，而是分配独立颜色
        color = COLORS_20[d % 20]
        h = Line2D([0], [0], color=color, linewidth=1.5)
        handles.append(h)
        labels.append(f'$D_{{UAV_{{{d+1}}}}}$')
        
    if extra_handles and extra_labels:
        handles.extend(extra_handles)
        labels.extend(extra_labels)
    return handles, labels


# =====================================================================
#  V9 绘图: 每个工况×算法 = 一幅独立的图
#  figsize = (SINGLE_COL_WIDTH, height)  单栏, 无标题
# =====================================================================

# ---- 轨迹图 (单幅) ----
def plot_trajectory_single(loader, res, prefix, method_name, out):
    """
    一个工况+一个算法 → 一幅独立轨迹图
    大图最大化：增加整体高度容纳图例，但绘图区域（正方形）占满整个宽度
    """
    # 关键修改1：增加整体高度，给顶部图例留足够空间
    # 宽度保持 3.5"，高度设为 4.2"（比原来 3.15" 大很多）
    total_height = SINGLE_COL_WIDTH * 1.2  # 4.2 英寸
    
    fig, ax = plt.subplots(figsize=(SINGLE_COL_WIDTH, total_height))

    pos = res['data']['agentspos']
    gc = GroupedColors(res['mapping'])

    for d in range(loader.n_defenders):
        end = res['repeat_start_rows'][d]
        x = pos[:end, loader.defender_x_cols[d]]
        y = pos[:end, loader.defender_y_cols[d]]
        ax.plot(x, y, color=gc.get_defender_color(d), lw=1.2, alpha=0.85)
        ax.plot(x[0], y[0], 'o', color=gc.get_defender_color(d), ms=3, mfc='white', mew=0.6, zorder=5)
        ax.plot(x[-1], y[-1], '*', color='red', ms=8, mec='darkred', mew=0.4, zorder=10)

    for a in range(loader.n_attackers):
        ep = res['episode_end']
        x = pos[:ep+1, loader.attacker_x_cols[a]]
        y = pos[:ep+1, loader.attacker_y_cols[a]]
        ax.plot(x, y, '--', color=gc.get_attacker_color(a), lw=1.3, alpha=0.9)
        if len(x) > 0:
            ax.plot(x[0], y[0], 's', color=gc.get_attacker_color(a), ms=4, zorder=5)

    ax.set_xlabel('$x$ (m)')
    ax.set_ylabel('$y$ (m)')
    ax.set_aspect('equal')  # 保持正方形比例
    ax.grid(True, ls='--', lw=0.3, color='#CCCCCC', alpha=0.5)
    _despine(ax)

    # 生成图例（放在绘图区域上方）
    handles, labels = _build_group_legend(res['mapping'])
    
    _legend_full_width(ax, handles, labels,
                       ncol=5, 
                       y_anchor=1.02,  # 图例在绘图区上方
                       fontsize=7, 
                       handlelength=1.5, 
                       columnspacing=0.8)

    # 关键修改2：调整边距，让正方形绘图区域最大化
    # 减小 bottom 和 left/right，让正方形占满宽度
    # 减小 top，让图例位于绘图区上方但不挤压绘图区
    
    # 计算合适的边距：让正方形轨迹图占满整个宽度
    # SINGLE_COL_WIDTH = 3.5"
    # 左右边距 0.15"  each -> 绘图区宽度 = 3.2"
    # 底部边距 0.15" -> 给 xlabel 留空间
    # 顶部边距 0.75" -> 给图例留空间（3行图例需要约 0.7"）
    
    fig.subplots_adjust(
        left=0.05,      # 左边距减小，让图更宽
        right=0.98,     # 右边距减小
        bottom=0.08,    # 底部边距（给xlabel）
        top=0.72        # 顶部边距：图例在 0.72-0.98 之间，绘图区在 0-0.72
    )

    stem = f'{method_name.lower()}_{prefix}_trajectory'
    _save_fig(fig, out, stem)




# def plot_trajectory_single(loader, res, prefix, method_name, out):
#     """
#     一个工况+一个算法 → 一幅独立轨迹图
#     巧妙实现：图例框撑满单栏宽度，但内部图例项保持紧凑居中
#     """
#     total_height = SINGLE_COL_WIDTH * 1.2  # 增加整体高度，给顶部图例留出空间
#     fig, ax = plt.subplots(figsize=(SINGLE_COL_WIDTH, total_height))

#     pos = res['data']['agentspos']
#     gc = GroupedColors(res['mapping'])

#     for d in range(loader.n_defenders):
#         end = res['repeat_start_rows'][d]
#         x = pos[:end, loader.defender_x_cols[d]]
#         y = pos[:end, loader.defender_y_cols[d]]
#         ax.plot(x, y, color=gc.get_defender_color(d), lw=1.2, alpha=0.85)
#         ax.plot(x[0], y[0], 'o', color=gc.get_defender_color(d), ms=3, mfc='white', mew=0.6, zorder=5)
#         ax.plot(x[-1], y[-1], '*', color='red', ms=8, mec='darkred', mew=0.4, zorder=10)

#     for a in range(loader.n_attackers):
#         ep = res['episode_end']
#         x = pos[:ep+1, loader.attacker_x_cols[a]]
#         y = pos[:ep+1, loader.attacker_y_cols[a]]
#         ax.plot(x, y, '--', color=gc.get_attacker_color(a), lw=1.3, alpha=0.9)
#         if len(x) > 0:
#             ax.plot(x[0], y[0], 's', color=gc.get_attacker_color(a), ms=4, zorder=5)

#     ax.set_xlabel('$x$ (m)')
#     ax.set_ylabel('$y$ (m)')
#     ax.set_aspect('equal')
#     ax.grid(True, ls='--', lw=0.3, color='#CCCCCC', alpha=0.5)
#     _despine(ax)

#     # 调整绘图区的上下左右边距，把上方位置腾出来给图例框
#     fig.subplots_adjust(left=0.08, right=0.95, bottom=0.08, top=0.75) 

#     # ================= 核心修改区 =================
#     # 在图片顶部新建一个 Axes（画图区域）作为图例的外框
#     # left=0.02, width=0.96 意味着它占据了单栏宽度的 96%（近乎撑满，留一点边防裁剪）
#     # leg_ax = fig.add_axes([0.02, 0.78, 0.96, 0.20]) 
#     leg_ax = fig.add_axes([0.0, 0.78, 1.0, 0.20])
#     leg_ax.set_xticks([]) # 隐藏坐标刻度
#     leg_ax.set_yticks([])
#     leg_ax.patch.set_facecolor('white')
#     leg_ax.patch.set_alpha(0.95)
#     # 给这个框画上灰色边框，伪装成图例框
#     for sp in leg_ax.spines.values():
#         sp.set_color('0.7')
#         sp.set_linewidth(1.0)

#     handles, labels = _build_group_legend(res['mapping'])
    
#     # 将图例画在 leg_ax 中心，并关闭它自带的边框（frameon=False）
#     leg_ax.legend(handles, labels, 
#                   loc='center', 
#                   ncol=5, 
#                   fontsize=7, 
#                   frameon=False,        # <--- 关键：关闭自带边框
#                   handlelength=1.5, 
#                   columnspacing=0.8,    # <--- 保持紧凑的间距
#                   handletextpad=0.3)
#     # ==============================================

#     stem = f'{method_name.lower()}_{prefix}_trajectory'
#     _save_fig(fig, out, stem)



# ---- 通用时间序列 (单幅) ----
def _plot_timeseries_single(loader, res, prefix, method_name, out,
                            data_key, col_getter, ylabel, metric_name,
                            smooth=1, ylim=None, hlines=None,
                            line_end_labels=None,
                            show_legend=False,
                            legend_ncol=10, legend_fontsize=5.5):
    """
    一个工况+一个算法 → 一幅独立的时间序列图
    extra_legend_items: list of (handle, label) 追加到 D-UAV 图例后面
    """
    fig, ax = plt.subplots(figsize=(SINGLE_COL_WIDTH,
                                    SINGLE_COL_WIDTH * 0.52))

    arr = res['data'][data_key]

    for d in range(loader.n_defenders):
        end = max(1, res['plot_end_rows'][d])
        y_data = arr[:end, col_getter(d)]
        if smooth > 1:
            y_data = uniform_filter1d(y_data, size=smooth, mode='nearest')
        t = np.arange(len(y_data)) * loader.dt
        # 修改点：不再使用组别颜色，强制分配 20 种独立颜色
        ax.plot(t, y_data, color=COLORS_20[d % 20],
                lw=0.8, alpha=0.8, zorder=3)

    if hlines:
        for yv, clr, ls_ in hlines:
            ax.axhline(y=yv, color=clr, linestyle=ls_, lw=0.9, alpha=0.7)

    if line_end_labels and hlines:
        x_right = max(ax.get_xlim()[1], 1e-6)
        x_text = x_right * 0.985
        for item in line_end_labels:
            yv = item[0]
            txt = item[1]
            # 如果提供了第三个参数，就用它；没提供就默认用 'bottom' (字在线上方)
            valign = item[2] if len(item) > 2 else 'bottom' 
            ax.text(x_text, yv, txt, color='r', fontsize=10, # 字号保持你修改后的大小
                    ha='right', va=valign, clip_on=True)

    ax.set_xlabel('$t$ (s)')
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(ylim)
    ax.set_xlim(left=0)
    ax.grid(True, ls='--', lw=0.2, color='#CCC', alpha=0.5)
    _despine(ax)

    if show_legend:
        handles, labels = _build_duav_legend(res['mapping'])
        _legend_full_width(ax, handles, labels,
                           ncol=legend_ncol, y_anchor=1.02,
                           fontsize=legend_fontsize,
                           handlelength=1.2, columnspacing=0.5)
        fig.subplots_adjust(top=0.80, bottom=0.14, left=0.14, right=0.99)
    else:
        fig.subplots_adjust(top=0.98, bottom=0.14, left=0.14, right=0.99)

    stem = f'{method_name.lower()}_{prefix}_{metric_name}'
    _save_fig(fig, out, stem)


# # ---- 各量快捷调用 ----
# def plot_ny_single(loader, res, prefix, method, out):
#     _plot_timeseries_single(
#         loader, res, prefix, method, out,
#         'agentsall', lambda d: loader.defender_ny_cols[d],
#         '$n_y$ (g)', 'ny', smooth=10, ylim=(-1.5, 1.5),
#         hlines=[(1, 'r', '--'), (-1, 'r', '--')],
#         line_end_labels=[(1, 'Limit 1g'), (-1, 'Limit -1g')],
#         show_legend=True)



# ---- 法向过载图 (单幅) ----
def plot_ny_single(loader, res, prefix, method, out):
    """
    修改点：去除单独的图例和高度设置，统一使用基础的时间序列绘图函数，
    确保实际绘图区域大小与其他曲线图(nx, velocity等)100%一致。
    """
    _plot_timeseries_single(
        loader, res, prefix, method, out,
        'agentsall', lambda d: loader.defender_ny_cols[d],
        '$n_y$ (g)', 'ny', smooth=10, ylim=(-1.5, 1.5),
        hlines=[(1, 'r', '--'), (-1, 'r', '--')],
        line_end_labels=[(1, 'Normal Acceleration Limit', 'bottom'), (-1, 'Normal Acceleration Limit', 'top')],
        show_legend=False)



# # ---- 独立图例生成 (单幅) ----
# def plot_standalone_duav_legend(out):
#     """
#     单独生成 D-UAV (20机) 的独立图例图片，供 LaTeX 中统一引用。
#     高度设为 0.55 英寸，恰好容纳两行图例。
#     """
#     fig, ax = plt.subplots(figsize=(SINGLE_COL_WIDTH, 0.55))
#     ax.axis('off')  # 隐藏所有的坐标轴、边框和背景
    
#     # 获取 20 种颜色的 handles 和 labels
#     handles, labels = _build_duav_legend(mapping=None)
    
#     # === 核心修改区：添加 expand 强制撑满 3.5 英寸宽度 ===
#     ax.legend(handles, labels, 
#               loc='center',
#               bbox_to_anchor=(0.0, 0.0, 1.0, 1.0), # 绑定整个坐标轴的宽与高
#               mode='expand',                       # 强制横向拉伸列间距，使其撑满 bbox
#               borderaxespad=0.0,                   # 去除边缘自带的 padding
#               ncol=5,              
#               fontsize=7,          
#               framealpha=0.95,
#               edgecolor='0.7',
#               fancybox=False,
#               frameon=True,
#               handlelength=1.5,                    # 线条稍微画长一点更好看
#               columnspacing=0.5,                   # 开启 expand 后，系统会自动计算间距，这里写小一点作为保底
#               handletextpad=0.3,
#               borderpad=0.4)

#     # 取消原有的边距限制，让 bbox 能完完全全接触到图片边缘
#     fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
    
#     _save_fig(fig, out, 'standalone_duav_legend')


def plot_standalone_duav_legend(out):
    """
    单独生成 D-UAV (20机) 的独立图例图片。
    巧妙实现：背景框撑满单栏宽度，内部项紧凑居中排列。
    """
    fig, ax = plt.subplots(figsize=(SINGLE_COL_WIDTH, 0.65))
    
    # 把 Axes 伪装成图例的外框
    ax.set_xticks([])
    ax.set_yticks([])
    ax.patch.set_facecolor('white')
    ax.patch.set_alpha(0.95)
    for sp in ax.spines.values():
        sp.set_color('0.7')
        sp.set_linewidth(1.0)
        
    handles, labels = _build_duav_legend(mapping=None)
    
    # 图例自身不带边框，自然居中
    ax.legend(handles, labels, 
              loc='center',
              ncol=5,              
              fontsize=7,          
              frameon=False,       # <--- 关键：关闭自带边框
              handlelength=1.5,    # <--- 和上方保持相同的紧凑间距
              columnspacing=0.8,
              handletextpad=0.3)

    # 将 Axes 拉伸到几乎占满整个画布（左右各留 2% 边距防止边框被图片切掉）
    fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.05)
    
    _save_fig(fig, out, 'standalone_duav_legend')


def plot_nx_single(loader, res, prefix, method, out):
    _plot_timeseries_single(
        loader, res, prefix, method, out,
        'agentsall', lambda d: loader.defender_nx_cols[d],
        '$n_x$ (g)', 'nx', smooth=10, ylim=(-0.3, 1.3),
        hlines=[(1.0, 'r', '--'), (-0.1, 'r', '--')],
        line_end_labels=[(1.0, 'Axial Acceleration Limit', 'bottom'), (-0.1, 'Axial Acceleration Limit', 'top')],
    show_legend=False)


def plot_velocity_single(loader, res, prefix, method, out):
    _plot_timeseries_single(
        loader, res, prefix, method, out,
        'agentsvel', lambda d: loader.defender_V_cols[d],
        '$V_D$ (m/s)', 'velocity', ylim=(15, 45),
        hlines=[(40, 'r', '--')],
        line_end_labels=[(40, 'Velocity Limit', 'bottom')],
        show_legend=False)


def plot_heading_single(loader, res, prefix, method, out):
    _plot_timeseries_single(
        loader, res, prefix, method, out,
        'agentsvel', lambda d: loader.defender_gamma_cols[d],
    '$\\gamma_D$ (rad)', 'heading',
    show_legend=False)


def plot_tgo_single(loader, res, prefix, method, out):
    fig, ax = plt.subplots(figsize=(SINGLE_COL_WIDTH,
                                    SINGLE_COL_WIDTH * 0.52))
    arr = res['data']['agentstimetgo']

    for d in range(loader.n_defenders):
        end = max(1, res['plot_end_rows'][d])
        y_data = arr[:end, loader.defender_tgo_cols[d]]
        t = np.arange(len(y_data)) * loader.dt
        # 修改点：使用 20 种独立颜色
        ax.plot(t, y_data, color=COLORS_20[d % 20],
                lw=0.8, alpha=0.8, zorder=3)

    ax.set_xlabel('$t$ (s)')
    ax.set_ylabel('$t_{go}$ (s)')
    ax.set_xlim(left=0)
    ax.grid(True, ls='--', lw=0.2, color='#CCC', alpha=0.5)
    _despine(ax)

    # 仅 tgo 恢复局部放大：突出末端同组同时收敛到 0
    def _draw_tgo_inset(target_ax):
        for dd in range(loader.n_defenders):
            _end = max(1, res['plot_end_rows'][dd])
            yd = arr[:_end, loader.defender_tgo_cols[dd]]
            tt = np.arange(len(yd)) * loader.dt
            # 修改点：局部放大图也使用 20 种独立颜色
            target_ax.plot(tt, yd, color=COLORS_20[dd % 20],
                           lw=0.6, alpha=0.85)

    # 动态计算末端放大范围
    mapping = res['mapping']
    group_tail_t = []
    group_tail_y = []
    for t_id in sorted(set(mapping.values())):
        defs = [d for d, tid in mapping.items() if tid == t_id]
        if not defs:
            continue

        min_len = min(max(2, res['plot_end_rows'][d]) for d in defs)
        t_vec = np.arange(min_len) * loader.dt
        if len(t_vec) < 2:
            continue

        total_t = t_vec[-1]
        tail_sec = min(12.0, max(4.0, 0.20 * total_t))
        t_start = max(0.0, total_t - tail_sec)
        i0 = np.searchsorted(t_vec, t_start, side='left')
        i0 = min(max(i0, 0), len(t_vec) - 1)

        group_tail_t.append((t_vec[i0], t_vec[-1]))
        for d in defs:
            y_vec = arr[:min_len, loader.defender_tgo_cols[d]]
            group_tail_y.append(y_vec[i0:])

    dyn_xlim, dyn_ylim = None, None
    if group_tail_t and group_tail_y:
        x_min = min(a for a, _ in group_tail_t)
        x_max = max(b for _, b in group_tail_t)
        if x_max - x_min < 2.0:
            mid = 0.5 * (x_min + x_max)
            x_min, x_max = mid - 1.0, mid + 1.0

        dyn_xlim = (x_min+3.0, x_max+1.0)
        dyn_ylim = (-0.1, 0.5)

    _add_zoom_inset_tgo(ax, prefix, _draw_tgo_inset, xlim=dyn_xlim, ylim=dyn_ylim)

    fig.subplots_adjust(top=0.98, bottom=0.14, left=0.14, right=0.99)
    stem = f'{method.lower()}_{prefix}_tgo'
    _save_fig(fig, out, stem)


def plot_distance_single(loader, res, prefix, method, out):
    _plot_timeseries_single(
        loader, res, prefix, method, out,
        'agentstimetgo', lambda d: loader.defender_dist_cols[d],
    'Range-to-go $D$ (m)', 'distance',
    show_legend=False)


# ---- tgo 误差 (单幅) ----
def plot_tgo_error_single(loader, res, prefix, method_name, out):
    fig, ax = plt.subplots(figsize=(SINGLE_COL_WIDTH,
                                    SINGLE_COL_WIDTH * 0.52))
    tgo_data = res['data']['agentstimetgo']
    mapping  = res['mapping']

    def _draw_tgo_error(target_ax):
        for t_id in sorted(set(mapping.values())):
            defs = [d for d, tid in mapping.items() if tid == t_id]
            if len(defs) < 2:
                continue
            min_len = min(max(1, res['plot_end_rows'][d]) for d in defs)
            tgo_all = np.column_stack(
                [tgo_data[:min_len, loader.defender_tgo_cols[d]] for d in defs])
            mean_tgo = tgo_all.mean(axis=1)
            for i, d in enumerate(defs):
                err = tgo_all[:, i] - mean_tgo
                t = np.arange(min_len) * loader.dt
                # 修改点：使用 20 种独立颜色
                target_ax.plot(t, err, color=COLORS_20[d % 20],
                               lw=0.8, alpha=0.8, zorder=3)
        target_ax.axhline(y=0, color='k', ls='-', lw=0.4, alpha=0.3)

    _draw_tgo_error(ax)
    ax.set_xlabel('$t$ (s)')
    ax.set_ylabel('$t_{go}$ error (s)')
    ax.set_xlim(left=0)
    ax.grid(True, ls='--', lw=0.2, color='#CCC', alpha=0.5)
    _despine(ax)

    # 该图不显示图例，减少重复
    fig.subplots_adjust(top=0.98, bottom=0.14, left=0.14, right=0.99)

    stem = f'{method_name.lower()}_{prefix}_tgo_error'
    _save_fig(fig, out, stem)


# ---- 时间同步柱状图 (单幅) ----
def plot_time_sync_single(loader, res, prefix, method_name, out):
    fig, ax = plt.subplots(figsize=(SINGLE_COL_WIDTH,
                                    SINGLE_COL_WIDTH * 0.55))
    intercept_t = res['repeat_start_rows'] * loader.dt
    mapping = res['mapping']
    tg = {}
    for d, tid in mapping.items():
        tg.setdefault(tid, []).append(intercept_t[d])
    targets = sorted(tg.keys())
    dt_vals = [max(tg[t]) - min(tg[t]) for t in targets]
    # 柱状图按目标组别上色（维持原逻辑，因为柱子代表的是组别）
    colors_list = [ACADEMIC_COLORS_8[t] for t in targets]
    bars = ax.bar([f'$A_{{{t+1}}}$' for t in targets], dt_vals,
                  color=colors_list, edgecolor='black', lw=0.5, alpha=0.85)
    for bar, dv in zip(bars, dt_vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f'{dv:.2f}', ha='center', va='bottom',
                fontsize=6, fontweight='bold')
    ax.set_xlabel('Target')
    ax.set_ylabel('$\\Delta t$ (s)')
    ax.set_ylim(0, max(max(dt_vals) * 1.4, 0.15))
    ax.grid(True, axis='y', ls='--', lw=0.2, color='#CCC', alpha=0.5)
    _despine(ax)
    plt.tight_layout()

    stem = f'{method_name.lower()}_{prefix}_time_sync'
    _save_fig(fig, out, stem)


# =====================================================================
#  Monte Carlo — 每个工况×算法 单独出图
# =====================================================================
EVAL_COL_LABELS = [
    '$E_{co-time}$ (s)',
    '$E_{n}$ (g)',
    '$E_{miss}$ (m)',
    'Energy consumption',
    '$E_{t}$ (s)',
]
MC_SHOW_COLS = [0, 1, 2]
MC_BOX_COLS  = [0, 1, 2, 3]


def plot_mc_scatter_single(eval_data, prefix, method_name, out):
    """每个指标一张独立散点图"""
    if eval_data is None:
        print(f"  ⚠ {method_name.lower()}_{prefix}_mc_scatter: eval data missing")
        return
    for col in MC_SHOW_COLS:
        fig, ax = plt.subplots(figsize=(SINGLE_COL_WIDTH,
                                        SINGLE_COL_WIDTH * 0.65))
        color = COLOR_MAPPO if 'mappo' in method_name.lower() else COLOR_PN
        lbl = SCENARIO_DISP.get(prefix, prefix)
        ax.scatter(np.arange(len(eval_data)), eval_data[:, col],
                   s=8, color=color, alpha=0.45, label=lbl, zorder=2)
        mean_val = np.mean(eval_data[:, col])
        ax.axhline(mean_val, color=color, ls='--', lw=1.2, alpha=0.9,
                   label=f'Mean={mean_val:.4f}')
        ax.set_xlabel('Monte Carlo runs')
        ax.set_ylabel(EVAL_COL_LABELS[col])
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(True, ls='--', lw=0.2, color='#CCC', alpha=0.5)
        _despine(ax)
        plt.tight_layout()
        stem = f'{method_name.lower()}_{prefix}_mc_scatter_{col+1}'
        _save_fig(fig, out, stem)
    print(f"  ✓ {method_name.lower()}_{prefix}_mc_scatter ({len(MC_SHOW_COLS)} figs)")


def plot_mc_boxplot_compare(m_eval, p_eval, prefix, out):
    """ART-MAPPO vs PN 箱线对比图 — 4 指标横排 (仿 v6 风格)"""
    if m_eval is None or p_eval is None:
        print(f"  ⚠ {prefix}_mc_boxplot_compare: eval data missing, skip")
        return
    sel_cols   = [0, 1, 2, 4]   # co-time, n, miss, Et (打击时间)
    sel_labels = [EVAL_COL_LABELS[c] for c in sel_cols]

    fig, axes = plt.subplots(1, len(sel_cols),
                             figsize=(DOUBLE_COL_WIDTH, SINGLE_COL_WIDTH * 0.6))
    for i, (col, lab) in enumerate(zip(sel_cols, sel_labels)):
        ax = axes[i]
        bp = ax.boxplot(
            [m_eval[:, col], p_eval[:, col]],
            labels=['ART-MAPPO', 'PN'],
            widths=0.5, patch_artist=True,
            medianprops=dict(color='black', lw=1.5),
            flierprops=dict(marker='o', markersize=2, alpha=0.3))
        bp['boxes'][0].set_facecolor(COLOR_MAPPO)
        bp['boxes'][0].set_alpha(0.6)
        bp['boxes'][1].set_facecolor(COLOR_PN)
        bp['boxes'][1].set_alpha(0.6)
        ax.set_ylabel(lab, fontsize=10)
        ax.set_xticklabels(['ART-MAPPO', 'PN'], fontsize=8) 
        ax.grid(True, axis='y', ls='--', lw=0.2, color='#CCC', alpha=0.5)
        _despine(ax)
        # case1 (nopn) 的 co-time 坐标范围固定为 0.0~0.3
        if prefix == 'nopn' and col == 0:
            ax.set_ylim(0.0, 0.3)
    plt.tight_layout()
    stem = f'{prefix}_mc_boxplot_compare'
    _save_fig(fig, out, stem)


# =====================================================================
#  统计表格
# =====================================================================
def generate_statistics_table(all_stats, all_eval_stats, output_path):
    fig, ax = plt.subplots(figsize=(DOUBLE_COL_WIDTH, 2.8))
    ax.axis('off')

    col_labels = [
        'Method', 'Maneuver',
        'Peak $|n_y|$\n(g)',
        'Mean $|n_y|_{end}$\n(g)',
        '$\\bar{\\Delta t}$\n(s)',
        'Max $\\Delta t$\n(s)',
        'MC $\\bar{E}_{co}$\n(s)',
        'MC $\\bar{E}_{n}$\n(g)',
        'MC $\\bar{E}_{miss}$\n(m)',
    ]

    maneuver_disp = {'nopn': 'Evasive', 'sin': 'Sinusoidal'}
    method_disp   = {'mappo': 'ART-MAPPO (RL)', 'pn': 'PN (baseline)'}

    table_data = []
    for key in sorted(all_stats.keys()):
        parts = key.split('_')
        method, maneuver = parts[0], parts[-1]
        s  = all_stats[key]
        es = all_eval_stats.get(key, {})
        table_data.append([
            method_disp.get(method, method),
            maneuver_disp.get(maneuver, maneuver),
            f'{s["peak_ny"]:.4f}',
            f'{s["mean_final_ny"]:.4f}',
            f'{s["delta_t_mean"]:.3f}',
            f'{s["delta_t_max"]:.3f}',
            f'{es.get("mc_co_time", 0):.4f}',
            f'{es.get("mc_n", 0):.4f}',
            f'{es.get("mc_miss", 0):.4f}',
        ])

    table = ax.table(cellText=table_data, colLabels=col_labels,
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.0, 1.5)
    for j in range(len(col_labels)):
        table[0, j].set_facecolor('#D4E6F1')
        table[0, j].set_text_props(fontweight='bold', fontsize=7)
    for i, row in enumerate(table_data):
        for j in range(len(col_labels)):
            cell = table[i + 1, j]
            cell.set_facecolor('#EBF5FB' if 'ART-MAPPO' in row[0] else '#F8F9F9')
    plt.tight_layout()
    # fig.savefig(output_path / 'statistics_table_v9.pdf')
    fig.savefig(output_path / 'statistics_table_v9.png')
    plt.close(fig)
    print(f"  ✓ statistics_table_v9")

    print("\n" + "=" * 110)
    print("  STATISTICAL COMPARISON TABLE (V9)")
    print("=" * 110)
    hdr = "  ".join(f'{c:<14}' for c in
        ['Method', 'Maneuver', 'PeakNy', 'FinalNy',
         'ΔtMean', 'ΔtMax', 'MC co-t', 'MC n', 'MC miss'])
    print(hdr)
    print("-" * 110)
    for r in table_data:
        print("  ".join(f'{c:<14}' for c in r))
    print("=" * 110)


# =====================================================================
#  数据集预处理
# =====================================================================
def process_single_dataset(loader, folder_name):
    data = loader.load_data(folder_name)
    if not data:
        return None
    pos = data['agentspos']
    episode_end = loader.find_episode_end(pos)
    rsr = loader.find_repeat_start_rows(pos, episode_end)
    per = loader.get_plot_end_rows(rsr)
    mapping = analyze_defender_target_mapping(loader, data, rsr)
    impact  = compute_impact_angles(loader, data, rsr, mapping)
    stats   = compute_statistics(loader, data, rsr, per, mapping)

    intercept_t = rsr * loader.dt
    tg = {}
    for d in range(loader.n_defenders):
        tid = mapping[d]
        tg.setdefault(tid, {'defs': [], 'times': []})
        tg[tid]['defs'].append(f'D{d+1}')
        tg[tid]['times'].append(intercept_t[d])

    print(f"  Data: {pos.shape}, episode_end={episode_end}"
          f" (t={episode_end * loader.dt:.2f}s)")
    for tid in sorted(tg.keys()):
        dt = max(tg[tid]['times']) - min(tg[tid]['times'])
        print(f"    A{tid+1}: {tg[tid]['defs']} -> dt={dt:.3f}s")

    return {
        'data': data, 'repeat_start_rows': rsr,
        'plot_end_rows': per, 'episode_end': episode_end,
        'mapping': mapping, 'impact_angles': impact, 'stats': stats,
    }


# =====================================================================
#  MAIN  —  每个工况×算法 = 一系列独立图
# =====================================================================
def main():
    print("=" * 70)
    print("  IEEE TASE Publication Quality Plots  V9")
    print("  Per-Scenario-Method: 每个工况×算法 = 独立单幅图")
    print("=" * 70)

    base = Path('/home/uav/00gao_xueshu/DT_PAPER/guidance_pic_code/'
                'test全面数据版本v1/test')
    out  = base / 'figures_v9'
    out.mkdir(exist_ok=True)
    loader = DataLoader(str(base))

    # ---- 定义所有 (method, prefix, folder, eval_path) ----
    datasets = {
        'MAPPO': {
            'nopn': {
                'folder': 'mappo_success_nopn',
                'eval':   base / 'mappo_success_nopn' / 'mappo_success_nopn_eval.txt',
            },
            'sin': {
                'folder': 'mappo_success_sin',
                'eval':   base / 'mappo_success_sin' / 'sinmappo_eval' / 'agentseval.txt',
            },
        },
        'PN': {
            'nopn': {
                'folder': 'pn_success_nopn',
                'eval':   base / 'pn_success_nopn' / 'pn_success_nopn_eval.txt',
            },
            'sin': {
                'folder': 'pn_success_sin',
                'eval':   base / 'pn_success_sin' / 'sinpn_eval' / 'agentseval.txt',
            },
        },
    }

    all_stats, all_eval_stats = {}, {}
    all_eval = {}   # 存储 eval 数据供跨算法对比: {prefix: {'MAPPO': ev, 'PN': ev}}

    # ---- 遍历每个 method × prefix 组合 ----
    for method_name, scenarios in datasets.items():
        for prefix, cfg in scenarios.items():
            combo_label = f'{method_name}_{prefix}'
            print(f"\n{'='*60}")
            print(f"  [{combo_label}] Loading: {cfg['folder']}")
            print(f"{'='*60}")

            res = process_single_dataset(loader, cfg['folder'])
            if res is None:
                print(f"  ✗ {combo_label}: no valid data, skip")
                continue

            ev = loader.load_eval_data(str(cfg['eval']))
            if ev is not None:
                print(f"  eval: {ev.shape[0]} episodes (capped at {MC_MAX_RUNS})")

            # 统计
            key = f'{method_name.lower()}_{prefix}'
            all_stats[key] = res['stats']
            if ev is not None:
                all_eval_stats[key] = {
                    'mc_co_time': np.mean(ev[:, 0]),
                    'mc_n':       np.mean(ev[:, 1]),
                    'mc_miss':    np.mean(ev[:, 2]),
                }
            # 收集 eval 供后续对比箱式图
            all_eval.setdefault(prefix, {})[method_name] = ev

            # ---- 生成该组合的所有图 ----
            print(f"\n  Generating {combo_label} figures...")

            plot_trajectory_single(loader, res, prefix, method_name, out)
            plot_ny_single(loader, res, prefix, method_name, out)
            plot_nx_single(loader, res, prefix, method_name, out)
            plot_velocity_single(loader, res, prefix, method_name, out)
            plot_heading_single(loader, res, prefix, method_name, out)
            plot_tgo_single(loader, res, prefix, method_name, out)
            plot_tgo_error_single(loader, res, prefix, method_name, out)
            plot_distance_single(loader, res, prefix, method_name, out)
            plot_time_sync_single(loader, res, prefix, method_name, out)

            # Monte Carlo 散点
            # 这里增加了条件判断，排除 PN 算法的 sin 工况的散点图
            if not (method_name.upper() == 'PN' and prefix == 'sin'):
                plot_mc_scatter_single(ev, prefix, method_name, out)

    # ---- 跨算法对比箱式图 (每个场景一张, MAPPO vs PN) ----
    print(f"\n{'='*60}")
    print(f"  Generating MC boxplot comparisons...")
    print(f"{'='*60}")
    for prefix, ev_dict in all_eval.items():
        m_ev = ev_dict.get('MAPPO')
        p_ev = ev_dict.get('PN')
        plot_mc_boxplot_compare(m_ev, p_ev, prefix, out)

    # ---- 跨工况统计表 ----
    print(f"\n{'='*60}")
    print(f"  Generating cross-scenario statistics...")
    print(f"{'='*60}")
    generate_statistics_table(all_stats, all_eval_stats, out)

    # ---> 新增：在此处生成独立的图例图片 <---
    plot_standalone_duav_legend(out)

    # 修改这里，只统计 png 文件的数量
    n_files = len(list(out.glob('*.png')))
    print(f"\n{'='*70}")
    print(f"  All V9 figures generated!")
    print(f"  Output: {out}")
    print(f"  Total files: {n_files}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()