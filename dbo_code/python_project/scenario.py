"""
scenario.py – 20 Fixed-wing UAVs striking 8 Targets
=====================================================
Task-assignment problem definition:
  • 20 UAVs, each with position (x,y), speed v, heading θ
  • 8 ground targets, each with position (x,y) and threat level
  • Decision: assign each target j to one UAV i  (many-to-one allowed)
  • Encoding: solution vector x ∈ R^8, decoded to UAV indices [0..19]
  • Cost = weighted sum of distance cost, heading-deviation cost,
           workload-balance cost, threat-exposure cost

Designed for meta-heuristic optimisation comparison.
"""
import numpy as np

# ═══════════════════════════════════════════════════════════
# Scene layout (reproducible)
# ═══════════════════════════════════════════════════════════
np.random.seed(42)

N_UAV = 20
N_TARGET = 8

# ── UAV parameters ──
# Positions: spread in a 100 × 100 km area (launch zone top-half)
UAV_POS = np.array([
    [10, 85], [15, 90], [25, 80], [30, 95], [40, 88],
    [50, 92], [55, 78], [60, 85], [70, 90], [75, 82],
    [80, 95], [85, 80], [90, 88], [95, 92], [20, 75],
    [35, 70], [45, 82], [65, 76], [78, 70], [88, 75],
], dtype=float)  # (20, 2) km

# Speeds: 150–250 m/s  (fixed-wing UAV typical)
UAV_SPEED = np.array([
    180, 200, 190, 210, 170, 220, 195, 205, 185, 230,
    175, 215, 200, 190, 225, 180, 210, 195, 200, 185,
], dtype=float)  # m/s

# Heading angles: 0–360 degrees (measured from North, clockwise)
UAV_HEADING = np.array([
    200, 210, 190, 215, 185, 220, 195, 205, 210, 200,
    225, 195, 210, 200, 230, 220, 205, 215, 190, 210,
], dtype=float)  # degrees

# ── Target parameters ──
TARGET_POS = np.array([
    [20, 20], [35, 15], [50, 30], [65, 10],
    [30, 40], [55, 45], [75, 25], [85, 35],
], dtype=float)  # (8, 2) km

# Threat levels for each target (higher = more dangerous)
TARGET_THREAT = np.array([0.6, 0.8, 0.5, 0.9, 0.4, 0.7, 0.85, 0.3], dtype=float)

# ── Weight coefficients for cost function ──
W_DIST = 0.35       # distance cost weight
W_HEADING = 0.20    # heading-deviation cost weight
W_BALANCE = 0.25    # workload-balance cost weight
W_THREAT = 0.20     # threat-exposure cost weight

# ═══════════════════════════════════════════════════════════
# Decode continuous vector → discrete assignment
# ═══════════════════════════════════════════════════════════
def decode_solution(x):
    """Convert continuous vector (dim=8, range [0, N_UAV-1]) to integer UAV indices."""
    assignment = np.clip(np.round(x).astype(int), 0, N_UAV - 1)
    return assignment


# ═══════════════════════════════════════════════════════════
# Cost sub-functions
# ═══════════════════════════════════════════════════════════
def _distance_cost(assignment):
    """Euclidean distance from assigned UAV to each target (normalised by max)."""
    dists = np.zeros(N_TARGET)
    for j in range(N_TARGET):
        uav_idx = assignment[j]
        dists[j] = np.linalg.norm(UAV_POS[uav_idx] - TARGET_POS[j])
    # Normalise
    return dists.sum() / (N_TARGET * np.sqrt(100**2 + 100**2))  # max possible ~141 km


def _heading_cost(assignment):
    """Cost of heading deviation: angle between current heading and
    bearing-to-target.  Larger deviation → larger turning cost for fixed-wing UAV."""
    cost = 0.0
    for j in range(N_TARGET):
        uav_idx = assignment[j]
        dx = TARGET_POS[j, 0] - UAV_POS[uav_idx, 0]
        dy = TARGET_POS[j, 1] - UAV_POS[uav_idx, 1]
        bearing = np.degrees(np.arctan2(dx, dy)) % 360  # from North
        heading = UAV_HEADING[uav_idx]
        delta = abs(bearing - heading)
        delta = min(delta, 360 - delta)
        cost += delta / 180.0  # normalise to [0,1]
    return cost / N_TARGET


def _balance_cost(assignment):
    """Penalise uneven workload distribution.
    Ideal: each UAV assigned ≤ 1 target. Penalty for overloaded UAVs."""
    counts = np.zeros(N_UAV)
    for j in range(N_TARGET):
        counts[assignment[j]] += 1
    # Standard deviation of workload (only assigned UAVs matter)
    # Also penalise any UAV with >2 targets heavily
    std_pen = np.std(counts)
    overload_pen = np.sum(np.maximum(counts - 2, 0))
    return (std_pen + overload_pen) / N_TARGET


def _threat_cost(assignment):
    """Lower-speed UAVs have longer exposure time to threats.
    Cost = threat_level × distance / speed (exposure time proxy)."""
    cost = 0.0
    max_time = np.sqrt(100**2 + 100**2) * 1000 / 150  # worst-case seconds
    for j in range(N_TARGET):
        uav_idx = assignment[j]
        dist_m = np.linalg.norm(UAV_POS[uav_idx] - TARGET_POS[j]) * 1000  # km→m
        flight_time = dist_m / UAV_SPEED[uav_idx]
        cost += TARGET_THREAT[j] * flight_time / max_time
    return cost / N_TARGET


# ═══════════════════════════════════════════════════════════
# Objective function (for optimisers)
# ═══════════════════════════════════════════════════════════
def cost_function(x):
    """Evaluate total weighted cost for a continuous solution vector x (dim=8)."""
    assignment = decode_solution(x)
    c_dist = _distance_cost(assignment)
    c_head = _heading_cost(assignment)
    c_bal  = _balance_cost(assignment)
    c_thr  = _threat_cost(assignment)
    total = W_DIST * c_dist + W_HEADING * c_head + W_BALANCE * c_bal + W_THREAT * c_thr
    return total


# ═══════════════════════════════════════════════════════════
# Problem interface  (lb, ub, dim, fobj)
# ═══════════════════════════════════════════════════════════
def get_task_assignment_problem():
    """Return (lb, ub, dim, fobj) compatible with all optimiser signatures."""
    dim = N_TARGET  # 8
    lb = 0.0
    ub = float(N_UAV - 1)  # 19.0
    return lb, ub, dim, cost_function


# ═══════════════════════════════════════════════════════════
# Pretty-print an assignment
# ═══════════════════════════════════════════════════════════
def print_assignment(x):
    assignment = decode_solution(x)
    print("\n  Target → UAV Assignment:")
    print("  " + "-"*40)
    for j in range(N_TARGET):
        uav_id = assignment[j]
        dist = np.linalg.norm(UAV_POS[uav_id] - TARGET_POS[j])
        print(f"  Target {j+1} ({TARGET_POS[j]}) → UAV {uav_id+1:>2d}  "
              f"(pos={UAV_POS[uav_id]}, v={UAV_SPEED[uav_id]:.0f}m/s, "
              f"dist={dist:.1f}km, threat={TARGET_THREAT[j]:.1f})")
    print(f"\n  Total cost = {cost_function(x):.6f}")
    # Check balance
    counts = np.zeros(N_UAV)
    for j in range(N_TARGET):
        counts[assignment[j]] += 1
    used = np.sum(counts > 0)
    print(f"  UAVs used: {used}/{N_UAV},  max load: {int(counts.max())}")
