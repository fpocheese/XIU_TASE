import numpy as np
from pathlib import Path

from onpolicy.envs.mpe.core import FighterWorld, FighterAgent, Landmark
from onpolicy.envs.mpe.scenario import BaseScenario


def _arg(args, name, default):
    return getattr(args, name, default) if args is not None else default


def _unit(v, fallback=None):
    norm = np.linalg.norm(v)
    if norm < 1e-9:
        if fallback is None:
            return np.zeros_like(v)
        return fallback.copy()
    return v / norm


def _wrap_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


class Scenario(BaseScenario):
    """3D cooperative air-defense scenario.

    Defenders are policy-controlled fixed-wing agents launched from the ground.
    Attackers are scripted high-altitude incoming UAVs. The horizontal geometry
    follows the two paper cases while adding altitude and vertical guidance.
    """

    def make_world(self, args):
        self.reward_w_dist = _arg(args, "reward_w_dist", 0.10)
        self.reward_w_angle = _arg(args, "reward_w_angle", 1.00)
        self.reward_w_hit = _arg(args, "reward_w_hit", 1.00)
        self.reward_w_coord = _arg(args, "reward_w_coord", 1.00)
        self.reward_w_energy = _arg(args, "reward_w_energy", 1.00)
        self.reward_w_sync = _arg(args, "reward_w_sync", 1.00)
        self.reward_alpha_dist = _arg(args, "reward_alpha_dist", 1e-3)
        self.reward_alpha_angle = _arg(args, "reward_alpha_angle", 1e-2)
        self.reward_alpha_coord = _arg(args, "reward_alpha_coord", 5e-3)
        self.reward_alpha_energy = _arg(args, "reward_alpha_energy", 5e-2)
        self.reward_alpha_sync = _arg(args, "reward_alpha_sync", 2e-2)
        self.reward_hit_bonus = _arg(args, "reward_hit_bonus", 3.0)
        self.reward_hit_shaping = _arg(args, "reward_hit_shaping", 0.0)
        self.reward_hit_band_ratio = _arg(args, "reward_hit_band_ratio", 20.0)
        self.reward_coord_bonus = _arg(args, "reward_coord_bonus", 0.1)
        self.reward_coord_tol = _arg(args, "reward_coord_tol", 0.5)
        self.reward_sync_bonus = _arg(args, "reward_sync_bonus", 0.5)
        self.reward_async_hit_penalty = _arg(args, "reward_async_hit_penalty", 0.0)
        self.reward_sync_tol = _arg(args, "reward_sync_tol", 0.5)
        self.reward_sync_power = _arg(args, "reward_sync_power", 0.5)
        self.reward_sync_hits = _arg(args, "reward_sync_hits", 0)
        self.reward_angle_power = _arg(args, "reward_angle_power", 0.3)
        self.reward_coord_power = _arg(args, "reward_coord_power", 0.3)
        self.reward_use_progress = _arg(args, "reward_use_progress", False)
        self.case = _arg(args, "case_3d", "case1")
        self.paper_preset_path = _arg(args, "paper_preset_path", "")
        self.paper_attacker_replay = bool(int(_arg(args, "paper_attacker_replay", 0)))
        self.paper_altitude = float(_arg(args, "paper_altitude", 120.0))
        self.paper_altitude_step = float(_arg(args, "paper_altitude_step", 0.0))
        self.paper_defender_climb_to_target = bool(int(_arg(args, "paper_defender_climb_to_target", 0)))
        self.paper_preset = self._load_paper_preset(self.paper_preset_path)
        self.reference_control = self._load_reference_control(_arg(args, "reference_control_root", ""))
        self.reward_w_ref_control = _arg(args, "reward_w_ref_control", 0.0)
        self.reward_w_ref_rate = _arg(args, "reward_w_ref_rate", 0.0)
        self.target_assignment_mode = str(_arg(args, "target_assignment_mode", "fixed")).lower()
        self.target_assignment_spread_weight = _arg(args, "target_assignment_spread_weight", 6.0)
        self.hit_radius = _arg(args, "hit_radius_3d", 3.0)
        self.protected_asset = np.array([0.0, 0.0, 0.0])
        self.attack_maneuver_gain = _arg(args, "attack_maneuver_gain", 1.20)
        self.attack_maneuver_offset_gain = _arg(args, "attack_maneuver_offset_gain", 1.25)
        self.case1_lateral_base = _arg(args, "case1_lateral_base", 0.95)
        self.case1_lateral_tail = _arg(args, "case1_lateral_tail", 0.40)
        self.case1_vertical_amp = _arg(args, "case1_vertical_amp", 0.35)
        self.case2_lateral_amp = _arg(args, "case2_lateral_amp", 1.00)
        self.case2_maneuver_freq = _arg(args, "case2_maneuver_freq", 2.0 * np.pi / 50.0)
        self.case2_vertical_amp = _arg(args, "case2_vertical_amp", 0.25)
        self.attack_maneuver_freq = _arg(args, "attack_maneuver_freq", 0.17)
        self.case2_vertical_freq_scale = _arg(args, "case2_vertical_freq_scale", 0.50)
        self.attack_maneuver_fade_range = _arg(args, "attack_maneuver_fade_range", 450.0)
        self.no_tailchase_gate = _arg(args, "no_tailchase_gate", 0.0)
        self.no_tailchase_rebound = _arg(args, "no_tailchase_rebound", 5.0)
        self.no_tailchase_penalty = _arg(args, "no_tailchase_penalty", 0.0)
        self.no_tailchase_terminate = bool(_arg(args, "no_tailchase_terminate", False))

        world = FighterWorld()
        world.dim_p = 3
        world.dim_c = 4
        world.action_dim = 3
        world.use_agent_script_callback = True
        world.defender_guidance_base_gain = _arg(args, "defender_guidance_base_gain", 0.0)
        world.defender_guidance_tau = _arg(args, "defender_guidance_tau", 1.2)
        world.defender_guidance_lead = _arg(args, "defender_guidance_lead", 1.0)
        world.defender_residual_scale = _arg(args, "defender_residual_scale", 1.0)
        world.defender_load_limit = _arg(args, "defender_load_limit", 1.0)
        world.defender_axial_min = _arg(args, "defender_axial_min", -0.1)
        world.defender_axial_max = _arg(args, "defender_axial_max", 1.0)
        world.defender_sync_speed_gain = _arg(args, "defender_sync_speed_gain", 0.0)
        world.defender_sync_tgo_ref = _arg(args, "defender_sync_tgo_ref", "mean")
        world.defender_speed_target = _arg(args, "defender_speed_target", 0.0)
        world.defender_speed_gain = _arg(args, "defender_speed_gain", 0.0)
        world.defender_min_accel_load = _arg(args, "defender_min_accel_load", 0.0)
        world.defender_speed_min = _arg(args, "defender_speed_min", 12.0)
        world.defender_speed_max = _arg(args, "defender_speed_max", 40.0)
        world.reference_control = self.reference_control
        world.defender_reference_blend = _arg(args, "defender_reference_blend", 0.0)
        world.defender_sensor_delay_steps = _arg(args, "defender_sensor_delay_steps", 0)
        world.defender_sensor_delay_compensate = bool(_arg(args, "defender_sensor_delay_compensate", False))
        world.defender_obs_pos_noise_std = _arg(args, "defender_obs_pos_noise_std", 0.0)
        world.defender_obs_vel_noise_std = _arg(args, "defender_obs_vel_noise_std", 0.0)
        world.defender_obs_filter_alpha = _arg(args, "defender_obs_filter_alpha", 1.0)
        world.defender_command_lag_tau = _arg(args, "defender_command_lag_tau", 0.0)
        world.reward_w_smooth = _arg(args, "reward_w_smooth", 0.0)
        world.attacker_speed_min = _arg(args, "attacker_speed_min", 12.0)
        world.attacker_speed_max = _arg(args, "attacker_speed_max", 65.0)
        world.attacker_axial_min = _arg(args, "attacker_axial_min", -4.0)
        world.attacker_axial_max = _arg(args, "attacker_axial_max", 4.0)

        num_good_agents = 8      # attackers
        num_adversaries = 20     # defenders
        world.agents = [FighterAgent() for _ in range(num_adversaries + num_good_agents)]
        world.food = [Landmark()]
        world.aaa = []
        world.bbb = []
        world.ccc = []
        world.landmarks = world.food

        assignment = [20, 21, 22, 23, 24, 25, 26, 27,
                      20, 21, 22, 23, 24, 25, 26, 27,
                      20, 21, 22, 23]
        for i, agent in enumerate(world.agents):
            agent.name = "agent %d" % i
            agent.namenumber = i
            agent.doneflag = False
            agent.collide = True
            agent.leader = i == 0
            agent.silent = i > 0
            agent.adversary = i < num_adversaries
            agent.size = 13 if agent.adversary else 15
            agent.accel = 3.0 if agent.adversary else 4.5
            agent.state.target = 10
            agent.state.done = False
            agent.target = assignment[i] if agent.adversary else 0
            agent.q_old = 0.0
            agent.state.timestep = 0.0
            if not agent.adversary:
                agent.action_callback = self.action_callback

        world.food[0].name = "protected asset"
        world.food[0].collide = False
        world.food[0].movable = False
        world.food[0].food = True
        world.food[0].size = 5
        world.food[0].boundary = False
        world.attacker_load_limit = _arg(args, "attacker_load_limit", 1.75)
        world.attacker_yaw_scale = _arg(args, "attacker_yaw_scale", 1.55)
        world.attacker_pitch_scale = _arg(args, "attacker_pitch_scale", 1.55)
        self.reset_world(world)
        return world

    def reset_world(self, world):
        rng = np.random
        world.landmarks[0].state.p_pos = self.protected_asset.copy()
        world.landmarks[0].state.p_vel = np.zeros(world.dim_p)

        for agent in world.agents:
            agent.doneflag = False
            agent.action.u = np.zeros(world.action_dim)
            agent.action.c = np.zeros(world.dim_c)
            agent.state.c = np.zeros(world.dim_c)
            agent.state.time_tgo = np.zeros(world.dim_p)
            agent.state.time_tgo_dist = np.zeros(world.dim_p)
            agent.state.load = np.zeros(world.action_dim)
            agent.state.doneflag_me_target = np.zeros(world.dim_p)
            agent.state.last_loadx = 0.0
            agent.state.last_loady = 0.0
            agent.state.load_delt_all = 0.0
            agent.state.last_q_dot = 0.0
            agent.state.kalman_p_last = 1.0
            agent.state.dist_target = 1000.0
            agent.state.defender_sensor_history = None
            agent.state.defender_obs_history = None
            agent.state.defender_sensor_filtered = None
            agent.state.defender_obs_filtered = None
            agent.state.defender_command_lag_load = None
            agent.state.reward_prev_load = None
            agent.state.reward_prev_ref_load_xy = None
            agent.state.eval = np.zeros(5)
            agent.state.eval_flag = 0
            agent.state.timestep = 0.0
            agent.state.timeover = False
            agent.state.actual_hit = False
            agent.state.hit_time = np.nan
            agent.state.hit_reward_paid = False
            agent.state.sync_reward_paid = False
            agent.state.no_tailchase_penalty_paid = False
            agent.state.tailchase_entered_gate = False
            agent.state.tailchase_failure = False
            agent.state.env_tailchase_failure = False
            agent.state.first_min_time = np.nan
            agent.state.first_min_distance = np.inf
            agent.state.max_rebound_after_first_min = 0.0
            agent.color = np.array([0.45, 0.95, 0.45]) if not agent.adversary else np.array([0.95, 0.45, 0.45])

        attackers = self.good_agents(world)
        defenders = self.adversaries(world)

        if self.paper_preset is not None:
            self._reset_paper_world(world, defenders, attackers)
            return

        att_angles = np.linspace(0.0, 2 * np.pi, len(attackers), endpoint=False) + rng.normal(0.0, 0.08, len(attackers))
        for j, agent in enumerate(attackers):
            radius = rng.uniform(1200.0, 1500.0)
            altitude = rng.uniform(650.0, 850.0)
            agent.state.p_pos = np.array([radius * np.cos(att_angles[j]), radius * np.sin(att_angles[j]), altitude])
            direction = _unit(self.protected_asset - agent.state.p_pos, np.array([0.0, 0.0, -1.0]))
            speed = rng.uniform(24.0, 30.0)
            agent.state.p_vel = speed * direction
            agent.state.v_vel = self._vel_to_flight_state(agent.state.p_vel)
            agent.state.phase = rng.uniform(0.0, 2 * np.pi)
            agent.state.qddd = np.zeros(world.dim_p)
            agent.state.qddd1 = np.zeros(world.dim_p)
            agent.state.k3 = np.zeros(world.dim_p)
            agent.state.k4 = np.zeros(world.dim_p)
            agent.state.k5 = np.zeros(world.dim_p)
            agent.state.kq = np.zeros(world.dim_p)

        def_angles = np.linspace(0.0, 2 * np.pi, len(defenders), endpoint=False) + rng.normal(0.0, 0.12, len(defenders))
        for i, agent in enumerate(defenders):
            radius = rng.uniform(20.0, 95.0)
            agent.state.p_pos = np.array([radius * np.cos(def_angles[i]), radius * np.sin(def_angles[i]), 0.0])

        if self.target_assignment_mode == "dynamic":
            assignment = self._dynamic_target_assignment(defenders, attackers)
        else:
            assignment = [agent.target for agent in defenders]

        for i, agent in enumerate(defenders):
            agent.target = int(assignment[i])
            target = world.agents[agent.target]
            direction = _unit(target.state.p_pos - agent.state.p_pos + np.array([0.0, 0.0, 35.0]))
            speed = rng.uniform(16.0, 22.0)
            agent.state.p_vel = speed * direction
            agent.state.v_vel = self._vel_to_flight_state(agent.state.p_vel)
            agent.state.lamda0 = agent.state.v_vel[2]
            agent.state.dist0 = max(np.linalg.norm(target.state.p_pos - agent.state.p_pos), 0.01)
            agent.state.dist1 = agent.state.dist0
            agent.state.qddd = np.zeros(world.dim_p)
            agent.state.qddd1 = np.zeros(world.dim_p)
            agent.state.k3 = np.zeros(world.dim_p)
            agent.state.k4 = np.zeros(world.dim_p)
            agent.state.k5 = np.zeros(world.dim_p)
            agent.state.kq = np.zeros(world.dim_p)


    def _paper_key(self):
        return "case2" if self.case == "case2" else "case1"

    def _load_paper_preset(self, path):
        if path is None or str(path).strip() == "":
            return None
        p = Path(path).expanduser()
        if not p.exists():
            raise FileNotFoundError("paper preset not found: %s" % p)
        return np.load(str(p), allow_pickle=False)

    def _load_reference_control(self, root):
        if root is None or str(root).strip() == "":
            return None
        root = Path(root).expanduser()
        folder_name = "mappo_success_sin" if self._paper_key() == "case2" else "mappo_success_nopn"
        folder = root / folder_name if (root / folder_name).exists() else root
        path = folder / "agentsall.txt"
        pos_path = folder / "agentspos.txt"
        if not path.exists():
            raise FileNotFoundError("reference agentsall.txt not found: %s" % path)

        def load_pad(p):
            lines = [line for line in Path(p).read_text().splitlines() if line.strip()]
            max_cols = max(len(line.split()) for line in lines)
            rows = []
            for line in lines:
                vals = [float(x) for x in line.split()]
                while len(vals) < max_cols:
                    vals.append(vals[-1] if vals else 0.0)
                rows.append(vals)
            return np.asarray(rows, dtype=np.float32)

        data = load_pad(path)
        end = data.shape[0]
        if pos_path.exists():
            pos = load_pad(pos_path)
            if pos.shape[0] > 1:
                jump = np.where(np.abs(np.diff(pos[:, 0])) > 100.0)[0]
                if len(jump) > 0:
                    end = int(jump[0] + 1)
        data = data[:end, :40]
        return data.reshape(data.shape[0], 20, 2)

    def _reset_paper_world(self, world, defenders, attackers):
        key = self._paper_key()
        preset = self.paper_preset
        assignment_key = f"{key}_assignment"
        if assignment_key in preset.files:
            assignment = preset[assignment_key].astype(int).tolist()
        else:
            assignment = preset["assignment"].astype(int).tolist()
        def_pos = preset[f"{key}_def_pos0"]
        def_vel = preset[f"{key}_def_vel0"]
        atk_pos = preset[f"{key}_atk_pos0"]
        atk_vel = preset[f"{key}_atk_vel0"]

        for j, agent in enumerate(attackers):
            altitude = self.paper_altitude + self.paper_altitude_step * (j - 0.5 * (len(attackers) - 1))
            agent.state.p_pos = np.array([atk_pos[j, 0], atk_pos[j, 1], altitude], dtype=float)
            agent.state.p_vel = np.array([atk_vel[j, 0], atk_vel[j, 1], 0.0], dtype=float)
            agent.state.v_vel = self._vel_to_flight_state(agent.state.p_vel)
            agent.state.phase = 0.0
            agent.state.qddd = np.zeros(world.dim_p)
            agent.state.qddd1 = np.zeros(world.dim_p)
            agent.state.k3 = np.zeros(world.dim_p)
            agent.state.k4 = np.zeros(world.dim_p)
            agent.state.k5 = np.zeros(world.dim_p)
            agent.state.kq = np.zeros(world.dim_p)

        for i, agent in enumerate(defenders):
            agent.target = int(assignment[i])
            agent.state.p_pos = np.array([def_pos[i, 0], def_pos[i, 1], 0.0], dtype=float)
            target = world.agents[agent.target]
            vz = 0.0
            if self.paper_defender_climb_to_target:
                horiz = max(np.linalg.norm(target.state.p_pos[:2] - agent.state.p_pos[:2]), 1.0)
                horiz_speed = max(np.linalg.norm(def_vel[i]), 1.0)
                vz = horiz_speed * (target.state.p_pos[2] - agent.state.p_pos[2]) / horiz
            agent.state.p_vel = np.array([def_vel[i, 0], def_vel[i, 1], vz], dtype=float)
            agent.state.v_vel = self._vel_to_flight_state(agent.state.p_vel)
            agent.state.lamda0 = agent.state.v_vel[2]
            agent.state.dist0 = max(np.linalg.norm(target.state.p_pos - agent.state.p_pos), 0.01)
            agent.state.dist1 = agent.state.dist0
            agent.state.qddd = np.zeros(world.dim_p)
            agent.state.qddd1 = np.zeros(world.dim_p)
            agent.state.k3 = np.zeros(world.dim_p)
            agent.state.k4 = np.zeros(world.dim_p)
            agent.state.k5 = np.zeros(world.dim_p)
            agent.state.kq = np.zeros(world.dim_p)

    def _dynamic_target_assignment(self, defenders, attackers):
        target_ids = [int(agent.namenumber) for agent in attackers]
        if len(defenders) < 2 * len(target_ids):
            return [agent.target for agent in defenders]

        defender_pos = np.array([agent.state.p_pos for agent in defenders], dtype=float)
        target_pos = {int(agent.namenumber): np.array(agent.state.p_pos, dtype=float) for agent in attackers}
        min_cost = []
        for tid in target_ids:
            dists = np.linalg.norm(defender_pos - target_pos[tid], axis=1)
            min_cost.append((float(np.min(dists)), tid))

        slots = []
        for tid in target_ids:
            slots.extend([tid, tid])
        extra_count = len(defenders) - len(slots)
        hard_targets = [tid for _, tid in sorted(min_cost, reverse=True)]
        slots.extend(hard_targets[:extra_count])

        target_index = {tid: j for j, tid in enumerate(target_ids)}
        target_cost = np.zeros((len(defenders), len(target_ids)), dtype=float)
        for i, defender in enumerate(defenders):
            speed = max(float(defender.state.v_vel[0]) if defender.state.v_vel is not None else 20.0, 1.0)
            for tid, j in target_index.items():
                dist = np.linalg.norm(target_pos[tid] - defender.state.p_pos)
                target_cost[i, j] = dist / speed

        cost = np.zeros((len(defenders), len(slots)), dtype=float)
        for i, defender in enumerate(defenders):
            speed = max(float(defender.state.v_vel[0]) if defender.state.v_vel is not None else 20.0, 1.0)
            for j, tid in enumerate(slots):
                dist = np.linalg.norm(target_pos[tid] - defender.state.p_pos)
                cost[i, j] = dist / speed
        try:
            from scipy.optimize import linear_sum_assignment
            rows, cols = linear_sum_assignment(cost)
            assignment = [None] * len(defenders)
            for r, c in zip(rows, cols):
                assignment[int(r)] = int(slots[int(c)])
            if all(t is not None for t in assignment):
                return self._refine_target_assignment(assignment, target_ids, target_cost)
        except Exception:
            pass
        return [agent.target for agent in defenders]

    def _refine_target_assignment(self, assignment, target_ids, target_cost):
        target_index = {tid: j for j, tid in enumerate(target_ids)}
        weight = float(self.target_assignment_spread_weight)

        def objective(assign):
            total = 0.0
            groups = {tid: [] for tid in target_ids}
            for i, tid in enumerate(assign):
                c = float(target_cost[i, target_index[int(tid)]])
                total += c
                groups[int(tid)].append(c)
            spread = 0.0
            for vals in groups.values():
                if len(vals) >= 2:
                    spread += max(vals) - min(vals)
            return total + weight * spread

        best = list(assignment)
        best_score = objective(best)
        improved = True
        passes = 0
        while improved and passes < 20:
            improved = False
            passes += 1
            for i in range(len(best)):
                for j in range(i + 1, len(best)):
                    if best[i] == best[j]:
                        continue
                    trial = list(best)
                    trial[i], trial[j] = trial[j], trial[i]
                    score = objective(trial)
                    if score + 1e-9 < best_score:
                        best = trial
                        best_score = score
                        improved = True
        return best

    def _vel_to_flight_state(self, vel):
        speed = max(np.linalg.norm(vel), 1e-6)
        yaw = np.arctan2(vel[1], vel[0])
        pitch = np.arcsin(np.clip(vel[2] / speed, -1.0, 1.0))
        return np.array([speed, pitch, yaw])

    def _flight_dir(self, agent):
        speed, pitch, yaw = agent.state.v_vel
        return np.array([np.cos(pitch) * np.cos(yaw),
                         np.cos(pitch) * np.sin(yaw),
                         np.sin(pitch)])

    def good_agents(self, world):
        return [agent for agent in world.agents if not agent.adversary]

    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]

    def is_collision(self, agent1, agent2):
        return np.linalg.norm(agent1.state.p_pos - agent2.state.p_pos) < self.hit_radius

    def _update_tailchase_metrics(self, agent, target, dist):
        if self.no_tailchase_gate <= 0.0 or not agent.adversary:
            return False
        if getattr(agent.state, "tailchase_failure", False):
            return True
        if dist <= self.no_tailchase_gate:
            if not getattr(agent.state, "tailchase_entered_gate", False):
                agent.state.tailchase_entered_gate = True
                agent.state.first_min_distance = float(dist)
                agent.state.first_min_time = float(agent.state.timestep)
            elif dist < float(getattr(agent.state, "first_min_distance", np.inf)):
                agent.state.first_min_distance = float(dist)
                agent.state.first_min_time = float(agent.state.timestep)
        if getattr(agent.state, "tailchase_entered_gate", False):
            rebound = float(dist) - float(agent.state.first_min_distance)
            agent.state.max_rebound_after_first_min = max(
                float(getattr(agent.state, "max_rebound_after_first_min", 0.0)),
                rebound,
            )
            if rebound > self.no_tailchase_rebound:
                agent.state.tailchase_failure = True
                agent.state.env_tailchase_failure = True
                return True
        return False

    def _same_target_group(self, agent, world):
        return [other for other in self.adversaries(world) if other.target == agent.target]

    def _group_tgo_values(self, agent, world):
        values = []
        target = world.agents[agent.target]
        for other in self._same_target_group(agent, world):
            if getattr(other.state, "actual_hit", False):
                continue
            dist = np.linalg.norm(target.state.p_pos - other.state.p_pos)
            values.append(dist / max(other.state.v_vel[0], 1.0))
        return values

    def _required_sync_hits(self, group_size):
        requested = int(self.reward_sync_hits) if self.reward_sync_hits is not None else 0
        if requested <= 0:
            return group_size
        return min(group_size, requested)

    def _group_arrival_times(self, agent, world, predict_remaining=False):
        target = world.agents[agent.target]
        times = []
        now = float(agent.state.timestep)
        for other in self._same_target_group(agent, world):
            if getattr(other.state, "actual_hit", False) and np.isfinite(other.state.hit_time):
                times.append(float(other.state.hit_time))
            elif predict_remaining:
                dist = np.linalg.norm(target.state.p_pos - other.state.p_pos)
                times.append(now + dist / max(other.state.v_vel[0], 1.0))
        return times

    def _sync_terminal_bonus(self, agent, world):
        group = self._same_target_group(agent, world)
        required = self._required_sync_hits(len(group))

        hit_times = self._group_arrival_times(agent, world, predict_remaining=False)
        if len(hit_times) >= required and (max(hit_times) - min(hit_times)) <= self.reward_sync_tol:
            return self.reward_sync_bonus

        arrival_times = self._group_arrival_times(agent, world, predict_remaining=True)
        if len(arrival_times) >= required and (max(arrival_times) - min(arrival_times)) <= self.reward_sync_tol:
            return self.reward_sync_bonus
        return 0.0

    def _async_terminal_penalty(self, agent, world):
        if self.reward_async_hit_penalty <= 0.0:
            return 0.0
        target = world.agents[agent.target]
        remaining_tgo = []
        for other in self._same_target_group(agent, world):
            if other is agent or getattr(other.state, "actual_hit", False):
                continue
            dist = np.linalg.norm(target.state.p_pos - other.state.p_pos)
            remaining_tgo.append(dist / max(other.state.v_vel[0], 1.0))
        if remaining_tgo and max(remaining_tgo) > self.reward_sync_tol:
            return self.reward_async_hit_penalty
        return 0.0

    def reward(self, agent, world):
        return self.adversary_reward(agent, world) if agent.adversary else np.array([0.0])

    def adversary_reward(self, agent, world):
        if agent.doneflag:
            if getattr(agent.state, "tailchase_failure", False):
                if self.no_tailchase_penalty > 0.0 and not getattr(agent.state, "no_tailchase_penalty_paid", False):
                    agent.state.no_tailchase_penalty_paid = True
                    return np.array([-self.no_tailchase_penalty], dtype=np.float32)
                return np.array([0.0])
            if getattr(agent.state, "actual_hit", False):
                terminal = 0.0
                if not getattr(agent.state, "hit_reward_paid", False):
                    agent.state.hit_reward_paid = True
                    terminal += self.reward_w_hit * self.reward_hit_bonus
                    terminal -= self.reward_w_sync * self._async_terminal_penalty(agent, world)
                if not getattr(agent.state, "sync_reward_paid", False):
                    sync_bonus = self._sync_terminal_bonus(agent, world)
                    if sync_bonus > 0.0:
                        agent.state.sync_reward_paid = True
                        terminal += self.reward_w_sync * sync_bonus
                if terminal != 0.0:
                    return np.array([terminal], dtype=np.float32)
            return np.array([0.0])
        target = world.agents[agent.target]
        rel = target.state.p_pos - agent.state.p_pos
        dist = max(np.linalg.norm(rel), 0.01)
        tailchase_failure = self._update_tailchase_metrics(agent, target, dist)

        if self.reward_use_progress:
            r_dist = (agent.state.dist1 - dist) / 1000.0
        else:
            r_dist = np.exp(-self.reward_alpha_dist * dist)
        agent.state.dist1 = dist

        los = _unit(rel)
        flight_dir = self._flight_dir(agent)
        angle_err = np.arccos(np.clip(np.dot(flight_dir, los), -1.0, 1.0))
        r_angle = -self.reward_alpha_angle * (angle_err ** self.reward_angle_power)

        r_hit = self.reward_hit_bonus if self.is_collision(agent, target) else 0.0
        if self.reward_hit_shaping > 0.0:
            band = max(self.hit_radius * self.reward_hit_band_ratio, self.hit_radius + 1e-6)
            r_hit += self.reward_hit_shaping * np.exp(-dist / band)
        r_energy = -self.reward_alpha_energy * np.sum(np.square(agent.action.u)) * world.dt

        tgo = dist / max(agent.state.v_vel[0], 1.0)
        group_tgo = self._group_tgo_values(agent, world)
        tgo_mean = np.mean(group_tgo) if group_tgo else tgo
        time_abs = abs(tgo - tgo_mean)
        r_coord = -self.reward_alpha_coord * (time_abs ** self.reward_coord_power)
        if time_abs <= self.reward_coord_tol:
            r_coord += self.reward_coord_bonus

        if len(group_tgo) >= 2:
            sync_spread = max(group_tgo) - min(group_tgo)
            r_sync = -self.reward_alpha_sync * (sync_spread ** self.reward_sync_power)
            if sync_spread <= self.reward_sync_tol:
                r_sync += self.reward_sync_bonus
        else:
            sync_spread = 0.0
            r_sync = 0.0

        agent.state.dist_target = dist
        agent.state.time_tgo[0] = tgo
        agent.state.time_tgo[1] = dist
        agent.state.time_tgo_dist[0] = time_abs
        agent.state.time_tgo_dist[1] = sync_spread
        agent.state.eval[0] = agent.state.timestep
        agent.state.eval[1] = np.linalg.norm(agent.state.load)
        agent.state.eval[2] = dist
        agent.state.eval[3] += np.sum(np.square(agent.state.load)) * world.dt

        r_smooth = 0.0
        smooth_weight = float(getattr(world, "reward_w_smooth", 0.0))
        if smooth_weight > 0.0 and agent.adversary:
            prev_load = getattr(agent.state, "reward_prev_load", None)
            if prev_load is not None and len(prev_load) == len(agent.state.load):
                r_smooth = -smooth_weight * float(np.sum(np.square(agent.state.load - prev_load)))
            agent.state.reward_prev_load = agent.state.load.copy()

        r_ref = 0.0
        ref = getattr(world, "reference_control", None)
        if ref is not None and agent.adversary:
            idx = int(np.clip(round(agent.state.timestep / max(world.dt, 1e-6)) - 1, 0, ref.shape[0] - 1))
            did = int(agent.namenumber)
            if did < ref.shape[1]:
                ref_load = np.asarray(ref[idx, did], dtype=float)
                load_xy = np.asarray(agent.state.load[:2], dtype=float)
                ref_weight = float(getattr(self, "reward_w_ref_control", 0.0))
                if ref_weight > 0.0:
                    r_ref -= ref_weight * float(np.mean(np.square(load_xy - ref_load)))
                ref_rate_weight = float(getattr(self, "reward_w_ref_rate", 0.0))
                if ref_rate_weight > 0.0 and idx > 0:
                    prev_load = getattr(agent.state, "reward_prev_ref_load_xy", None)
                    if prev_load is not None and len(prev_load) == 2:
                        ref_delta = ref_load - np.asarray(ref[idx - 1, did], dtype=float)
                        load_delta = load_xy - np.asarray(prev_load, dtype=float)
                        r_ref -= ref_rate_weight * float(np.mean(np.square(load_delta - ref_delta)))
                    agent.state.reward_prev_ref_load_xy = load_xy.copy()

        rew = (self.reward_w_dist * r_dist +
               self.reward_w_angle * r_angle +
               self.reward_w_hit * r_hit +
               self.reward_w_coord * r_coord +
               self.reward_w_sync * r_sync +
               self.reward_w_energy * r_energy +
               r_smooth +
               r_ref)
        if tailchase_failure and self.no_tailchase_penalty > 0.0:
            rew -= self.no_tailchase_penalty
            agent.state.no_tailchase_penalty_paid = True
        return np.array([rew], dtype=np.float32)

    def _observed_target_state(self, agent, target, world):
        delay_steps = max(0, int(getattr(world, "defender_sensor_delay_steps", 0)))
        pos_std = max(0.0, float(getattr(world, "defender_obs_pos_noise_std", 0.0)))
        vel_std = max(0.0, float(getattr(world, "defender_obs_vel_noise_std", 0.0)))
        if delay_steps <= 0 and pos_std <= 0.0 and vel_std <= 0.0:
            return target.state.p_pos, target.state.p_vel

        hist = getattr(agent.state, "defender_obs_history", None)
        if hist is None:
            hist = []
        target_vel = target.state.p_vel if target.state.p_vel is not None else np.zeros(3)
        hist.append((target.state.p_pos.copy(), target_vel.copy()))
        max_len = delay_steps + 1
        if len(hist) > max_len:
            hist = hist[-max_len:]
        agent.state.defender_obs_history = hist
        sensed_pos, sensed_vel = hist[0] if len(hist) <= delay_steps else hist[-max_len]
        sensed_pos = sensed_pos.copy()
        sensed_vel = sensed_vel.copy()
        if bool(getattr(world, "defender_sensor_delay_compensate", False)) and delay_steps > 0:
            sensed_pos += sensed_vel * (delay_steps * world.dt)
        if pos_std > 0.0:
            sensed_pos += np.random.normal(0.0, pos_std, size=3)
        if vel_std > 0.0:
            sensed_vel += np.random.normal(0.0, vel_std, size=3)
        alpha = float(getattr(world, "defender_obs_filter_alpha", 1.0))
        if alpha < 1.0:
            alpha = max(0.0, alpha)
            prev = getattr(agent.state, "defender_obs_filtered", None)
            if prev is None:
                filt_pos, filt_vel = sensed_pos, sensed_vel
            else:
                filt_pos = np.asarray(prev[0], dtype=float) + alpha * (sensed_pos - np.asarray(prev[0], dtype=float))
                filt_vel = np.asarray(prev[1], dtype=float) + alpha * (sensed_vel - np.asarray(prev[1], dtype=float))
            agent.state.defender_obs_filtered = (filt_pos.copy(), filt_vel.copy())
            sensed_pos, sensed_vel = filt_pos, filt_vel
        return sensed_pos, sensed_vel

    def observation(self, agent, world):
        target = world.agents[agent.target]
        target_pos, target_vel = self._observed_target_state(agent, target, world)
        rel = target_pos - agent.state.p_pos
        dist = max(np.linalg.norm(rel), 0.01)
        true_dist = max(np.linalg.norm(target.state.p_pos - agent.state.p_pos), 0.01)
        self._update_tailchase_metrics(agent, target, true_dist)
        los = rel / dist
        rel_vel = target_vel - agent.state.p_vel
        closing = np.dot(rel_vel, los)
        flight_dir = self._flight_dir(agent)
        angle_err = np.arccos(np.clip(np.dot(flight_dir, los), -1.0, 1.0))
        tgo = dist / max(agent.state.v_vel[0], 1.0)

        group_tgo = []
        for other in self.adversaries(world):
            if other.target == agent.target and not other.doneflag:
                d = np.linalg.norm(target_pos - other.state.p_pos)
                group_tgo.append(d / max(other.state.v_vel[0], 1.0))
        tgo_mean = np.mean(group_tgo) if group_tgo else tgo
        tgo_min = np.min(group_tgo) if group_tgo else tgo
        tgo_max = np.max(group_tgo) if group_tgo else tgo
        sync_spread = tgo_max - tgo_min

        obs = np.array([
            dist / 2000.0,
            rel[2] / 1000.0,
            closing / 100.0,
            angle_err / np.pi,
            agent.state.v_vel[0] / max(float(getattr(world, "defender_speed_max", 40.0)), 1.0),
            np.linalg.norm(target_vel) / 65.0,
            agent.state.v_vel[1] / np.pi,
            _wrap_pi(agent.state.v_vel[2]) / np.pi,
            los[0],
            los[1],
            los[2],
            (tgo - tgo_mean) / 50.0,
            sync_spread / 50.0,
            (tgo - tgo_min) / 50.0,
            agent.state.load[0],
            agent.state.load[1],
            agent.state.load[2] if len(agent.state.load) > 2 else 0.0,
        ], dtype=np.float32)
        agent.state.dist_target = dist
        return obs

    def done_callback(self, agent, world):
        if agent.doneflag:
            return True

        target = world.agents[agent.target]
        dist = np.linalg.norm(target.state.p_pos - agent.state.p_pos)
        tailchase_failure = self._update_tailchase_metrics(agent, target, dist)
        if tailchase_failure and self.no_tailchase_terminate:
            agent.doneflag = True
            agent.state.actual_hit = False
            agent.state.env_tailchase_failure = True
            return True
        if agent.adversary and self.is_collision(agent, world.agents[agent.target]):
            agent.state.doneflag_me_target[0] = 1
            agent.doneflag = True
            agent.state.actual_hit = not tailchase_failure
            agent.state.hit_time = agent.state.timestep

            same_target = [other for other in self.adversaries(world) if other.target == agent.target]
            if same_target and all(getattr(other.state, "actual_hit", False) for other in same_target):
                target.doneflag = True
            return True
        return False

    def action_callback(self, agent, world):
        if self.paper_preset is not None and self.paper_attacker_replay:
            return self._paper_attacker_action(agent, world)
        target = world.landmarks[0]
        rel = target.state.p_pos - agent.state.p_pos
        los = _unit(rel, np.array([0.0, 0.0, -1.0]))
        speed = max(agent.state.v_vel[0], 1.0)
        flight_dir = self._flight_dir(agent)

        vertical = np.array([0.0, 0.0, 1.0])
        horizontal_perp = _unit(np.cross(vertical, los), np.array([1.0, 0.0, 0.0]))
        phase = getattr(agent.state, "phase", 0.0)
        t = agent.state.timestep
        if self.case == "case2":
            maneuver = self.case2_lateral_amp * np.sin(self.attack_maneuver_freq * t + phase) * horizontal_perp
            vertical_freq = max(1e-6, self.case2_vertical_freq_scale * self.attack_maneuver_freq)
            maneuver += self.case2_vertical_amp * np.sin(vertical_freq * t + 0.7 * phase) * vertical
        else:
            if t < 15.0:
                sign = 1.0 if (agent.namenumber % 2 == 0) else -1.0
                maneuver = sign * self.case1_lateral_base * horizontal_perp
                maneuver += self.case1_vertical_amp * np.sin(self.attack_maneuver_freq * t + phase) * vertical
            else:
                maneuver = np.zeros(3)

        if self.case == "case1" and t >= 15.0:
            offset = np.zeros(3)
        else:
            offset = self.attack_maneuver_offset_gain * (
                0.03 * np.sin(0.22 * t) * horizontal_perp
                + 0.03 * np.sin(0.19 * t) * vertical
            )
        maneuver = (maneuver + offset) * self.attack_maneuver_gain
        fade_range = max(float(self.attack_maneuver_fade_range), 1.0)
        maneuver *= np.clip(np.linalg.norm(rel) / fade_range, 0.18, 1.0)

        desired_dir = _unit(los + maneuver, los)
        desired_vel = np.clip(speed, 20.0, 32.0) * desired_dir
        acc_cmd = (desired_vel - agent.state.p_vel) / 1.2

        vhat = _unit(agent.state.p_vel, flight_dir)
        axial = np.dot(acc_cmd, vhat) / 9.81
        normal = acc_cmd - np.dot(acc_cmd, vhat) * vhat
        horizontal_vel = np.array([agent.state.p_vel[0], agent.state.p_vel[1], 0.0])
        hhat = _unit(horizontal_vel, np.array([1.0, 0.0, 0.0]))
        lhat = np.array([-hhat[1], hhat[0], 0.0])

        agent.action.u = np.array([
            np.clip(np.dot(normal, lhat) / 9.81, -1.0, 1.0),
            np.clip(normal[2] / 9.81, -1.0, 1.0),
        ], dtype=np.float32)
        return agent.action


    def _paper_attacker_action(self, agent, world):
        key = self._paper_key()
        j = int(agent.namenumber) - 20
        dt = max(float(getattr(world, "dt", 0.05)), 1e-6)

        def preset_scalar(name, default):
            if self.paper_preset is None or name not in self.paper_preset.files:
                return float(default)
            values = np.asarray(self.paper_preset[name], dtype=float).reshape(-1)
            if values.size == 0:
                return float(default)
            return float(values[j] if values.size > 1 else values[0])

        rel_to_asset = world.landmarks[0].state.p_pos - agent.state.p_pos
        horizontal_dist = max(float(np.linalg.norm(rel_to_asset[:2])), 1.0)
        strike_pitch = np.arctan2(rel_to_asset[2], horizontal_dist)
        desired_pitch = np.clip(strike_pitch, -0.45, 0.05)

        speed = max(float(agent.state.v_vel[0]), 1.0)
        pitch = float(agent.state.v_vel[1])
        vel_xy = agent.state.p_vel[:2]
        rel_xy = rel_to_asset[:2]
        dist_xy = max(float(np.linalg.norm(rel_xy)), 1e-6)
        q = np.arctan2(rel_xy[1], rel_xy[0])
        vel_r_xy = -vel_xy
        q_dot = (vel_r_xy[1] * np.cos(q) - vel_r_xy[0] * np.sin(q)) / dist_xy
        horizontal_speed0 = float(np.linalg.norm(self.paper_preset[f"{key}_atk_vel0"][j]))
        desired_horizontal_speed = max(12.0, min(float(getattr(world, "attacker_speed_max", 80.0)), horizontal_speed0))
        desired_speed = desired_horizontal_speed / max(np.cos(desired_pitch), 0.30)
        load_x = (desired_speed - speed) / (9.81 * dt)

        if key == "case2" and "case2_sin_A" in self.paper_preset.files:
            sign = preset_scalar("case2_sin_sign", 1.0)
            N = preset_scalar("case2_sin_N", 3.0)
            omega = preset_scalar("case2_sin_omega", self.case2_maneuver_freq)
            amp = preset_scalar("case2_sin_A", self.case2_lateral_amp)
            phase = preset_scalar("case2_sin_phase", 0.0)
            bias = preset_scalar("case2_sin_bias", 0.0)
            # Keep the identified sinusoidal penetration maneuver in the
            # midcourse, then fade it out near the protected asset so the
            # attacker still performs a terminal strike instead of orbiting.
            fade = np.clip(horizontal_dist / max(float(self.attack_maneuver_fade_range), 1.0), 0.0, 1.0)
            load_y = (N * sign * q_dot * speed) / 9.8
            load_y += fade * ((amp * np.sin(omega * agent.state.timestep + phase)) / 9.8 + bias)
        elif key == "case1" and "case1_nopn_const_load" in self.paper_preset.files:
            sign = preset_scalar("case1_nopn_sign", 1.0)
            N = preset_scalar("case1_nopn_N", 3.0)
            const_load = preset_scalar("case1_nopn_const_load", 0.0)
            switch_t = preset_scalar("case1_nopn_switch_time", 10.0)
            width = max(preset_scalar("case1_nopn_transition_width", 10.0), 1e-6)
            x = np.clip((agent.state.timestep - (switch_t - 0.5 * width)) / width, 0.0, 1.0)
            blend = x * x * (3.0 - 2.0 * x)
            pn_load = N * sign * q_dot * speed / 9.8
            load_y = (1.0 - blend) * const_load + blend * pn_load
        else:
            speed_ref = self.paper_preset[f"{key}_attacker_speed"]
            yaw_ref = self.paper_preset[f"{key}_attacker_yaw"]
            step = int(round(float(agent.state.timestep) / dt))
            next_step = min(step + 1, speed_ref.shape[0] - 1)
            desired_yaw = float(yaw_ref[next_step, j])
            yaw_error = _wrap_pi(desired_yaw - float(agent.state.v_vel[2]))
            load_y = yaw_error * speed * max(np.cos(pitch), 0.25) / (9.81 * dt)

        load_z = (desired_pitch - pitch) * speed / (9.81 * dt)

        yaw_scale = max(float(getattr(world, "attacker_yaw_scale", 1.0)), 1e-6)
        pitch_scale = max(float(getattr(world, "attacker_pitch_scale", 1.0)), 1e-6)
        agent.action.u = np.array([load_x, load_y / yaw_scale, load_z / pitch_scale], dtype=np.float32)
        return agent.action

    def _trust_guide_actions(self, agent, world):
        """Return clipped 3-D PN and boundary-probe commands for exploration.

        These commands are exposed only through ``info`` and are never part of
        the deployed learned policy.  The PN term uses the measured 3-D LOS
        rate, while the axial component reduces the same-target time-to-go
        mismatch.
        """
        target = world.agents[agent.target]
        rel = np.asarray(target.state.p_pos - agent.state.p_pos, dtype=float)
        dist = max(float(np.linalg.norm(rel)), 1e-6)
        los = rel / dist
        own_vel = np.asarray(agent.state.p_vel, dtype=float)
        target_vel = np.asarray(target.state.p_vel, dtype=float)
        rel_vel = target_vel - own_vel
        radial_rate = float(np.dot(rel_vel, los))
        closing = max(-radial_rate, 0.0)
        transverse_rel_vel = rel_vel - radial_rate * los
        nav_constant = 3.0
        pn_accel = nav_constant * closing * transverse_rel_vel / dist

        vhat = _unit(own_vel, self._flight_dir(agent))
        pn_accel = pn_accel - float(np.dot(pn_accel, vhat)) * vhat
        horizontal_vel = np.array([own_vel[0], own_vel[1], 0.0])
        hhat = _unit(horizontal_vel, np.array([1.0, 0.0, 0.0]))
        lateral_hat = np.array([-hhat[1], hhat[0], 0.0])

        tgo = dist / max(float(agent.state.v_vel[0]), 1.0)
        group_tgo = self._group_tgo_values(agent, world)
        tgo_mean = float(np.mean(group_tgo)) if group_tgo else tgo
        axial = np.clip(0.08 * (tgo - tgo_mean), -0.1, 1.0)
        pn = np.array(
            [
                axial,
                np.clip(np.dot(pn_accel, lateral_hat) / 9.81, -1.0, 1.0),
                np.clip(pn_accel[2] / 9.81, -1.0, 1.0),
            ],
            dtype=np.float32,
        )
        probe = np.array(
            [
                1.0 if tgo > tgo_mean else -0.1,
                1.0 if pn[1] >= 0.0 else -1.0,
                1.0 if pn[2] >= 0.0 else -1.0,
            ],
            dtype=np.float32,
        )
        return pn, probe

    def info(self, agent, world):
        pn, probe = self._trust_guide_actions(agent, world)
        return {
            "pn_action": pn,
            "probe_action": probe,
            "actual_hit": bool(getattr(agent.state, "actual_hit", False)),
            "hit_time": float(getattr(agent.state, "hit_time", np.nan)),
            "target_id": int(agent.target),
        }
