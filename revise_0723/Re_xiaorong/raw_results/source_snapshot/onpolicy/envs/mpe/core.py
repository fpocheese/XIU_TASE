import numpy as np
import math
from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenarios
from scipy.signal import butter, lfilter


# physical/external base state of all entites   所有实体的物理/外部基态
class EntityState(object):
    def __init__(self):
        # physical position
        self.p_pos = None
        # physical velocity
        self.p_vel = None

        # self.v_vel = None


# state of agents (including communication and internal/mental state)智能体们的状态（包括沟通和内部/精神状态）
class AgentState(EntityState):
    def __init__(self):
        super(AgentState, self).__init__()
        # communication utterance
        self.c = None
        self.v_vel = None
        self.dist0 = None
        self.dist1 = None
        self.lamda0 = None
        self.target = None
        self.done = None
        self.dist_target = None


# state of fixed wing agents固定翼智能体的状态
# 2020/04/10
class FighterState(EntityState):
    def __init__(self):
        super(FighterState, self).__init__()
        # trajectory coordination: velocity gamma psi  轨迹协调：速度伽玛 psi
        self.v_vel = None
        self.qddd = None
        self.qddd1 = None
        self.k3 = None
        self.k4 = None
        self.k5 = None
        self.dist0 = None
        self.dist1 = None
        self.time_tgo = None
        self.time_tgo_dist = None
        self.load = None
        self.last_loadx = None
        self.last_loady = None
        self.load_delt_all = None
        self.last_q_dot = None
        self.kalman_p_last = None
        self.namenumber = None
        self.doneflag_me_target = None
        self.eval = None
        self.eval_flag = None
        self.timestep = None
        self.timeover = None
        self.defender_sensor_history = None
        self.defender_command_lag_load = None

# action of the agent  智能体的动作
class Action(object):
    def __init__(self):
        # physical action身体的动作
        self.u = None
        # communication action沟通的动作
        self.c = None


# properties and state of physical world entity物理世界实体的属性和状态
class Entity(object):
    def __init__(self):
        # name 
        self.name = ''
        # properties:属性
        self.size = 0.050
        # entity can move / be pushed实体能不能动
        self.movable = False
        # entity collides with others 实体与他人发生碰撞
        self.collide = True
        # material density (affects mass)  材料密度（影响质量）
        self.density = 25.0
        # color  颜色
        self.color = None
        # max speed and accel  最大速度和加速度
        self.max_speed = None
        self.accel = None
        # state实体的状态，就是上面的实体状态的类
        self.state = EntityState()
        # mass  质量
        self.initial_mass = 1.0

    @property
    def mass(self):  #质量
        return self.initial_mass


# properties of landmark entities地标实体的属性  其实就是对实体类的一个继承
class Landmark(Entity):
    def __init__(self):
        super(Landmark, self).__init__()


# properties of agent entities 智能体实体的属性   也是对实体类的一个继承
class Agent(Entity):
    def __init__(self):
        super(Agent, self).__init__()
        # agents are movable by default  默认智能体是可以移动的
        self.movable = True
        # cannot send communication signals   无法发送通讯信号  这里是false说明可以发送信号
        self.silent = False
        # cannot observe the world   可以观察世界
        self.blind = False
        # physical motor noise amount  物理电机噪声量  咱这里没有
        self.u_noise = None
        # communication noise amount   通讯噪声  咱没有
        self.c_noise = None
        # control rangeEntity   控制范围实体
        self.u_range = 1.0
        # state   智能体的状态  用的是之前的类
        self.state = AgentState()
        # action  动作   用的是前面的动作类来代替的
        self.action = Action()
        # script behavior to execute  执行脚本的行为
        self.action_callback = None


# properties of fighter agent 战斗机智能体的特性   战斗机实体的类他所拥有的属性   是前面实体类的一个继承！
# 用于维护智能体的状态信息（agent.state, agent.action），以及生存状态（doneflag）
# 里面用到了之前定义的 FighterState() 来作为飞行器状态
##用到了之前定义的 Action() 来作为动作函数
# 2020/04/10
class FighterAgent(Entity):
    def __init__(self):
        super(FighterAgent, self).__init__()
        # agents are movable by default  默认智能体是可以移动的
        self.movable = True
        # cannot send communication signals  不能发送交流信号  咱这里不能
        self.silent = True
        # cannot observe the world   可以观测世界
        self.blind = False
        # physical motor noise amount   发动机的噪声  没有
        self.u_noise = None
        # communication noise amount    交流的噪声  没有
        self.c_noise = None
        # control rangeEntityu_range  控制实体的范围
        self.u_range = 1.0
        # stateu_range   战斗机的状态，咱使用到了之前定义的战斗机状态类，也就归属到战斗机实体的属性当中去
        self.state = FighterState()
        # action   动作   用的是之前的动作类
        self.action = Action()
        # script behavior to execute
        self.action_callback = None
        # self.action_callback = action_callback()
        # qi fei zhongliang   起飞时的重量
        self.takeoff_mass = 200000
        # maximum horizental load  最大的法向过载
        self.load_max = 5
        self.alpha = 0
        # done flag   做完的标志
        self.doneflag = False
        self.max_vel = 1.0
        self.timestep = 0
#战斗机初始的质量
    def flymass(self):
        return self.takeoff_mass



# multi fixed wing world
# 2020/04/10
class FighterWorld(object):
    def __init__(self):
        # list of agents and entities (can change at execution-time!)
        self.agents = []
        self.landmarks = []
        # communication channel dimensionality
        self.dim_c = 0
        # position dimensionality
        self.dim_p = 2
        # color dimensionality
        self.dim_color = 3
        # simulation timestep
        self.dt = 0.05
        # physical damping
        self.damping = 0.25
        # contact response parameters
        self.contact_force = 1e+2
        self.contact_margin = 1
        # zhpng li acc
        self.gravity_acc = 9.81
        # max vel and min vel
        self.velmin = 100
        self.velmax = 1000

        self.scale = 2000

        self.cutoff_frequency = 0.01
        self.load_x_filtered = []
        self.load_y_filtered = []
        self.b, self.a = butter(1, self.cutoff_frequency, fs=1 / self.dt)

    # return all entities in the world
    @property
    def entities(self):
        return self.agents + self.landmarks

    # return all agents controllable by external policies
    @property
    def policy_agents(self):
        return [agent for agent in self.agents if agent.action_callback is None]

    # return all agents controlled by world scripts
    @property
    def scripted_agents(self):
        return [agent for agent in self.agents if agent.action_callback is not None]

    def butter_filter(self, data, data_filtered):
        """使用一阶低通滤波器对数据进行滤波"""
        if not data_filtered:  # 初始化时使用初始值
            data_filtered = [data] * len(self.b)
        filtered = lfilter(self.b, self.a, [data] + data_filtered)
        return filtered[0], filtered[1:].tolist()

    # update state of the world
    def step(self):
        # set actions for scripted agents
        # for agent in self.scripted_agents:
        #     agent.action = agent.action_callback(agent, self)

##只有maddpg才需要这些人为噪声，别的算法是不需要这些噪声的
        # for agent in self.agents:
        #     if agent.movable:
        #         noise = 0.5 * np.random.randn(*agent.action.u.shape) * agent.u_noise if agent.u_noise else 0.0
        #         agent.action.u += noise

        for i, agent in enumerate(self.agents):
            if self.dim_p == 3:
                action = np.asarray(agent.action.u, dtype=float) if agent.action.u is not None else np.array([])
                if action.size >= 3:
                    load_y = float(action[1])
                    load_z = float(action[2])
                    load_x = float(action[0])
                elif action.size == 2:
                    # 2D overloading output: [pitch, yaw]
                    load_x = 0.0
                    load_y = float(action[0])
                    load_z = float(action[1])
                else:
                    load_x = load_y = load_z = 0.0

                if agent.adversary:
                    base_gain = float(getattr(self, "defender_guidance_base_gain", 0.0))
                    if base_gain > 0.0:
                        base_x, base_y, base_z = self.defender_base_load(agent)
                        residual_scale = float(getattr(self, "defender_residual_scale", 1.0))
                        load_x = base_gain * base_x + residual_scale * load_x
                        load_y = base_gain * base_y + residual_scale * load_y
                        load_z = base_gain * base_z + residual_scale * load_z
                    ref = getattr(self, "reference_control", None)
                    ref_blend = float(getattr(self, "defender_reference_blend", 0.0))
                    if ref is not None and ref_blend > 0.0:
                        step_idx = int(np.clip(round(agent.state.timestep / max(self.dt, 1e-6)), 0, ref.shape[0] - 1))
                        did = int(getattr(agent, "namenumber", i))
                        if did < ref.shape[1]:
                            ref_load = np.asarray(ref[step_idx, did], dtype=float)
                            beta = max(0.0, min(1.0, ref_blend))
                            load_x = (1.0 - beta) * load_x + beta * float(ref_load[0])
                            load_y = (1.0 - beta) * load_y + beta * float(ref_load[1])
                    axial_min = float(getattr(self, "defender_axial_min", -0.1))
                    axial_max = float(getattr(self, "defender_axial_max", 1.0))
                    load_x = max(axial_min, min(axial_max, load_x))
                    load_limit = float(getattr(self, "defender_load_limit", 1.0))
                    load_y = max(-load_limit, min(load_limit, load_y))
                    load_z = max(-load_limit, min(load_limit, load_z))
                    lag_tau = float(getattr(self, "defender_command_lag_tau", 0.0))
                    if lag_tau > 0.0:
                        prev = getattr(agent.state, "defender_command_lag_load", None)
                        if prev is None or len(prev) != 3:
                            prev = np.zeros(3, dtype=float)
                        cmd = np.array([load_x, load_y, load_z], dtype=float)
                        alpha = self.dt / max(lag_tau + self.dt, self.dt)
                        lagged = np.asarray(prev, dtype=float) + alpha * (cmd - np.asarray(prev, dtype=float))
                        lagged[0] = max(axial_min, min(axial_max, float(lagged[0])))
                        lagged[1] = max(-load_limit, min(load_limit, float(lagged[1])))
                        lagged[2] = max(-load_limit, min(load_limit, float(lagged[2])))
                        agent.state.defender_command_lag_load = lagged.copy()
                        load_x, load_y, load_z = float(lagged[0]), float(lagged[1]), float(lagged[2])
                else:
                    axial_min = float(getattr(self, "attacker_axial_min", -0.10))
                    axial_max = float(getattr(self, "attacker_axial_max", 1.0))
                    load_x = max(axial_min, min(axial_max, load_x))
                    attacker_yaw_scale = getattr(self, "attacker_yaw_scale", 1.0)
                    attacker_pitch_scale = getattr(self, "attacker_pitch_scale", 1.0)
                    attacker_load_limit = getattr(self, "attacker_load_limit", 1.0)
                    load_y = max(-attacker_load_limit, min(attacker_load_limit, load_y * attacker_yaw_scale))
                    load_z = max(-attacker_load_limit, min(attacker_load_limit, load_z * attacker_pitch_scale))

                if agent.state.load is None or len(agent.state.load) != 3:
                    agent.state.load = np.zeros(3)
                agent.state.load[0] = load_x
                agent.state.load[1] = load_y
                agent.state.load[2] = load_z

                if not agent.doneflag:
                    speed = max(float(agent.state.v_vel[0]), 1.0)
                    pitch = float(agent.state.v_vel[1])
                    yaw = float(agent.state.v_vel[2])

                    speed += load_x * self.gravity_acc * self.dt
                    if agent.adversary:
                        defender_speed_min = float(getattr(self, "defender_speed_min", 12.0))
                        defender_speed_max = float(getattr(self, "defender_speed_max", 40.0))
                        speed = max(defender_speed_min, min(defender_speed_max, speed))
                    else:
                        speed_min = float(getattr(self, "attacker_speed_min", 12.0))
                        speed_max = float(getattr(self, "attacker_speed_max", 32.0))
                        speed = max(speed_min, min(speed_max, speed))
                    pitch += (self.gravity_acc * load_z / max(speed, 1.0)) * self.dt
                    pitch = max(-0.45 * np.pi, min(0.45 * np.pi, pitch))
                    yaw += (self.gravity_acc * load_y / max(speed * max(np.cos(pitch), 0.25), 1.0)) * self.dt

                    agent.state.v_vel[0] = speed
                    agent.state.v_vel[1] = pitch
                    agent.state.v_vel[2] = yaw
                    pos_dot = self.state_dot_rotation(agent)
                    agent.state.p_vel = pos_dot
                    agent.state.p_pos += agent.state.p_vel * self.dt
                    if agent.state.p_pos[2] < 0.0:
                        agent.state.p_pos[2] = 0.0
                        if agent.state.p_vel[2] < 0.0:
                            agent.state.p_vel[2] = 0.0
                continue

            if agent.adversary:

                load_y = agent.action.u[1]
                load_x = agent.action.u[0]

                # # 对输入进行低通滤波
                # load_x_filtered, self.load_x_filtered = self.butter_filter(load_x, self.load_x_filtered)
                # load_y_filtered, self.load_y_filtered = self.butter_filter(load_y, self.load_y_filtered)

                # 限幅操作
                if agent.state.dist_target > 600 :
                    load_x_filtered = max(-0.01, min(1, load_x))
                    load_y_filtered = max(-1, min(1, load_y))
                else:
                    # load_x_filtered = max(-0.01, min(1, load_x))
                    # load_y_filtered = max(-1, min(1, load_y))
                    # ##------------------滤波方法-----------------------#
                    k_filter = 1.0
                    load_y_filtered = k_filter * load_y + (1 - k_filter) * agent.state.load[1]
                    load_x_filtered = k_filter * load_x + (1 - k_filter) * agent.state.load[0]
                    load_x_filtered = max(-0.01, min(1, load_x_filtered))
                    load_y_filtered = max(-1, min(1, load_y_filtered))

                # ###----------------------传统方法协同制导----------------------####
                # load_y_filtered = max(-1, min(1, agent.state.k3[0]))
                # acc_x = (agent.state.time_tgo_dist[1] - agent.state.v_vel[0]) / self.dt
                # load_x_filtered = max(-0.01, min(1, acc_x / self.gravity_acc))
                # # load_x_filtered = 0
                # ##----------------------传统方法协同制导----------------------####

                if agent.state.v_vel[0] == 40:
                    load_x_filtered = 0

                agent.state.load[0] = load_x_filtered
                agent.state.load[1] = load_y_filtered
                vx_dot = load_x_filtered*self.gravity_acc
                vy_dot = (self.gravity_acc * load_y_filtered) / (agent.state.v_vel[0])

                if not agent.doneflag:
                    agent.state.v_vel[0] += vx_dot * self.dt
                    agent.state.v_vel[1] += vy_dot * self.dt
                    # if agent.state.v_vel[0] < 20:
                    #     agent.state.v_vel[0] = 20
                    if agent.state.v_vel[0] > 40:
                        agent.state.v_vel[0] = 40
                    pos_dot = self.state_dot_rotation(agent)
                    agent.state.p_vel = pos_dot
                    agent.state.p_pos += agent.state.p_vel * self.dt

            if not agent.adversary:

                load_y = agent.action.u[0]
                load_z = agent.action.u[1]
                attacker_yaw_scale = getattr(self, "attacker_yaw_scale", 1.0)
                attacker_pitch_scale = getattr(self, "attacker_pitch_scale", 1.0)
                attacker_load_limit = getattr(self, "attacker_load_limit", 1.0)
                load_y = float(load_y) * attacker_yaw_scale
                load_z = float(load_z) * attacker_pitch_scale
                load_y = max(-attacker_load_limit, min(attacker_load_limit, load_y))
                load_z = max(-attacker_load_limit, min(attacker_load_limit, load_z))
                agent.state.load[0] = load_y
                agent.state.load[1] = load_z
                if not agent.doneflag:
                    speed = max(float(agent.state.v_vel[0]), 1.0)
                    pitch = float(agent.state.v_vel[1])
                    yaw = float(agent.state.v_vel[2])
                    pitch += (self.gravity_acc * load_z / max(speed, 1.0)) * self.dt
                    pitch = max(-0.45 * np.pi, min(0.45 * np.pi, pitch))
                    yaw += (self.gravity_acc * load_y / max(speed * max(np.cos(pitch), 0.25), 1.0)) * self.dt
                    agent.state.v_vel[0] = speed
                    agent.state.v_vel[1] = pitch
                    agent.state.v_vel[2] = yaw
                    pos_dot = self.state_dot_rotation(agent)
                    agent.state.p_vel = pos_dot
                    agent.state.p_pos += agent.state.p_vel * self.dt

        # update agent communication state
        for agent in self.agents:
            self.update_agent_state(agent)
            agent.state.timestep += self.dt





    def _unit3(self, vec, fallback):
        norm = np.linalg.norm(vec)
        if norm < 1e-9:
            return np.array(fallback, dtype=float)
        return vec / norm

    def _defender_sensed_target_state(self, agent, target):
        delay_steps = max(0, int(getattr(self, "defender_sensor_delay_steps", 0)))
        pos_std = max(0.0, float(getattr(self, "defender_obs_pos_noise_std", 0.0)))
        vel_std = max(0.0, float(getattr(self, "defender_obs_vel_noise_std", 0.0)))
        if delay_steps <= 0 and pos_std <= 0.0 and vel_std <= 0.0:
            return target.state.p_pos, target.state.p_vel

        hist = getattr(agent.state, "defender_sensor_history", None)
        if hist is None:
            hist = []
        target_vel = target.state.p_vel if target.state.p_vel is not None else np.zeros(3)
        hist.append((target.state.p_pos.copy(), target_vel.copy()))
        max_len = delay_steps + 1
        if len(hist) > max_len:
            hist = hist[-max_len:]
        agent.state.defender_sensor_history = hist
        sensed_pos, sensed_vel = hist[0] if len(hist) <= delay_steps else hist[-max_len]
        sensed_pos = sensed_pos.copy()
        sensed_vel = sensed_vel.copy()
        if bool(getattr(self, "defender_sensor_delay_compensate", False)) and delay_steps > 0:
            sensed_pos += sensed_vel * (delay_steps * self.dt)
        if pos_std > 0.0:
            sensed_pos += np.random.normal(0.0, pos_std, size=3)
        if vel_std > 0.0:
            sensed_vel += np.random.normal(0.0, vel_std, size=3)
        alpha = float(getattr(self, "defender_obs_filter_alpha", 1.0))
        if alpha < 1.0:
            alpha = max(0.0, alpha)
            prev = getattr(agent.state, "defender_sensor_filtered", None)
            if prev is None:
                filt_pos, filt_vel = sensed_pos, sensed_vel
            else:
                filt_pos = np.asarray(prev[0], dtype=float) + alpha * (sensed_pos - np.asarray(prev[0], dtype=float))
                filt_vel = np.asarray(prev[1], dtype=float) + alpha * (sensed_vel - np.asarray(prev[1], dtype=float))
            agent.state.defender_sensor_filtered = (filt_pos.copy(), filt_vel.copy())
            sensed_pos, sensed_vel = filt_pos, filt_vel
        return sensed_pos, sensed_vel

    def defender_base_load(self, agent):
        try:
            target = self.agents[int(agent.target)]
        except Exception:
            return 0.0, 0.0, 0.0
        if target.state.p_pos is None or agent.state.p_pos is None:
            return 0.0, 0.0, 0.0

        target_pos, target_vel = self._defender_sensed_target_state(agent, target)
        rel = target_pos - agent.state.p_pos
        dist = max(float(np.linalg.norm(rel)), 1e-6)
        los = rel / dist
        own_vel = agent.state.p_vel if agent.state.p_vel is not None else self.state_dot_rotation(agent)
        speed = max(float(agent.state.v_vel[0]), 1.0)
        rel_vel = target_vel - own_vel
        closing = max(-float(np.dot(rel_vel, los)), 1.0)
        lead_scale = float(getattr(self, "defender_guidance_lead", 1.0))
        lead_time = np.clip(lead_scale * dist / max(speed + closing, 1.0), 0.0, 80.0)
        aim_point = target_pos + target_vel * lead_time
        desired_dir = self._unit3(aim_point - agent.state.p_pos, los)
        desired_vel = speed * desired_dir
        tau = max(float(getattr(self, "defender_guidance_tau", 1.2)), self.dt)
        acc_cmd = (desired_vel - own_vel) / tau

        vhat = self._unit3(own_vel, self.state_dot_rotation(agent))
        normal_acc = acc_cmd - np.dot(acc_cmd, vhat) * vhat
        horizontal_vel = np.array([own_vel[0], own_vel[1], 0.0])
        hhat = self._unit3(horizontal_vel, [1.0, 0.0, 0.0])
        lhat = np.array([-hhat[1], hhat[0], 0.0])
        load_y = float(np.dot(normal_acc, lhat) / self.gravity_acc)
        load_z = float(normal_acc[2] / self.gravity_acc)
        tgo = dist / max(speed, 1.0)
        group_tgo = []
        for other in self.agents:
            if not getattr(other, "adversary", False) or other.target != agent.target or other.doneflag:
                continue
            other_dist = np.linalg.norm(target_pos - other.state.p_pos)
            group_tgo.append(other_dist / max(float(other.state.v_vel[0]), 1.0))
        ref_mode = str(getattr(self, "defender_sync_tgo_ref", "mean")).lower()
        if group_tgo:
            if ref_mode == "max":
                tgo_ref = float(np.max(group_tgo))
            elif ref_mode == "min":
                tgo_ref = float(np.min(group_tgo))
            else:
                tgo_ref = float(np.mean(group_tgo))
        else:
            tgo_ref = tgo
        sync_gain = float(getattr(self, "defender_sync_speed_gain", 0.0))
        load_x = sync_gain * (tgo - tgo_ref)

        speed_target = float(getattr(self, "defender_speed_target", 0.0))
        speed_gain = float(getattr(self, "defender_speed_gain", 0.0))
        if speed_target > 0.0 and speed_gain > 0.0:
            speed_load = speed_gain * (speed_target - speed) / self.gravity_acc
            if speed < speed_target:
                speed_load = max(speed_load, float(getattr(self, "defender_min_accel_load", 0.0)))
            load_x += speed_load
        return load_x, load_y, load_z

    def state_dot_rotation(self, agent):
        if self.dim_p == 3:
            px_dot = agent.state.v_vel[0] * np.cos(agent.state.v_vel[1]) * np.cos(agent.state.v_vel[2])
            py_dot = agent.state.v_vel[0] * np.cos(agent.state.v_vel[1]) * np.sin(agent.state.v_vel[2])
            pz_dot = agent.state.v_vel[0] * np.sin(agent.state.v_vel[1])
            return np.array([px_dot, py_dot, pz_dot])
        px_dot = agent.state.v_vel[0] * np.cos(agent.state.v_vel[1])
        py_dot = agent.state.v_vel[0] * np.sin(agent.state.v_vel[1])
        p_dot = np.array([px_dot, py_dot])
        return p_dot



    def update_agent_state(self, agent):
        # set communication state (directly for now)
        if agent.silent:
            agent.state.c = np.zeros(self.dim_c)
        else:
            noise = np.random.randn(*agent.action.c.shape) * agent.c_noise if agent.c_noise else 0.0
            agent.state.c = agent.action.c + noise
