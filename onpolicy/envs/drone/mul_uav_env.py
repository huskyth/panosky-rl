import copy
import json
import math
import random
import warnings

import os
import numpy as np
from gym import spaces
from onpolicy.utils.format_logger import AppLogger
from onpolicy.envs.drone.weapons.entries.uav.uav_enum import UAVState, AttackState
from onpolicy.envs.drone.maps.map import Map
from onpolicy.utils.util import compute_distance
from onpolicy.envs.drone.uav_meta_info import TrainUAV
from pathlib import Path
from onpolicy.envs.drone.weapons.interfaces.environment_interface import EnvironmentInterface
from onpolicy.utils.math_tool import fly_from_9_selections, angle_2_radian, length_of_vector, normalize

warnings.filterwarnings('ignore')
logger = AppLogger().get_logger()


def cos_theta(target, position, v):
    to = np.array(target) - np.array(position)
    cos_ = (to * np.array(v)).sum() / (to ** 2).sum() ** 0.5 / (np.array(v) ** 2).sum() ** 0.5
    logger.info(f"cos = {cos_}")
    return cos_


json_path = Path(__file__).parent / "jsons"
if not json_path.exists():
    json_path.mkdir()


# 武器环境已知的环境
class MultiUavEnv:
    # 这边先垂直后水平，有问题再说
    ACTION_SET = [
        (angle_2_radian(10), angle_2_radian(-10)),
        (angle_2_radian(10), angle_2_radian(0)),
        (angle_2_radian(10), angle_2_radian(10)),
        (angle_2_radian(0), angle_2_radian(-10)),
        (angle_2_radian(0), angle_2_radian(0)),
        (angle_2_radian(0), angle_2_radian(10)),
        (angle_2_radian(-10), angle_2_radian(-10)),
        (angle_2_radian(-10), angle_2_radian(0)),
        (angle_2_radian(-10), angle_2_radian(10)),
    ]

    def dump(self, reason):
        if self.is_debug and len(self.episode_data):
            with open(
                    str(json_path / f"{self.mode}模式下第{self.n_episode}个Epoch一共{self._episode_steps}步，【结果】：{reason.split('-')[0]}.json"),
                    'w',
                    encoding="utf-8") as f:
                data = {'target': self.target, 'weapon': self.weapon, 'episode_data': self.episode_data,
                        "reason": reason,
                        "map_max_x": self.map.map_max_x,
                        "map_max_y": self.map.map_max_y,
                        "max_available_height": self.max_available_height,
                        "n_uav": self.n_total_uavs,
                        }

                json.dump(data, f, ensure_ascii=False)

                self.episode_data = []
                import gc
                gc.collect()

    def init_from_config(self, cf):
        self.map_output_dimension = cf.getint("env", "map_output_dimension")
        self.task_success_radius = cf.getfloat("env", "task_success_radius")  # 任务成功半径 m
        self.uav_obs_radius = cf.getfloat("env", "uav_obs_radius")  # 局部观测半径 m
        self.uav_velocity_value = cf.getfloat("env", "uav_velocity_value")  # 飞行速度 m/s
        self.uav_length = cf.getfloat("env", "uav_length")  # 无人机长度
        self.coll_safe_dis = cf.getfloat("env", "coll_safe_dis")  # 碰撞安全距离
        self.dis_target_weapon = cf.getfloat("env", "dis_target_weapon")  # 目标和武器的距离
        self.dis_target_uav_min = cf.getint("env", "dis_target_uav_min")  # 目标和UAV的最小距离
        self.dis_target_uav_max = cf.getint("env", "dis_target_uav_max")  # 目标和UAV的最大距离

        self.uva_init_height = cf.getfloat("env", "uva_init_height")  # 无人机初始高度

        self.max_available_height = cf.getfloat("constraints", "max_available_height")  # 最高可飞行高度
        self.min_available_height = cf.getfloat("constraints", "min_available_height")  # 最低可飞行高度

        # 有关奖励函数的参数
        self.decoy_safe_dis = cf.getfloat("reward", "decoy_safe_dis")  # 引诱的安全距离
        self.uav_coll_penalty = cf.getfloat("reward", "uav_coll_penalty")  # UAV碰撞时的惩罚
        self.step_penalty = cf.getfloat("reward", "step_penalty")  # 执行动作的惩罚
        self.decoy_success_reward = cf.getfloat("reward", "decoy_success_reward")  # 引诱成功的奖励
        self.decoy_dis_reward_index = cf.getfloat("reward", "decoy_dis_reward_index")  # 引诱距离奖励的比例系数
        self.task_success_reward = cf.getfloat("reward", "task_success_reward")  # 任务成功的奖励
        self.task_dis_reward_index = cf.getfloat("reward", "task_dis_reward_index")  # 任务距离奖励的比例系数

        # 计算无人机数量
        self.n_task_uavs = cf.getint("env", "n_task_uavs")  # 任务机数量
        self.n_decoy_uavs = cf.getint("env", "n_decoy_uavs")  # 诱饵机数量
        self.n_total_uavs = self.n_task_uavs + self.n_decoy_uavs  # 无人机总数量

        # 初始化地图
        map_data_file = cf.get("map", "map_data_file")  # 地图文件路径
        map_resolution = cf.getfloat("map", "map_resolution")  # 地图分辨率
        self.map = Map(map_data_file, map_resolution, self.uav_length)

        assert self.task_success_radius > self.uav_velocity_value

    def init_space(self):
        # 定义状态空间：线性数据 + 图像

        # configure spaces
        self.action_space = []
        self.observation_space = []
        self.share_observation_space = []
        obs_dim = None
        for agent_id in range(self.n_total_uavs):
            self.action_space.append(spaces.Discrete(9))
            obs_dim = self.get_observation_size_of_a_uav()
            observation_space = spaces.Dict({
                # 1) 线性数据：一维连续向量（Box）
                "linear": spaces.Box(low=-np.inf, high=+np.inf, shape=(obs_dim,), dtype=np.float32),

                # 2) 图像数据：高/宽/通道（最常用格式 HWC）
                "image": spaces.Box(
                    low=0,  # 像素 0~255
                    high=255,
                    shape=(1, 32, 32),  # 高84 × 宽84 × 3通道(RGB)
                    dtype=np.uint8
                )
            })
            self.observation_space.append(observation_space)
        self.share_observation_space = [
            spaces.Dict({
                # 1) 线性数据：一维连续向量（Box）
                "linear": spaces.Box(low=-np.inf, high=+np.inf, shape=(self.n_total_uavs * obs_dim,), dtype=np.float32),

                # 2) 图像数据：高/宽/通道（最常用格式 HWC）
                "image": spaces.Box(
                    low=0,  # 像素 0~255
                    high=255,
                    shape=(self.n_total_uavs, 32, 32),  # 高84 × 宽84 × 3通道(RGB)
                    dtype=np.uint8
                )
            })
            for _ in range(self.n_total_uavs)]

    def __init__(self, rank, mode="train", cf=None, episode_limit=500, is_debug=False):
        self.rank = rank
        # train为训练模式，test为测试模式
        self.mode = mode
        self.is_debug = is_debug
        self.is_use_weapon = True

        self.right_vector = None

        # 初始化回合数
        self.n_episode = 0
        self._episode_steps = 0
        self.max_episode_steps = episode_limit  # 任务的最大回合数
        # 从游戏获取参数并保存
        self.radar_detect_radius = EnvironmentInterface.get_rader_detect_radius()  # 雷达检测半径 m
        self.weapon_fire_radius = EnvironmentInterface.get_fire_distance()  # 武器开火半径 m

        # 初始化无人机集群
        self.raw_uavs = []
        self.episode_data = []
        # 初始化目标和武器
        self.target = [0, 0, 0]
        self.weapon = [0, 0, 0]

        self.init_from_config(cf)
        self.init_space()

        self.reward = None
        self.is_terminal = [False for _ in range(self.n_total_uavs)]

        logger.info(
            f"PID-{os.getpid()}, 【{'训练' if self.mode == 'train' else '评估'}】环境（武器位置已知）初始化完成，"
            f"任务机数量为{self.n_task_uavs}，诱饵机数量为{self.n_decoy_uavs}，是否使用武器【{self.is_use_weapon}】")

    def reset(self, tar_pos=None, wea_pos=None, uav_pos=None):
        """
        每个UAV的状态包含 [x, y, z, v_x, v_y, v_z, decoy_flag, attacked_state, status]
        decoy_flag表示无人机是否为诱饵机，诱饵机为1，任务机为2
        attacked_state，不被攻击为0，被攻击为1，被摧毁为2
        status表示无人机状态，存活为"Alive"，死亡为"UAV_collision/Ground_collision/Destroyed"

        训练环境随机生成武器、目标和无人机集群的初始位置
        测试环境需要中通过传参控制武器、目标和无人机集群的初始位置
        """

        self.target = [0, 0, 0]
        self.weapon = [0, 0, 0]
        self.raw_uavs = []
        self.episode_data = []  # 存储历史的数据
        self._episode_steps = 0
        self.right_vector = [[1, 0, 0] for i in range(self.n_total_uavs)]
        self.reward = None
        self.is_terminal = [False for _ in range(self.n_total_uavs)]

        # 重置目标位置
        # 为了保证目标和武器都在地图内，预留dis_target_weapon的空间

        self.target[0] = float(random.uniform(self.map.map_min_x + self.dis_target_weapon,
                                              self.map.map_max_x - self.dis_target_weapon))
        self.target[1] = float(random.uniform(self.map.map_min_y + self.dis_target_weapon,
                                              self.map.map_max_y - self.dis_target_weapon))

        self.target[2] = float(self.map.search_nh(self.target[0], self.target[1]) + self.coll_safe_dis)
        # 重置武器位置 theta 为 目标和武器连线与X轴正向夹角 范围为 [0，2pai）
        theta_w = 2 * math.pi * random.uniform(0, 1)
        self.weapon[0] = float(self.target[0] + self.dis_target_weapon * math.cos(theta_w))
        self.weapon[1] = float(self.target[1] + self.dis_target_weapon * math.sin(theta_w))
        self.weapon[2] = float(self.map.search_nh(self.weapon[0], self.weapon[1]) + self.coll_safe_dis)
        # 你的循环代码（已修改为 3D 真实距离）
        for uav in range(self.n_total_uavs):
            uav_x = 0.0
            uav_y = 0.0
            uav_z = 0.0
            available = False

            # 目标点坐标（确保是3维）
            target_x, target_y, target_z = self.target[0], self.target[1], self.target[2]

            while not available:
                # --------------------------
                # 步骤1：随机生成 严格3D距离 r
                # --------------------------
                r = random.uniform(self.dis_target_uav_min, self.dis_target_uav_max)

                # --------------------------
                # 步骤2：先生成随机方向向量 (dx, dy, dz)
                # --------------------------
                dx = random.uniform(-1, 1)
                dy = random.uniform(-1, 1)
                dz = random.uniform(-1, 1)

                # --------------------------
                # 步骤3：归一化向量 → 长度=1
                # --------------------------
                norm = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
                dx_norm = dx / norm
                dy_norm = dy / norm
                dz_norm = dz / norm

                # --------------------------
                # 步骤4：缩放为 严格3D距离 = r
                # --------------------------
                uav_x = target_x + r * dx_norm
                uav_y = target_y + r * dy_norm
                uav_z = target_z + r * dz_norm

                # --------------------------
                # 步骤5：应用你的安全高度限制（关键！限制后重新校准距离）
                # --------------------------
                # 最低安全高度（地面+安全距离）
                min_safe_z = float(self.map.search_nh(uav_x, uav_y) + 2 * self.coll_safe_dis)
                # 无人机最低初始高度
                min_z_limit = max(self.uva_init_height, min_safe_z)

                # 如果当前高度低于限制 → 抬高到最低高度，然后重新校准3D距离
                if uav_z < min_z_limit:
                    uav_z = min_z_limit

                    # 重新计算水平分量，保证 3D 距离 严格 = r
                    dz_new = uav_z - target_z
                    remaining_2d = math.sqrt(r ** 2 - dz_new ** 2)

                    # 归一化水平方向
                    xy_norm = math.sqrt(dx_norm ** 2 + dy_norm ** 2)
                    uav_x = target_x + remaining_2d * (dx_norm / xy_norm)
                    uav_y = target_y + remaining_2d * (dy_norm / xy_norm)

                # --------------------------
                # 步骤6：判断位置是否可用
                # --------------------------
                available = self.judge_random_position_available(uav_x, uav_y, uav_z)
            # 初始化速度（原有逻辑）
            init_vel = self._init_toward_velocity([uav_x, uav_y, uav_z], self.target)
            temp_uav = TrainUAV(uav_x, uav_y, uav_z, *init_vel, AttackState.SAFE, UAVState.ALIVE)
            self.raw_uavs.append(temp_uav)

        # 把无人机总数量、UAV初始位置、武器初始位置传给游戏
        if self.is_use_weapon:
            EnvironmentInterface.reset(self.n_total_uavs, self.weapon,
                                       [self.raw_uavs[i].velocity for i in range(self.n_total_uavs)],
                                       [self.raw_uavs[i].position for i in range(self.n_total_uavs)])

        data_save = {"uva_state": [x.to_dict() for x in self.raw_uavs],
                     "uva_actions": [-1 for _ in range(self.n_total_uavs)],
                     "_episode_steps": self._episode_steps}
        self.episode_data.append(data_save)

        # 获取UAV集群观测并返回

        return self.get_state_of_all_uav()

    def _init_toward_velocity(self, uav, target):
        init_vel = [0, 0, 0]
        three_dim_dis = compute_distance(uav, target)
        init_vel[2] = (target[2] - uav[2]) * self.uav_velocity_value / three_dim_dis
        sin_beta = (target[2] - uav[2]) / three_dim_dis
        cos_beta = (1 - sin_beta ** 2) ** 0.5
        two_dim_dis = ((target[0] - uav[0]) ** 2 + (target[1] - uav[1]) ** 2) ** 0.5
        init_vel[0] = (target[0] - uav[0]) * self.uav_velocity_value * cos_beta / two_dim_dis
        init_vel[1] = (target[1] - uav[1]) * self.uav_velocity_value * cos_beta / two_dim_dis
        return init_vel

    # 判断随机生成的UAV位置是否可用
    def judge_random_position_available(self, uav_x, uav_y, uav_z):
        if uav_x < self.map.map_min_x or uav_x > self.map.map_max_x:
            return False
        if uav_y < self.map.map_min_y or uav_y > self.map.map_max_y:
            return False
        available = True
        if len(self.raw_uavs) > 0:
            # 循环当前无人机列表判断是否位置是否会相互碰撞
            for i in range(len(self.raw_uavs)):
                dis = compute_distance([uav_x, uav_y, uav_z], self.raw_uavs[i].position)
                # 如果距离会导致与当前任何一架无人机碰撞则返回False重新生成
                if dis < self.uav_length + self.coll_safe_dis:
                    available = False
        return available

    def get_state_of_all_uav(self):
        return [self.get_observation_of_a_uav(i) for i in range(self.n_total_uavs)]

    def get_observation_of_a_uav(self, uav_id):
        assert self.is_use_weapon is True
        assert self.n_total_uavs == 2
        normal_ = np.array([self.map.map_max_x, self.map.map_max_y, self.max_available_height])
        # 友军的位置和速度、受攻击状态
        # TODO://待检查，友军和武器有关，观测半径可以后续再加
        position = self.raw_uavs[uav_id].position
        map_obs = self.map.generate_img(position[0], position[1])[0][0]

        team = []
        for i in range(self.n_total_uavs):
            # 本机位置和速度
            temp_uav = self.raw_uavs[i]
            position = temp_uav.get_normalize_position(normal_)
            velocity = temp_uav.get_normalize_velocity()
            right_vector = normalize(self.right_vector[i])
            team += position
            team += velocity
            team += right_vector
            team += [temp_uav.is_attacked_state.value]
            team += [1 if i == 0 else 0]

        team += [uav_id]
        weapon = (np.array(self.weapon) / normal_).tolist()
        target = (np.array(self.target) / normal_).tolist()
        team += weapon
        team += target
        return team, [map_obs]

    def step(self, action):
        # 输入动作先转换为两个离散动作
        self.is_terminal = [False for _ in range(self.n_total_uavs)]
        actions = []
        # 对于每个智能体
        for idx in range(self.n_total_uavs):
            action_index = int(np.where(action[idx] == 1)[0])
            actions.append([action_index])

        # 获取执行动作前的UAV集群观测
        last_state = copy.deepcopy(self.raw_uavs)
        # 计算每架UAV执行动作后的速度、位置、碰撞情况以及与武器之间是否存在山体遮挡
        for idx in range(self.n_total_uavs):
            if self.raw_uavs[idx].status != UAVState.ALIVE:
                continue
            actual_action = actions[idx][0]
            radian_action = self.ACTION_SET[actual_action]

            # 计算执行动作之后的速度和位置
            temp = self.raw_uavs[idx]
            velocity = temp.velocity
            position = temp.position
            right = self.right_vector[idx]
            next_position, after_rotate_velocity_direction_in_parent, after_rotate_horizontal_right_vector_in_parent \
                = fly_from_9_selections(*radian_action, velocity, right, position)
            actual_velocity = (np.array(after_rotate_velocity_direction_in_parent) * self.uav_velocity_value).tolist()
            self.right_vector[idx] = after_rotate_horizontal_right_vector_in_parent
            # 判断是否会发生碰撞
            self.judge_uav_collision_and_set(idx, *next_position)

            # 执行动作改变UAV状态
            self.raw_uavs[idx].set_position(*next_position)
            self.raw_uavs[idx].set_velocity(*actual_velocity)
        self._episode_steps += 1  # 当前的回合数增加上动作执行的回合数。

        # 把位置传给游戏，然后获取执行动作后的受击状态并保存
        # TODO://待检查，先注释，训练完成了再解开
        if self.is_use_weapon:
            position = [self.raw_uavs[u].position for u in range(self.n_total_uavs)]
            velocity = [self.raw_uavs[u].velocity for u in range(self.n_total_uavs)]
            game_uav_list = EnvironmentInterface.step(position, velocity)
            for u in range(self.n_total_uavs):
                if game_uav_list[u] is None:
                    self.raw_uavs[u].is_attacked_state = AttackState.DESTROYED
                    self.raw_uavs[u].status = UAVState.DESTROYED
                else:
                    self.raw_uavs[u].is_attacked_state = game_uav_list[u].get_attacked_state()
                if self.raw_uavs[u].is_attacked_state == AttackState.DESTROYED:
                    self.raw_uavs[u].status = UAVState.DESTROYED
        data_save = {"uva_state": [x.to_dict() for x in self.raw_uavs], "uva_actions": action.tolist(),
                     "_episode_steps": self._episode_steps}
        self.episode_data.append(data_save)
        # 计算奖励值和终止符号
        self.set_reward(last_state)
        ret_reward = [[x] for x in self.reward]
        return self.get_state_of_all_uav(), ret_reward, self.is_terminal, [self.raw_uavs[i].status.value for i in
                                                                           range(self.n_total_uavs)]

    def judge_uav_collision_and_set(self, current_uav_idx, next_x, nex_y, next_z):
        p_x, p_y, p_z = self.raw_uavs[current_uav_idx].position
        # 判断是否与地面发生碰撞
        if self.map.judge_mountain(p_x, p_y, p_z, next_x, nex_y, next_z, self.coll_safe_dis, 'ground'):
            # 如果路径中碰撞也算
            self.raw_uavs[current_uav_idx].status = UAVState.GROUND_COLLISION
        # 判断是否与无人机发生碰撞
        for u_idx in range(self.n_total_uavs):
            temp = self.raw_uavs[u_idx]
            # 非本机
            if u_idx == current_uav_idx:
                continue
            if temp.status != UAVState.ALIVE:
                continue
            if compute_distance([next_x, nex_y, next_z], temp.position) < self.uav_length + self.coll_safe_dis:
                self.raw_uavs[u_idx].status = UAVState.UAV_COLLISION
                self.raw_uavs[current_uav_idx].status = UAVState.UAV_COLLISION

        # 判断是否超界
        if (
                next_x < self.map.map_min_x or next_x > self.map.map_max_x or nex_y < self.map.map_min_y or nex_y > self.map.map_max_y
                or next_z < self.min_available_height or next_z > self.max_available_height):
            self.raw_uavs[current_uav_idx].status = UAVState.GROUND_COLLISION

    def set_reward(self, last_p):
        # TODO://待检查
        """
        终止条件：
            - 无任务机存活
            - 任务机完成任务
        诱饵机奖励：
            - 引诱完成奖励：第一次到达雷达检测半径和武器开火半径中间
            - 引诱距离奖励: 鼓励任务机到目标的平均距离 - 诱饵机到武器的距离 <= decoy_safe_dis
            - 碰撞惩罚：发生碰撞
        任务机奖励：
            - 任务完成奖励：诱饵机完成引诱且任务机完成任务
            - 目标距离奖励: 鼓励任务机靠近目标点
            - 山体遮挡奖励: 无人机在武器检测范围内且存在山体遮挡
            - 碰撞惩罚：发生碰撞
        """
        # TODO://待检查，诱饵机奖励
        current_p = self.raw_uavs

        if self.is_use_weapon:
            self.reward = [0 for _ in range(self.n_total_uavs)]
            assert self.n_total_uavs == 2

            current_distance_to_target_1 = compute_distance(current_p[1].position, self.target)

            if current_p[1].status != UAVState.ALIVE:
                self.reward = [0 for _ in range(self.n_total_uavs)]
                msg = f'任务机败亡-任务机1状态为：{current_p[1].status.value}'
                msg += f'-奖励-{self.reward}'
                self.n_episode = self.n_episode + 1
                self.is_terminal = [True for _ in range(self.n_total_uavs)]
                logger.info(
                    f"PID-{os.getpid()}, mode-{self.mode}, episode-{self.n_episode}\033[31m[terminated]：{msg}\033[0m")
                self.dump(msg)
                return

            if current_p[0].is_attacked_state in [AttackState.ATTACKING,
                                                  AttackState.DESTROYED] and current_distance_to_target_1 <= self.task_success_radius:
                self.is_terminal = [True for _ in range(self.n_total_uavs)]
                self.reward = [self.task_success_reward for _ in range(self.n_total_uavs)]
                logger.info(
                    f"PID-{os.getpid()}, mode-{self.mode}, episode-{self.n_episode}, \033[32m[terminated]："
                    f"任务完成-UAV索引-【1】\033[0m，受攻击状态为 {current_p[0].is_attacked_state}")
                self.dump("任务完成")
                self.n_episode = self.n_episode + 1
                return



        else:
            assert self.n_total_uavs == 1
            self.reward = [0]

            uav = current_p[0]
            if uav.status != UAVState.ALIVE:
                self.reward = [0]
                self.n_episode = self.n_episode + 1
                self.is_terminal = [True]
                state = uav.status.value
                msg = f'{self._episode_steps} step：无人机伤亡，无法完成任务, {state}'
                logger.info(f"【PID: {os.getpid()}】\033[91m{msg}\033[0m")
                self.dump(msg)
                return

            current_distance_to_target = compute_distance(uav.position, self.target)
            if uav.status == UAVState.ALIVE and current_distance_to_target <= self.task_success_radius:
                self.reward[0] = 1
                self.n_episode = self.n_episode + 1
                msg = "任务完成"
                self.is_terminal = [True for _ in range(self.n_total_uavs)]
                logger.info(f"【PID: {os.getpid()}】\033[92m{self._episode_steps} step：任务完成\033[0m")
                self.dump(msg)
                return

        if self._episode_steps >= self.max_episode_steps:
            self.reward = [0 for _ in range(self.n_total_uavs)]
            msg = f'{self._episode_steps} step：超出最大步数限制'
            msg += f'-奖励-{self.reward}'
            self.n_episode = self.n_episode + 1
            self.is_terminal = [True for _ in range(self.n_total_uavs)]
            logger.info(
                f"PID-{os.getpid()}, mode-{self.mode}, "
                f"episode-{self.n_episode}\033[31m[terminated]："
                f"{msg}\033[0m，距离为{compute_distance(self.target, current_p[0].position)}")
            self.dump(msg)
            return

    def compute_init_velocity(self):
        # 保持设定速率不变
        speed = self.uav_velocity_value
        # 生成三维随机方向单位向量
        # 球坐标系随机方位角、俯仰角
        yaw = random.uniform(0, 2 * math.pi)  # 水平0~360度
        pitch = random.uniform(-math.pi / 2, math.pi / 2)  # 俯仰-90~90度

        # 球面坐标转笛卡尔单位方向向量
        dx = math.cos(pitch) * math.cos(yaw)
        dy = math.cos(pitch) * math.sin(yaw)
        dz = math.sin(pitch)

        # 合成随机方向、固定速率的速度
        init_vel = [
            dx * speed,
            dy * speed,
            dz * speed
        ]
        return init_vel

    def seed(self, seed_num=None):
        pass

    def get_observation_size_of_a_uav(self):
        """ Returns the shape of the observation
        注意：地形额外加
        （位置、速度、受攻击状态、是否是诱饵机）--按照次序--N * 8
        8: 自身-位置、速度、受攻击状态、是否是诱饵机
        3: 目标坐标
        3: 武器坐标
        3: 考虑到右向量和飞行有关，所以要加进去
        (N * 8 + 8 + 3 + 3)
        """
        if self.is_use_weapon:
            return self.n_total_uavs * 11 + 7
        else:
            return 3 + 3 + 3 + 3 + 3

    def close(self):
        pass
