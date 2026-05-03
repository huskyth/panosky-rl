from common.logger import *
from entries.phalanx.components.rader import *
from AStar.config import *


class TrackRader(Rader):
    def __init__(self, config):
        super().__init__(config)
        self.track_distance = self.config.track_distance
        self.fire_distance = self.config.fire_distance
        self.minimum_fire_distance = self.config.minimum_fire_distance
        self.minimum_track_distance = self.config.minimum_track_distance
        GameConfig.THREAT_LEVEL_THRESHOLD = velocity_constant / self.track_distance

        # 舷基座的位置，相对于绝对坐标
        self.position = config.common_config['position']
        temp = self.position.copy()
        temp[0] += 1
        self.top_project_position = temp
        self.top_angle_with_xoy = 0
        # self.top_angle_with_xoy 为角度，且为与水平面夹脚，上为正方向
        self.rotate_angular_velocity = self.config.rotate_angular_velocity
        self.rotate_angular_acceleration = self.config.rotate_angular_acceleration
        self.need_adjust_board_time = 0
        self.end_theta = 0
        self.end_position = 0

        self.current_target = None
        # 度/秒，单位度
        # 舷头此时的位置，相对于绝对坐标系
        self.capture_time = 0
        self.search_rader = None

    def _cal_time(self, theta):
        '''
        Todo: theta为度
        '''
        t1 = 2 * sqrt(theta / self.rotate_angular_acceleration)
        t2 = theta / self.rotate_angular_velocity + self.rotate_angular_velocity / self.rotate_angular_acceleration
        return min(t2, t1)

    def adjust_end_set_value(self):
        logger.info("调弦结束，赋值", is_in_file=False)
        self.top_project_position = self.end_position
        self.top_angle_with_xoy = self.end_theta
        self.need_adjust_board_time = 0

    def calculate_adjust_data(self):
        '''
        TODO:Test
        '''
        horizontal_board_projection_vector = subtraction_of_2_vector(self.top_project_position, self.position)
        uav_position = self.current_target.position
        uav_projection_point = [uav_position[0], uav_position[1], self.position[2]]
        uav_weapon_vector = subtraction_of_2_vector(uav_projection_point, self.position)

        self.end_position = add_of_2_vector(self.position, normalize(uav_weapon_vector))

        will_rotate_angle = cal_angle_of_2_vector(horizontal_board_projection_vector, uav_weapon_vector)
        horizontal_time = self._cal_time(will_rotate_angle)

        small = distance_of_2_point(self.position, uav_projection_point)
        large = distance_of_2_point(self.position, uav_position)
        uav_theta = radian_2_angle(acos_(small / large))
        self.end_theta = abs(uav_theta) if uav_position[2] >= self.position[2] else -abs(uav_theta)
        vertical_time = self._cal_time(abs(self.top_angle_with_xoy - self.end_theta))

        adjust_time = max(horizontal_time, vertical_time)
        logger.info("计算一次调弦时间：{}".format(adjust_time))
        self.need_adjust_board_time = adjust_time

    def build_dict(self):
        return self.config.build_dict()

    def set_search_rader(self, search_rader):
        self.search_rader = search_rader

    def get_current_target(self):
        '''
        当前目标必须不为空才调用这个方法
        :return:
        '''
        assert self.current_target is not None, "current target is None"
        return self.current_target

    def try_get_current_target(self):
        '''
        当前目标可以为空就调用这个方法
        :return:
        '''
        return self.current_target

    def check_can_fire(self):
        can_fire = (self.capture_time == CAPTURE_TIME)
        if can_fire:
            logger.info("可以进入开火状态，捕获时间为：" + str(self.capture_time))
            self.capture_time = 0

        return can_fire

    def step_capture(self):
        self.capture_time += 1
        logger.info("捕获时间自增，当前捕获时间是：" + str(self.capture_time))

    def reset_capture_time(self):
        self.capture_time = 0

    def current_target_exist(self):
        return self.current_target is not None

    def remove_target(self):
        '''
        雷达的目标移除
        :return:
        '''
        if self.current_target is None:
            log(LogType.ERROR, "移除目标时，当前目标不存在")
            assert False
            return
        self.current_target.reset_attacked_state()
        self.current_target = None

    def _handle_equal_condition(self, first_select_target_tuple, threat_level_uav_list):
        '''
        :param first_select_target_tuple: （ entry[0]: 威胁程度，entry[1]: 对应无人机）
        :param threat_level_uav_list:
        :return: 返回无人机实例
        '''
        threashold = 0.0
        close_list = [first_select_target_tuple[1]]
        for entry in threat_level_uav_list:
            if first_select_target_tuple[1] is entry[1]: continue
            if abs(first_select_target_tuple[0] - entry[0]) <= threashold:
                close_list.append(entry[1])
        if len(close_list) == 1: return close_list[0]
        return_uav = first_select_target_tuple[1]
        if GameConfig.threat_level_priority == ThreatLevelPriority.DISTANCE:
            return_uav = sorted(close_list, key=lambda item: distance_of_2_point(self.position, item.get_position()))[0]
        elif GameConfig.threat_level_priority == ThreatLevelPriority.VELOCITY:
            return_uav = sorted(close_list, key=lambda item: item.get_velocity_size())[-1]
        return return_uav

    def confirm_track_target(self, search_rader, uav_list, get_uav_index_fun):
        '''
        :param search_rader:
        :param uav_list:
        :return:
        '''
        # 计算威胁级别和判定范围
        threat_level_uav_list = self._adjust_board_prepare(search_rader, uav_list, get_uav_index_fun)
        if 0 == len(threat_level_uav_list):
            logger.info("没有搜索到或者没有在跟踪范围内")
            return None
        else:
            logger.info("搜索到或者在跟踪范围内")
        # （ entry[0]: 威胁程度，entry[1]: 对应无人机）
        max_threat_level = threat_level_uav_list[0][0]
        self.current_target = threat_level_uav_list[0][1]
        threat_level_string = ["威胁程度：" + str(entry[0]) + ", 对应无人机id：" + get_uav_index_fun(entry[1]) + "; " for
                               entry in
                               threat_level_uav_list]
        logger.info("威胁程度概述：" + str(threat_level_string), is_in_file=False)
        for entry in threat_level_uav_list:
            if max_threat_level < entry[0] and not entry[1].get_mountain_exist_bool(
                    self.search_rader.is_a_uav_in_search_range(entry[1])):
                max_threat_level = entry[0]
                self.current_target = entry[1]
        if max_threat_level > GameConfig.THREAT_LEVEL_THRESHOLD:
            logger.info("找到了最大威胁程度大于阈值 " + str(

                GameConfig.THREAT_LEVEL_THRESHOLD) + "的无人机：" + self.current_target.print_self())
            self.current_target = self._handle_equal_condition((max_threat_level, self.current_target),
                                                               threat_level_uav_list)
            assert self.current_target is not None, "_handle_equal_condition function exception"
        else:
            logger.info("没有找到，因为最大威胁程度小于阈值 " + str(GameConfig.THREAT_LEVEL_THRESHOLD))
            self.current_target = None
        return self.current_target

    def adjust_board(self, fun):
        '''
        调整角度
        :return: True代表调舷成功
        '''
        self._step_adjust_board_one_time()
        end = self.need_adjust_board_time <= 0
        if end:
            self.adjust_end_set_value()
            return True
        return False

    def _step_adjust_board_one_time(self):
        '''
        time--
        :return:
        '''

        self.need_adjust_board_time -= UNIT_TIME

    def _adjust_board_prepare(self, search_rader, uav_list, fun):
        '''
        :param search_rader:
        :return: 返回元组列表，元组内容为（威胁程度，对应无人机），返回为空代表没有搜索到或者在追踪范围之外
        '''
        # 检测是否有飞机进入追踪范围，然后确定并输出追踪目标
        uav_to_track_list = search_rader.get_mark_3_time_uav_list(uav_list)
        uav_to_track_list = [a_uav for a_uav in uav_to_track_list if self._is_a_uav_in_track_range(a_uav, fun)]
        threat_level_uav_list = [(self._cal_threat_level(a_uav), a_uav) for a_uav in uav_to_track_list]
        return threat_level_uav_list

    def _is_a_uav_in_track_range(self, a_uav, fun):
        log(LogType.INFO,
            "最小跟踪距离为：" + str(self.minimum_track_distance) + "，跟踪最远距离为：" + str(
                self.track_distance) + "，密集阵与id为" + fun(a_uav) + "的无人机的距离：" + str(
                distance_of_2_point(self.position,
                                    a_uav.position)), is_in_file=False)
        return is_a_point_in_a_sphere(self.track_distance, self.position, a_uav.position) and is_a_point_out_a_sphere(
            self.minimum_track_distance, self.position, a_uav.position)

    def is_target_in_track_range(self, fun):
        assert self.current_target is not None, "current target is None"
        log(LogType.INFO,
            "最小跟踪距离为：" + str(self.minimum_track_distance) + "，跟踪最远距离为：" + str(
                self.track_distance) + "，密集阵与id为" + fun(self.current_target) + "的无人机的距离：" + str(
                distance_of_2_point(self.position,
                                    self.current_target.position)), is_in_file=False)
        return is_a_point_in_a_sphere(self.track_distance, self.position,
                                      self.current_target.position) and is_a_point_out_a_sphere(
            self.minimum_track_distance, self.position, self.current_target.position)

    def is_target_can_use_because_no_mountain(self):
        assert self.current_target is not None, "current target is None"
        return not self.current_target.get_mountain_exist_bool(
            self.search_rader.is_a_uav_in_search_range(self.current_target))

    def is_target_in_fire_range(self):
        assert self.current_target is not None, "current target is None"
        log(LogType.INFO,
            "最小开火距离为：" + str(self.minimum_fire_distance) + "，开火最远距离为：" + str(
                self.fire_distance) + "，密集阵与id为" + self.current_target.get_id() + "无人机的距离：" + str(
                distance_of_2_point(self.position,
                                    self.current_target.position)))
        return is_a_point_in_a_sphere(self.fire_distance, self.position,
                                      self.current_target.position) and is_a_point_out_a_sphere(
            self.minimum_fire_distance, self.position, self.current_target.position)

    def _cal_threat_level(self, single_uav):
        from numpy import Inf
        '''
        :param single_uav:
        :return: 计算威胁程度
        '''
        uav_position = single_uav.position
        distance = distance_of_2_point(uav_position, self.position)
        distance_variation = self._cal_projection_velocity(single_uav)
        log(LogType.INFO,
            "uav_position = {}, self.position = {},distance_variation = {}".format(uav_position, self.position,
                                                                                   distance_variation))
        if distance == 0: return Inf
        return distance_variation / distance

    def _cal_projection_velocity(self, single_uav):
        '''
        :return: 计算投影速度
        '''
        uav_velocity, uav_velocity_direction, uav_position = single_uav.velocity, single_uav.velocity_direction, single_uav.position
        track_2_uav_vector = normalize(subtraction_of_2_vector(self.position, uav_position))
        log(LogType.INFO,
            "uav_velocity = {}, uav_velocity_direction = {},track_2_uav_vector = {}".format(uav_velocity,
                                                                                            uav_velocity_direction,
                                                                                            track_2_uav_vector))
        return abs(
            dot_of_2_vector(track_2_uav_vector, scalar_mul_vector(uav_velocity, normalize(uav_velocity_direction))))


if __name__ == '__main__':
    from factory.config_factory import ConfigEnum, ConfigFactory
    from entries.phalanx.phalanx import Phalanx
    from factory.uav_factory import UAVFactory

    tr = Phalanx(ConfigFactory.create(ConfigEnum.phalanx))
    tr.track_rader.current_target = UAVFactory.create()
    tr.track_rader.calculate_adjust_data()
