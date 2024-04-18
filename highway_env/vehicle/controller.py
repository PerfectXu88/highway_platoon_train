from typing import List, Tuple, Union, Optional

import numpy as np
import copy
from highway_env import utils
from highway_env.road.road import Road, LaneIndex, Route
from highway_env.utils import Vector
from highway_env.vehicle.kinematics import Vehicle
from highway_env.envs.common.newLQR import calculate_acceleration

class ControlledVehicle(Vehicle):
    """
#太粗糙，看看变量有没有别的计算方法
#加入碰撞体积

    A vehicle piloted by two low-level controller, allowing high-level actions such as cruise control and lane changes.

    - The longitudinal controller is a speed controller;
    - The lateral controller is a heading controller cascaded with a lateral position controller.
    """

    target_speed: float
    """ Desired velocity."""

    """Characteristic time"""
    TAU_ACC = 0.6  # [s]
    TAU_HEADING = 0.2  # [s]
    TAU_LATERAL = 0.6  # [s]

    TAU_PURSUIT = 0.5 * TAU_HEADING  # [s]
    KP_A = 1 / TAU_ACC
    KP_HEADING = 1 / TAU_HEADING
    KP_LATERAL = 1 / TAU_LATERAL  # [1/s]
    MAX_STEERING_ANGLE = np.pi / 3  # [rad]车辆的最大转向角度
    DELTA_SPEED = 0.5  # [m/s]表示车辆在一次控制周期内的速度变化限制

    LENGTH = 5.0
    """ Vehicle length [m] """
    WIDTH = 2.0
    """ Vehicle width [m] """

    def __init__(self,
                 road: Road,
                 position: Vector,#初始位置（坐标）
                 heading: float = 0,#车辆的初始航向角度（弧度）
                 speed: float = 0,#车辆的初始速度
                 target_lane_index: LaneIndex = None,#表示车辆的目标车道索引。如果未提供此参数，将使用车辆当前所在的车道索引作为目标车道索引
                 target_speed: float = None,#表示车辆的目标速度。如果未提供此参数，将使用车辆的当前速度作为目标速度
                 route: Route = None):#表示车辆的路径。路径通常是车辆要遵循的一系列道路段
        super().__init__(road, position, heading, speed)
        self.target_lane_index = target_lane_index or self.lane_index
        self.target_speed = target_speed or self.speed
        self.route = route
#该对象具有与现有车辆对象相同的动力学和目标状态，但其他属性使用默认值。这个方法的作用是克隆一个现有车辆对象，以便在模拟中创建与原车辆相似的新车辆。
    @classmethod
    def create_from(cls, vehicle: "ControlledVehicle") -> "ControlledVehicle":
        """
        Create a new vehicle from an existing one.

        The vehicle dynamics and target dynamics are copied, other properties are default.

        :param vehicle: a vehicle
        :return: a new vehicle at the same dynamical state
        """
        v = cls(vehicle.road, vehicle.position, heading=vehicle.heading, speed=vehicle.speed,
                target_lane_index=vehicle.target_lane_index, target_speed=vehicle.target_speed,
                route=vehicle.route)
        return v
#允许车辆规划到达指定目的地的路径
    def plan_route_to(self, destination: str) -> "ControlledVehicle":
        """
        Plan a route to a destination in the road network

        :param destination: a node in the road network
        """
        try:
            path = self.road.network.shortest_path(self.lane_index[1], destination)
        except KeyError:
            path = []
        if path:
            self.route = [self.lane_index] + [(path[i], path[i + 1], None) for i in range(len(path) - 1)]
        else:
            self.route = [self.lane_index]
        return self
#改变目标车道和速度，然后执行相应的纵向和横向控制，最终将这些控制输出应用到车辆上
    def act(self, action: Union[dict, str] = None) -> None:
        """
        Perform a high-level action to change the desired lane or speed.

        - If a high-level action is provided, update the target speed and lane;
        - then, perform longitudinal and lateral control.

        :param action: a high-level action
        """
        self.follow_road()
        if action == "FASTER":
            self.target_speed += self.DELTA_SPEED
        elif action == "SLOWER":
            self.target_speed -= self.DELTA_SPEED
        #如果 action 为 "LANE_RIGHT"，则将车辆的目标车道索引（self.target_lane_index）向右切换一个车道，如果目标车道可达的话
        elif action == "LANE_RIGHT":
            _from, _to, _id = self.target_lane_index
            target_lane_index = _from, _to, np.clip(_id + 1, 0, len(self.road.network.graph[_from][_to]) - 1)
            if self.road.network.get_lane(target_lane_index).is_reachable_from(self.position):
                self.target_lane_index = target_lane_index
        elif action == "LANE_LEFT":
            _from, _to, _id = self.target_lane_index
            target_lane_index = _from, _to, np.clip(_id - 1, 0, len(self.road.network.graph[_from][_to]) - 1)
            if self.road.network.get_lane(target_lane_index).is_reachable_from(self.position):
                self.target_lane_index = target_lane_index

        action = {"steering": self.steering_control(self.target_lane_index),#用于执行横向控制，计算车辆的方向盘转向角度
                  "acceleration": self.speed_control(self.target_speed)}#用于执行纵向控制，计算车辆的加速度
        action['steering'] = np.clip(action['steering'], -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE)#确保方向盘转向角度不超出最大允许值
        super().act(action)
#确保车辆在到达当前车道的末端时自动切换到下一个车道
    def follow_road(self) -> None:
        """At the end of a lane, automatically switch to a next one."""
        if self.road.network.get_lane(self.target_lane_index).after_end(self.position):
            self.target_lane_index = self.road.network.next_lane(self.target_lane_index,
                                                                 route=self.route,
                                                                 position=self.position,
                                                                 np_random=self.road.np_random)
#
    def steering_control(self, target_lane_index: LaneIndex) -> float:
        """
        Steer the vehicle to follow the center of an given lane.

        1. Lateral position is controlled by a proportional controller yielding a lateral speed command
        2. Lateral speed command is converted to a heading reference
        3. Heading is controlled by a proportional controller yielding a heading rate command
        4. Heading rate command is converted to a steering angle

        :param target_lane_index: index of the lane to follow
        :return: a steering wheel angle command [rad]
        """
        target_lane = self.road.network.get_lane(target_lane_index)
        lane_coords = target_lane.local_coordinates(self.position)
        lane_next_coords = lane_coords[0] + self.speed * self.TAU_PURSUIT
        lane_future_heading = target_lane.heading_at(lane_next_coords)

        # Lateral position control横向速度指令
        lateral_speed_command = - self.KP_LATERAL * lane_coords[1]
        # Lateral speed to heading航向角度指令
        heading_command = np.arcsin(np.clip(lateral_speed_command / utils.not_zero(self.speed), -1, 1))
        heading_ref = lane_future_heading + np.clip(heading_command, -np.pi/4, np.pi/4)
        # Heading control航向速度指令
        heading_rate_command = self.KP_HEADING * utils.wrap_to_pi(heading_ref - self.heading)
        # Heading rate to steering angle方向盘转向角度指令
        slip_angle = np.arcsin(np.clip(self.LENGTH / 2 / utils.not_zero(self.speed) * heading_rate_command, -1, 1))
        steering_angle = np.arctan(2 * np.tan(slip_angle))
        steering_angle = np.clip(steering_angle, -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE)
        return float(steering_angle)

    def speed_control(self, target_speed: float) -> float:
        """
        Control the speed of the vehicle.

        Using a simple proportional controller.

        :param target_speed: the desired speed
        :return: an acceleration command [m/s2]
        """
        return self.KP_A * (target_speed - self.speed)#加速度
#帮助车辆在接近路口时选择下一个路径
    def get_routes_at_intersection(self) -> List[Route]:
        """Get the list of routes that can be followed at the next intersection."""
        if not self.route:
            return []
        for index in range(min(len(self.route), 3)):
            try:
                next_destinations = self.road.network.graph[self.route[index][1]]
            except KeyError:
                continue
            if len(next_destinations) >= 2:
                break
        else:
            return [self.route]
        next_destinations_from = list(next_destinations.keys())
        routes = [self.route[0:index+1] + [(self.route[index][1], destination, self.route[index][2])]
                  for destination in next_destinations_from]
        return routes
#根据上面路径决定设置车辆在下一个路口要遵循的道路
    def set_route_at_intersection(self, _to: int) -> None:
        """
        Set the road to be followed at the next intersection.

        Erase current planned route.

        :param _to: index of the road to follow at next intersection, in the road network
        """

        routes = self.get_routes_at_intersection()
        if routes:
            if _to == "random":
                _to = self.road.np_random.randint(len(routes))
            self.route = routes[_to % len(routes)]
#根据车辆的规划路线和恒定速度，预测车辆在未来时间步内的位置和航向角度，以支持交通模拟中的轨迹预测
    def predict_trajectory_constant_speed(self, times: np.ndarray) -> Tuple[List[np.ndarray], List[float]]:
        """
        Predict the future positions of the vehicle along its planned route, under constant speed

        :param times: timesteps of prediction
        :return: positions, headings
        """
        coordinates = self.lane.local_coordinates(self.position)
        route = self.route or [self.lane_index]
        pos_heads = [self.road.network.position_heading_along_route(route,
                                                                    coordinates[0] + self.speed * t,
                                                                    0,
                                                                    self.lane_index)
                     for t in times]
        return tuple(zip(*pos_heads))#返回两个列表，一个包含未来位置的 NumPy 数组列表，另一个包含未来航向角度的浮点数列表


class MDPVehicle(ControlledVehicle):

    """A controlled vehicle with a specified discrete range of allowed target speeds."""
    DEFAULT_TARGET_SPEEDS = np.linspace(20.0, 30.0, 3)
    old_target_speeds = [20, 20]
    dt = 1/15.0

    def __init__(self,
                 road: Road,
                 position: List[float],
                 heading: float = 0,
                 speed: float = 0,
                 target_lane_index: Optional[LaneIndex] = None,
                 target_speed: Optional[float] = None,
                 target_speeds: Optional[Vector] = None,#表示车辆可以跟随的离散目标速度的列表，如果未提供此参数，将使用默认值 DEFAULT_TARGET_SPEEDS，该值是一个包含三个目标速度的 NumPy 数组
                 route: Optional[Route] = None) -> None:
        """
        Initializes an MDPVehicle

        :param road: the road on which the vehicle is driving
        :param position: its position
        :param heading: its heading angle
        :param speed: its speed
        :param target_lane_index: the index of the lane it is following
        :param target_speed: the speed it is tracking
        :param target_speeds: the discrete list of speeds the vehicle is able to track, through faster/slower actions
        :param route: the planned route of the vehicle, to handle intersections
        """
        super().__init__(road, position, heading, speed, target_lane_index, target_speed, route)
        self.target_speeds = np.array(target_speeds) if target_speeds is not None else self.DEFAULT_TARGET_SPEEDS
        self.speed_index = self.speed_to_index(self.target_speed)
        self.target_speed = self.index_to_speed(self.speed_index)
#加减速操作
    def act(self, action: Union[dict, str] = None) -> None:
        """
        Perform a high-level action.

        - If the action is a speed change, choose speed from the allowed discrete range.
        - Else, forward action to the ControlledVehicle handler.

        :param action: a high-level action
        """
        i = 0
        pos = self.position[0]
        while i < 3:     #获取现在受控车的编号
            if pos == self.road.vehicles[i].position[0]:
                break
            i += 1
        if action == 'MAINTAIN':
          a0, a1 = calculate_acceleration(self.road)
          if i == 2:
              speed = 20
          elif i == 1:
              speed = self.old_target_speeds[1] + a1 * self.dt
              self.old_target_speeds[1] = speed
              # print("车1：", speed)
          else:
              speed = self.old_target_speeds[0] + a0 * self.dt
              self.old_target_speeds[0] = speed
              # print("车2：", speed)
          self.target_speed = speed
          # print(a0, a1)
        else:
          if action == "FASTER":
            self.speed_index = self.speed_to_index(self.speed) + 1
          elif action == "SLOWER":
            self.speed_index = self.speed_to_index(self.speed) - 1
          else:
            super().act(action)
            return
          self.speed_index = int(np.clip(self.speed_index, 0, self.target_speeds.size - 1))
          self.target_speed = self.index_to_speed(self.speed_index)
          if i==1:
              self.old_target_speeds[1] = self.target_speed    #更新target_speed
          if i==0:
              self.old_target_speeds[0] = self.target_speed
          super().act()
#根据速度索引，从允许的目标速度范围中获取相应的速度值
    def index_to_speed(self, index: int) -> float:
        """
        Convert an index among allowed speeds to its corresponding speed

        :param index: the speed index []
        :return: the corresponding speed [m/s]
        """
        return self.target_speeds[index]
#用于找到与给定速度最接近的允许目标速度的索引
    def speed_to_index(self, speed: float) -> int:
        """
        Find the index of the closest speed allowed to a given speed.

        Assumes a uniform list of target speeds to avoid searching for the closest target speed

        :param speed: an input speed [m/s]
        :return: the index of the closest speed allowed []
        """
        x = (speed - self.target_speeds[0]) / (self.target_speeds[-1] - self.target_speeds[0])
        return np.int64(np.clip(np.round(x * (self.target_speeds.size - 1)), 0, self.target_speeds.size - 1))
#也用于找到与给定速度最接近的允许目标速度的索引。不同之处在于，它使用了 DEFAULT_TARGET_SPEEDS 默认目标速度范围
    @classmethod
    def speed_to_index_default(cls, speed: float) -> int:
        """
        Find the index of the closest speed allowed to a given speed.

        Assumes a uniform list of target speeds to avoid searching for the closest target speed

        :param speed: an input speed [m/s]
        :return: the index of the closest speed allowed []
        """
        x = (speed - cls.DEFAULT_TARGET_SPEEDS[0]) / (cls.DEFAULT_TARGET_SPEEDS[-1] - cls.DEFAULT_TARGET_SPEEDS[0])
        return np.int64(np.clip(
            np.round(x * (cls.DEFAULT_TARGET_SPEEDS.size - 1)), 0, cls.DEFAULT_TARGET_SPEEDS.size - 1))

    #获取车辆的速度索引，要么从车辆对象的属性中获取，要么通过计算车辆的当前速度来确定速度索引
    @classmethod
    def get_speed_index(cls, vehicle: Vehicle) -> int:
        return getattr(vehicle, "speed_index", cls.speed_to_index_default(vehicle.speed))
#预测车辆在一系列未来高级别操作下的未来轨迹，包括车辆的位置、速度和其他状态信息
    def predict_trajectory(self, actions: List, action_duration: float, trajectory_timestep: float, dt: float) \
            -> List[ControlledVehicle]:
        """
        Predict the future trajectory of the vehicle given a sequence of actions.

        :param actions: a sequence of future actions.
        :param action_duration: the duration of each action.
        :param trajectory_timestep: the duration between each save of the vehicle state.
        :param dt: the timestep of the simulation
        :return: the sequence of future states
        """
        states = []
        v = copy.deepcopy(self)
        t = 0
        for action in actions:
            v.act(action)  # High-level decision
            for _ in range(int(action_duration / dt)):
                t += 1
                v.act()  # Low-level control action
                v.step(dt)
                if (t % int(trajectory_timestep / dt)) == 0:
                    states.append(copy.deepcopy(v))
        return states
