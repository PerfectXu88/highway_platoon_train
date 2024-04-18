from typing import Dict, Text, Tuple

import numpy as np

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.road.road import Road, RoadNetwork
from highway_env.utils import near_split
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle

from itertools import repeat, product
from gym.envs.registration import register
from highway_env.road.lane import LineType, StraightLane, CircularLane, SineLane
from highway_env.vehicle.behavior import IDMVehicle


# Observation = np.ndarray


class my_own_highway(AbstractEnv):
    """
    A highway driving environment.

    The vehicle is driving on a straight highway with several lanes, and is rewarded for reaching a high speed,
    staying on the rightmost lanes and avoiding collisions.
    """

    @classmethod
    def default_config(cls) -> dict:                #此处有修改
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics",
                "vehicles_count": 5,
                "features": ["presence", "x", "y", "vx", "vy", "acceleration"],
                "features_range":{
                    "x": [-100, 100],
                    "y": [-100, 100],
                    "vx": [-20, 20],
                    "vy": [-20, 20],
                    "acceleration": [-10, 10]
                },
                "absolute": False,
                "order": "sorted",
                "see_behind": "true"
                # "observation_config": {
                #  "type": "Kinematics",
                # }
            },
            "action": {
                "type": "MultiDiscreteMetaAction",
                # "type": "MultiAgentAction",
                "action_config": {
                    "type": "DiscreteMetaAction",
                },
                "longitudinal": True,
                "lateral": True
            },
            # "simulation_frequency": 20,
            # "policy_frequency": 20,
            "lanes_count": 3,
            "duration": 600, # 单位为s
            # "vehicles_count": 50,
            "controlled_vehicles": 3,
            "other_vehicles": 50,
            "Normal_vehicles_type": 'highway_env.vehicle.behavior.IDMVehicle', "Normal_vehicles":10,
            "Aggressive_vehicles_type": 'highway_env.vehicle.behavior.AggressiveVehicle', "Aggressive_vehicles": 10,
            "DefensiveVehicle_vehicles_type": 'highway_env.vehicle.behavior.DefensiveVehicle', "Defensive_vehicles": 10,
            "initial_lane_id": 0,
            "ego_spacing": 2,
            "vehicles_density": 1,
            "collision_reward": -200,    # The reward received when colliding with a vehicle.
            "right_lane_reward": 0.1,  # The reward received when driving on the right-most lanes, linearly mapped to
                                       # zero for other lanes.
            "high_speed_reward": 0.4,  # The reward received when driving at full speed, linearly mapped to zero for
                                       # lower speeds according to config["reward_speed_range"].
            "lane_change_reward": 0,   # The reward received at each lane change action.
            # "reward_speed_range": [20, 30],
            "reward_speed_range": [30, 40],
            "normalize_reward": True,
            "offroad_terminal": False,
            "screen_width": 1200,
            "screen_hight": 300
        })
        return config

    def _reset(self) -> None:
        self._create_road()
        self._create_vehicles()

    def _create_road(self) -> None:
        """Create a road composed of straight adjacent lanes."""
        # road = Road(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])
        road = Road(network=RoadNetwork.straight_road_network(self.config["lanes_count"], speed_limit=40),
                         np_random=self.np_random, record_history=self.config["show_trajectories"])
        self.road = road

    # def _create_road(self) -> None:
    #     net = RoadNetwork
    #     speedlimits = [120,60,60,None]
    #     lane = StraightLane([0,0],[100,0], line_types=(LineType.CONTINUOUS, LineType.CONTINUOUS),width = 5,speed_limit=speedlimits[0])
    #     self.lane = lane
    #
    #     net.add_lane("a","b", lane)
    #

    # def _create_vehicles(self) -> None:
    #     """Create some new random vehicles of a given type, and add them on the road."""
    #     other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
    #     other_per_controlled = near_split(self.config["other_vehicles"], num_bins=self.config["controlled_vehicles"])
    #
    #     self.controlled_vehicles = []
    #     for others in other_per_controlled:
    #         vehicle = Vehicle.create_random(
    #             self.road,
    #             speed=25,
    #             lane_id=self.config["initial_lane_id"],
    #             spacing=self.config["ego_spacing"]
    #         )
    #         vehicle = self.action_type.vehicle_class(self.road, vehicle.position, vehicle.heading, vehicle.speed)
    #         self.controlled_vehicles.append(vehicle)
    #         self.road.vehicles.append(vehicle)
    #
    #         for _ in range(others):
    #             vehicle = other_vehicles_type.create_random(self.road, spacing=1 / self.config["vehicles_density"])
    #             vehicle.randomize_behavior()
    #             self.road.vehicles.append(vehicle)

    def _create_vehicles(self) -> None:
        rng = self.np_random
        self.controlled_vehicles = []
        for i in range(self.config["controlled_vehicles"]):
            # vehicle = self.action_type.vehicle_class(self.road,vehicle)
            vehicle = Vehicle.create_CACC(
                cls = Vehicle,
                road = self.road,
                speed = 25,
                lane_id = self.config["initial_lane_id"],
                spacing = self.config["ego_spacing"]
            )
            vehicle_1 = self.action_type.vehicle_class(self.road,
                                                       vehicle.position + 2*np.float64(Vehicle.LENGTH),
                                                       vehicle.heading,
                                                       vehicle.speed)
            self.controlled_vehicles.append(vehicle_1)
            self.road.vehicles.append(vehicle_1)

        front_vehicle = IDMVehicle.make_on_lane(self.road,
                                                lane_index = self.config["initial_lane_id"],
                                                longitudinal=vehicle.position + 20*np.float64(Vehicle.LENGTH),
                                                speed=rng.uniform(low=10,high=25))
        self.road.vehicles.append(front_vehicle)

        left_vehicle = IDMVehicle.make_on_lane(self.road,
                                               lane_index=self.config["initial_lane_id"]+1,
                                               longitudinal=vehicle.position,
                                               id = 2,
                                               # lateral = -1.7+10,
                                               speed=rng.uniform(low=20,high=40))
        self.road.vehicles.append(left_vehicle)

        # right_vehicle = IDMVehicle.make_on_lane(self.road,
        #                                         lane_index=self.config["initial_lane_id"],
        #                                         longitudinal=vehicle.position,
        #                                         # lateral = 0,
        #                                         speed=rng.uniform(low=20,high=40))
        # self.road.vehicles.append(right_vehicle)

        # other_vehicles = [[self.config["Normal_vehicles_type"],self.config["Normal_vehicles"]],
        #                   [self.config["Aggressive_vehicles_type"],self.config["Aggressive_vehicles"]],
        #                   [self.config["DefensiveVehicle_vehicles_type"],self.config["Defensive_vehicles"]]
        #                   ]
        other_vehicles = [[self.config["Normal_vehicles_type"],1]]
        for type in other_vehicles:
            other_vehicles_type = utils.class_from_path(type[0])
            for _ in range(type[1]):
                vehicle = other_vehicles_type.create_random(self.road, spacing=1 / self.config["vehicles_density"])
                vehicle.randomize_behavior()
                self.road.vehicles.append(vehicle)



    def _reward(self, action: Action) -> float:
        """
        The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
        :param action: the last action performed
        :return: the corresponding reward
        """
        rewards = self._rewards(action)
        reward = sum(self.config.get(name, 0) * reward for name, reward in rewards.items())
        if self.config["normalize_reward"]:
            reward = utils.lmap(reward,
                                [self.config["collision_reward"],
                                 self.config["high_speed_reward"] + self.config["right_lane_reward"]],
                                [0, 1])
        reward *= rewards['on_road_reward']
        return reward

    def _rewards(self, action: Action) -> Dict[Text, float]:
        neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
        lane = self.vehicle.target_lane_index[2] if isinstance(self.vehicle, ControlledVehicle) \
            else self.vehicle.lane_index[2]
        # Use forward speed rather than speed, see https://github.com/eleurent/highway-env/issues/268
        forward_speed = self.vehicle.speed * np.cos(self.vehicle.heading)
        scaled_speed = utils.lmap(forward_speed, self.config["reward_speed_range"], [0, 1])
        return {
            "collision_reward": float(self.vehicle.crashed),
            "right_lane_reward": lane / max(len(neighbours) - 1, 1),
            "high_speed_reward": np.clip(scaled_speed, 0, 1),
            "on_road_reward": float(self.vehicle.on_road)
        }

    # def _is_terminated(self) -> bool:
    #     """The episode is over if the ego vehicle crashed."""
    #     return (self.vehicle.crashed or
    #             self.config["offroad_terminal"] and not self.vehicle.on_road)

   #是否终止
    def _is_terminated(self) -> bool:

        ori_mark = False
        for ij in range(len(self.controlled_vehicles)):
            ori_mark = ori_mark or self.controlled_vehicles[ij].crashed

        return (ori_mark or
            self.steps >= self.config["duration"]
            or not self.vehicle.on_road)


    def _is_truncated(self) -> bool:
        """The episode is truncated if the time limit is reached."""
        return self.time >= self.config["duration"]

