from typing import Dict, Text, Tuple
import numpy as np
from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.road.road import Road, RoadNetwork
from highway_env.utils import near_split
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle
import math
import random

from itertools import repeat, product
from gym.envs.registration import register
from highway_env.road.lane import LineType, StraightLane, CircularLane, SineLane
from highway_env.vehicle.behavior import IDMVehicle,acciVehicle


# Observation = np.ndarray


class jam_normal_up(AbstractEnv):
    """
    A highway driving environment.

    The vehicle is driving on a straight highway with several lanes, and is rewarded for reaching a high speed,
    staying on the rightmost lanes and avoiding collisions.
    """

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            # "observation": {
            #     "type": "GrayscaleObservation",
            #     "observation_shape": (11, 11),
            #     "stack_size": 7,
            #     "weights": [0.2989, 0.5870, 0.1140],  # weights for RGB conversion
            #     "scaling": 1.75,
            # },
            "observation": {
                "type": "Kinematics",
                "vehicles_count": 15,
                "features": ["presence", "x", "y", "vx", "vy"],
                "features_range": {
                    "x": [-100, 100],
                    "y": [-100, 100],
                    "vx": [-20, 20],
                    "vy": [-20, 20]
                },
                "absolute": False,
                "order": "sorted",
                "see_behind":"true"
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
            "policy_frequency": 1,
            "lanes_count": 3,
            "duration": 1800,  # 单位为s
            # "vehicles_count": 50,
            "controlled_vehicles": 3,
            "still_bleibend": 4,
            "Normal_vehicles_type": 'highway_env.vehicle.behavior.IDMVehicle', "Normal_vehicles": 10,
            "Aggressive_vehicles_type": 'highway_env.vehicle.behavior.AggressiveVehicle', "Aggressive_vehicles": 10,
            "DefensiveVehicle_vehicles_type": 'highway_env.vehicle.behavior.DefensiveVehicle', "Defensive_vehicles": 10,
            "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle","Normal_vehicles": 10,
            "initial_lane_id": 0,
            "ego_spacing": 2,


            "other_vehicles": 30,
            "vehicles_density": 1,


            "forward_speed": -1,
            "acc_reward":1,

            "on_road_reward": 0.05,
            "distance_reward": 10,
            "sameline_reward": 10,
            'far_reward': 10,
            "equal_speed":2,
            "Aktionen_Messung": 0,
            "Gering_Geschwind": -0.0,
            "collision_reward": -200,  # The reward received when colliding with a vehicle.
            "right_lane_reward": 0,  # The reward received when driving on the right-most lanes, linearly mapped to
            # zero for other lanes.
            "high_speed_reward": 50,  # The reward received when driving at full speed, linearly mapped to zero for
            # lower speeds according to config["reward_speed_range"].
            "line_change_reward": 0,  # The reward received at each lane change action.

            "reward_speed_range": [28, 33],

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
        rng = self.np_random
        """Create a road composed of straight adjacent lanes."""
        # road = Road(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])
        road = Road(network=RoadNetwork.straight_road_network(self.config["lanes_count"], speed_limit=30),
                    np_random=self.np_random, record_history=self.config["show_trajectories"])
        self.road = road

    def _create_vehicles(self) -> None:
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        rng = self.np_random
        self.controlled_vehicles = []
        #生成控制车
        for i in range(self.config["controlled_vehicles"]):
            # vehicle = self.action_type.vehicle_class(self.road,vehicle)
            vehicle = Vehicle.create_CACC(
                cls=Vehicle,
                road=self.road,
                speed=28,
                lane_id=self.config["initial_lane_id"],
                spacing=self.config["ego_spacing"],
                y = -6
            )

            vehicle_1 = self.action_type.vehicle_class(self.road,
                                                       vehicle.position + 2 * np.float64(Vehicle.LENGTH),
                                                       vehicle.heading,
                                                       vehicle.speed,
                                                       target_speed=33
                                                       )
            self.controlled_vehicles.append(vehicle_1)
            self.road.vehicles.append(vehicle_1)




        # others=20
        # for _ in range(others):
        #     vehicle = other_vehicles_type.create_random(self.road, spacing=1 / self.config["vehicles_density"])
        #     vehicle.randomize_behavior()
        #     self.road.vehicles.append(vehicle)
        #


        #生成他车
        jam=[
            # [174.94942691   4.         ],# ————————————————————————
            # [184.94942691   ,4.           ],
            # [194.94942691   ,4.     ],# ————————————————————————
            [205.80962922   ,0.        ],
            [216.22140462   ,8.         ],
            [228.5351024   ,8.       ],
            [239.62648826   ,0. ], # ————————————————————————
            [251.18755074   ,4.  ],
            [262.11204656   ,4.        ],
            [273.13737287   ,4. ],
            [286.02768007   ,0.         ],
            [299.22544339   ,0.         ],
            [311.98872132   ,8.         ],
            [323.51540789   ,0.        ],
            [334.74029529   ,4.         ],
            [345.54986178   ,4.        ],
            [356.83199336   ,8.        ], # ————————————————————————
            [367.61570171   ,4.         ],
            [378.54046433   ,0.        ],
            [389.28487285   ,4.        ],
            [401.97185417   ,4.        ], # ——————————————————————————
            [414.78452908   ,0.        ],
            [426.45654771   ,8.        ],

        ]

        for i in range(len(jam)):
            jam_vehicle = IDMVehicle.make_on_lane(self.road,
                                                    lane_index=self.config["initial_lane_id"],
                                                    longitudinal=jam[i][0],
                                                    lateral=jam[i][1],
                                                    speed=rng.uniform(low=25, high=26))
            self.road.vehicles.append(jam_vehicle)

        help=[
            [165.17794036, 0],
            [171.17794036, 8],
            [160.17794036, 4],
            [189.17794036, 8],
        ]

        for i in range(len(help)):
            help_vehicle = IDMVehicle.make_on_lane(self.road,
                                                   lane_index=self.config["initial_lane_id"],
                                                   longitudinal=help[i][0],
                                                   lateral=help[i][1],
                                                   speed=rng.uniform(low=25, high=28))
            self.road.vehicles.append(help_vehicle)



    def _sameline(self, sameline_reward):
        sameline_reward = len(set(sameline_reward))
        if sameline_reward == 1:
            return 1
        elif sameline_reward == 2:
            return -0.5
        else:
            return -1

    def _distance(self,distance):
        distance = sorted(distance)
        reward = 0
        for i in range(len(distance)-1):
            diff = distance[i+1] - distance[i]
            if diff >= 30 :
                reward += -10
            elif diff >= 20 :
                reward += -1
            if diff < 20 and diff >= 10:
                reward += 0
            if diff < 10:
                reward += 1 / (abs(diff-9)+1)
        return reward



    def _far(self,):
        far=[]
        for i in range(len(self.controlled_vehicles)):
            far.append(self.controlled_vehicles[i].destination[0])
        re_far = min(far) - 162
        if re_far < 100:
            return 0
        if re_far < 320:
            return re_far / 320
        else:
            return 1
        # return 1-np.exp(-re_far)


    def _Aktion_Messung(self,aktionen,i):
        j = 0
        punkte = 0
        while j < i:
            if aktionen[j]!= 1:
                punkte-=0.1
            j+=1
            continue
        return punkte

    def _for_speed(self,forward_speed):
        j=0
        k=0
        for i in range(len(forward_speed)):
            if forward_speed[i]<20:
                k=k+1
        return k

    def _acc_re(self,car_acclration):
        j=0
        for i in range(len(car_acclration)):
            j = j + (car_acclration[i])**2
        k=utils.lmap(j, [0,15], [0, 1])
        return k


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
                                 self.config["high_speed_reward"] +
                                 self.config["on_road_reward"] +
                                 self.config["sameline_reward"] +
                                 self.config["distance_reward"] +
                                 self.config["far_reward"] +
                                 self.config["equal_speed"] +
                                 self.config["Aktionen_Messung"]+
                                 self.config["forward_speed"]+
                                 self.config["acc_reward"]
                                 ],
                                [0, 1])
        reward *= rewards['on_road_reward']
        return reward

    def _rewards(self, action: Action) -> Dict[Text, float]:
        # neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
        # lane = self.vehicle.target_lane_index[2] if isinstance(self.vehicle, ControlledVehicle) \
        #     else self.vehicle.lane_index[2]
        # Use forward speed rather than speed, see https://github.com/eleurent/highway-env/issues/268
        # forward_speed = self.vehicle.speed * np.cos(self.vehicle.heading)
        # scaled_speed = utils.lmap(forward_speed, self.config["reward_speed_range"], [0, 1])
        forward_speed = []
        high_speed_reward = []
        on_road_reward = []
        collision_reward = []
        distance = []
        sameline_reward = []
        equal_speed = []
        aktionen=[]
        car_acclration=[]
        akt = action
        i = 0
        for this_vehicle in self.controlled_vehicles:
            forward_speed.append(this_vehicle.speed * np.cos(this_vehicle.speed))
            scaled_speed = utils.lmap(np.mean(forward_speed), self.config["reward_speed_range"], [0, 1])
            high_speed_reward.append(np.clip(scaled_speed, 0, 1))
            on_road_reward.append(this_vehicle.on_road)
            collision_reward.append(this_vehicle.crashed)
            distance.append(this_vehicle.position[0])
            sameline_reward.append(this_vehicle.lane_index[2])
            equal_speed.append(this_vehicle.speed)
            car_acclration.append(this_vehicle.action.get('acceleration'))
            aktionen.append(akt % 5)
            akt = akt / 5
            i += 1
        #map1 = self._mindis()

        return {
            # "collision_reward": float(self.vehicle.crashed),
            # "right_lane_reward": lane / max(len(neighbours) - 1, 1),
            # "high_speed_reward": np.clip(scaled_speed, 0, 1),
            # "on_road_reward": float(self.vehicle.on_road),
            "on_road_reward": float(sum(on_road_reward) / len(on_road_reward)),
            "high_speed_reward": float(sum(high_speed_reward) / len(high_speed_reward)),
            "collision_reward": float(max(collision_reward)),
            "distance_reward": float(self._distance(distance)),
            "sameline_reward": float(self._sameline(sameline_reward)),
            "far_reward":float(self._far()),
            "equal_speed" : float(1/(np.var(equal_speed)+1)),
            "Aktionen_Messung": float(self._Aktion_Messung(aktionen, i)),
            "forward_speed":float(self._for_speed(forward_speed)),
            # "acc_reward":float(self._acc_re(car_acclration))
        }

    # def _is_terminated(self) -> bool:
    #     """The episode is over if the ego vehicle crashed."""
    #     return (self.vehicle.crashed or
    #             self.config["offroad_terminal"] and not self.vehicle.on_road)


    def _is_terminated(self) -> bool:
        return (self.controlled_vehicles[0].crashed or
                self.controlled_vehicles[1].crashed or
                self.controlled_vehicles[2].crashed or
                self.steps >= self.config["duration"]
                or not self.vehicle.on_road)

    def _is_truncated(self) -> bool:
        """The episode is truncated if the time limit is reached."""
        return self.time >= self.config["duration"]
