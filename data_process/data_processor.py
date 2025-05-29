import json
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from data_process.ego_process import calculate_additional_ego_states,sampled_past_ego_states_to_array
from data_process.agent_process import sampled_tracked_objects_to_array_list, sampled_static_objects_to_array_list, agent_past_process,agent_future_process
from data_process.map_process import get_neighbor_vector_set_map, map_process
from data_process.roadblock_utils import route_roadblock_correction
from data_process.reference_process import get_single_reference_line,egoSE2_to_Frenet
from data_process.option_reward_label import get_option,get_action_reward,get_option_reward
from data_process.utils import draw_rectangle,get_nearest_line,absolute_to_relative_poses,se2_transform
from data_process.data_normalize import normalize_data

from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.actor_state.state_representation import Point2D
from nuplan.planning.utils.multithreading.worker_parallel import SingleMachineParallelExecutor
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
from nuplan.planning.training.preprocessing.features.trajectory_utils import convert_absolute_to_relative_poses
from nuplan.planning.training.preprocessing.utils.agents_preprocessing import AgentInternalIndex



class DataProcessor(object):
    def __init__(self, config):

        self.past_time_horizon = 4 # [seconds]
        self.num_past_poses = 8 * self.past_time_horizon 
        self.future_time_horizon = 8 # [seconds]
        self.num_future_poses = 16 * self.future_time_horizon

        self.num_agents = config.agent_num
        self.num_static = config.static_objects_num
        self.max_ped_bike = 10 # Limit the number of pedestrians and bicycles in the agent.
        self._radius = 200 # [m] query radius scope relative to the current pose.

        self._map_features = ['LANE', 'LEFT_BOUNDARY', 'RIGHT_BOUNDARY', 'ROUTE_LANES'] # name of map features to be extracted.
        self._max_elements = {'LANE': config.lane_num, 'LEFT_BOUNDARY': config.lane_num, 'RIGHT_BOUNDARY': config.lane_num, 'ROUTE_LANES': config.route_num} # maximum number of elements to extract per feature layer.
        self._max_points = {'LANE': config.lane_len, 'LEFT_BOUNDARY': config.lane_len, 'RIGHT_BOUNDARY': config.lane_len, 'ROUTE_LANES': config.route_len} # maximum number of points per feature to extract per feature layer.
        self.standard = config.normalize

    # Use for inference
    def process_data(self, history_buffer, traffic_light_data, map_api, route_roadblock_ids, mission_goal, _last_global_trajectory, device='cpu'):
        _map_features = ['LANE', 'LEFT_BOUNDARY', 'RIGHT_BOUNDARY', 'ROUTE_LANES']

        ego_past_state = history_buffer.ego_states[-2:]
        ego_state = history_buffer.ego_states[-1]

        ego_coords = Point2D(ego_state.rear_axle.x, ego_state.rear_axle.y)
        anchor_ego_state = np.array([ego_state.rear_axle.x, ego_state.rear_axle.y, ego_state.rear_axle.heading], dtype=np.float64)

        ego_past_state = sampled_past_ego_states_to_array(ego_past_state)
        ego_past_state = absolute_to_relative_poses(ego_past_state, anchor_ego_state)
        sample_interval = history_buffer.sample_interval if history_buffer.sample_interval is not None else 0.1
        sample_step = int(0.5 / sample_interval)
        ego_current_state = calculate_additional_ego_states(ego_past_state,[0, sample_interval*1e6])

        _last_relative_trajectory = se2_transform(_last_global_trajectory, anchor_ego_state)[0] if _last_global_trajectory is not None else None

        route_roadblock_ids = route_roadblock_correction(
            ego_state, map_api, route_roadblock_ids
        )
        coords, traffic_light_data, speed_limit, lane_route = get_neighbor_vector_set_map(
            map_api, _map_features, ego_coords, self._radius, traffic_light_data
        )
        vector_map = map_process(route_roadblock_ids, anchor_ego_state, coords, traffic_light_data, speed_limit, lane_route, _map_features, self._max_elements, self._max_points)

        lanes = vector_map['lanes']
        lanes_xy = lanes[:, :, :2].tolist()
        route_lanes = vector_map['route_lanes']
        route_xy = route_lanes[:,:, :2].tolist()
        if mission_goal is not None:
            mission_goal = np.array([[mission_goal.x, mission_goal.y, mission_goal.heading]], dtype=np.float64)
            mission_point = se2_transform(mission_goal, anchor_ego_state)[0][0]
        else:
             mission_point = None   

        reference_line = get_single_reference_line(
            route_xy, 120, lanes_xy, mission_point,_last_relative_trajectory,
        )

        observation_buffer = history_buffer.observation_buffer
        neighbor_agents_past, neighbor_agents_types = sampled_tracked_objects_to_array_list(observation_buffer)
        static_objects, static_objects_types = sampled_static_objects_to_array_list(observation_buffer[-1])
        _, neighbor_agents_past, static_objects = \
            agent_past_process(None, neighbor_agents_past, neighbor_agents_types, self.num_agents, static_objects, static_objects_types, self.num_static, anchor_ego_state, reference_line)
        indices = np.arange(len(neighbor_agents_past) - 1, -1, -sample_step)[ :8]
        neighbor_agents_past = neighbor_agents_past[indices[::-1]]

        start_point = ego_current_state[:2]
        F_self = egoSE2_to_Frenet(ego_current_state, reference_line)
        ego_current_state[1] = F_self[1]
        ego_current_state[2] = np.cos(F_self[2])
        ego_current_state[3] = np.sin(F_self[2])
        data = {
            "ego_current_state": ego_current_state,
            "neighbor_agents_past": neighbor_agents_past,
            "static_objects": static_objects,
            'reference_line': reference_line,
        }
        data.update(vector_map)
        data = normalize_data(data, self.standard)
        del data['lanes_speed_limit'], data['route_lanes_speed_limit'], data['lanes_has_speed_limit'], data['route_lanes_has_speed_limit']
        
        for key in data.keys():
            data[key] = torch.tensor(data[key], dtype=torch.float32).unsqueeze(0).to(device) # add batch dimension
        data['reference_line'] = data['reference_line'].view(-1).unsqueeze(0)
        return data, reference_line, start_point