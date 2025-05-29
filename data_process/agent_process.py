"""
Module: Agent Data Preprocessing Functions
Description: This module contains functions for agents related data processing.

Categories:
    1. Get list of agent array from raw data
    2. Get agents array for model input
"""
import numpy as np
from typing import Dict

from nuplan.planning.training.preprocessing.utils.agents_preprocessing import AgentInternalIndex
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks

from data_process.utils import absolute_to_relative_poses
from data_process.reference_process import se2_ro_Frenet, se2_to_Frenet_bacth

# =====================
# 1. Get list of agent array from raw data
# =====================
def _extract_agent_array(tracked_objects, track_token_ids, object_types):
    """
    Extracts the relevant data from the agents present in a past detection into a array.
    Only objects of specified type will be transformed. Others will be ignored.
    The output is a array as described in AgentInternalIndex
    :param tracked_objects: The tracked objects to turn into a array.
    :track_token_ids: A dictionary used to assign track tokens to integer IDs.
    :object_type: TrackedObjectType to filter agents by.
    :return: The generated array and the updated track_token_ids dict.
    """
    agents = tracked_objects.get_tracked_objects_of_types(object_types)
    agent_types = []
    output = np.zeros((len(agents), AgentInternalIndex.dim()), dtype=np.float64)
    max_agent_id = len(track_token_ids)

    for idx, agent in enumerate(agents):
        if agent.track_token not in track_token_ids:
            track_token_ids[agent.track_token] = max_agent_id
            max_agent_id += 1
        track_token_int = track_token_ids[agent.track_token]

        output[idx, AgentInternalIndex.track_token()] = float(track_token_int)
        output[idx, AgentInternalIndex.vx()] = agent.velocity.x
        output[idx, AgentInternalIndex.vy()] = agent.velocity.y
        output[idx, AgentInternalIndex.heading()] = agent.center.heading
        output[idx, AgentInternalIndex.width()] = agent.box.width
        output[idx, AgentInternalIndex.length()] = agent.box.length
        output[idx, AgentInternalIndex.x()] = agent.center.x
        output[idx, AgentInternalIndex.y()] = agent.center.y
        agent_types.append(agent.tracked_object_type)

    return output, track_token_ids, agent_types


def sampled_tracked_objects_to_array_list(past_tracked_objects):
    """
    Arrayifies the agents features from the provided past detections.
    For N past detections, output is a list of length N, with each array as described in `_extract_agent_array()`.
    :param past_tracked_objects: The tracked objects to arrayify.
    :return: The arrayified objects.
    """
    object_types = [TrackedObjectType.VEHICLE, TrackedObjectType.PEDESTRIAN, TrackedObjectType.BICYCLE]
    output = []
    output_types = []
    track_token_ids = {}

    for i in range(len(past_tracked_objects)):
        if type(past_tracked_objects[i]) == DetectionsTracks:
            track_object = past_tracked_objects[i].tracked_objects
        else:
            track_object = past_tracked_objects[i]
        arrayified, track_token_ids, agent_types = _extract_agent_array(track_object, track_token_ids, object_types)
        output.append(arrayified)
        output_types.append(agent_types)

    return output, output_types

def sampled_static_objects_to_array_list(present_tracked_objects):

    static_object_types = [TrackedObjectType.CZONE_SIGN,
                    TrackedObjectType.BARRIER,
                    TrackedObjectType.TRAFFIC_CONE,
                    TrackedObjectType.GENERIC_OBJECT
                    ]

    if type(present_tracked_objects) == DetectionsTracks:
        present_tracked_objects = present_tracked_objects.tracked_objects

    static_obj = present_tracked_objects.get_tracked_objects_of_types(static_object_types)
    agent_types = []
    output = np.zeros((len(static_obj), 5), dtype=np.float64)

    for idx, agent in enumerate(static_obj):
        output[idx, 0] = agent.center.x
        output[idx, 1] = agent.center.y
        output[idx, 2] = agent.center.heading
        output[idx, 3] = agent.box.width
        output[idx, 4] = agent.box.length
        agent_types.append(agent.tracked_object_type)

    return output, agent_types


# =====================
# 2. Get agents array for model input
# =====================

def agent_past_process(past_ego_states, past_tracked_objects, tracked_objects_types, num_agents, static_objects, static_objects_types, num_static, anchor_ego_state,reference_line):
    """
    This function process the data from the raw agent data.
    :param past_ego_states: The input array data of the ego past.
    :param past_tracked_objects: The input array data of agents in the past.
    :param tracked_objects_types: The type of agents in the past.
    :param num_agents: Clip the number of agents.
    :param static_objects: The input array data of static objects in the past.
    :param static_objects_types: The type of static objects in the past.
    :param num_static: Clip the number of static objects.
    :param anchor_ego_state: Ego current state
    :return: ego, agents, selected_indices, static_objects
    """
    agents_states_dim = 8 # x, y, cosh, sinh, vx, vy, length, width
    ego_history = past_ego_states
    agents = past_tracked_objects

    if past_ego_states is not None:
        ego = absolute_to_relative_poses(ego_history, anchor_ego_state)
        start_point = ego[-1][:2]
    else:
        ego = None
        start_point = np.zeros((2,), dtype=np.float32)
    agent_history = agents
    agent_types = tracked_objects_types
   
    local_coords_agent_states = []
    for agent_state in agent_history:
        local_coords_agent_states.append(absolute_to_relative_poses(agent_state, anchor_ego_state, 'agent'))

    agents_array = np.zeros((len(agent_history), num_agents, agents_states_dim+3), dtype=np.float32)

    for i, agent_state in enumerate(local_coords_agent_states):
        agent_distance_to_ego = []
        for j in range(agent_state.shape[0]):
            x = agent_state[j, AgentInternalIndex.x()]
            y = agent_state[j, AgentInternalIndex.y()]
            distance = np.hypot(x, y)
            agent_distance_to_ego.append(distance)
        agent_indices = list(np.argsort(agent_distance_to_ego))[:num_agents]
        for j, k in enumerate(agent_indices):
            agents_array[i, j, :3] = se2_ro_Frenet(agents_array[i, j, :3], reference_line, start_point)
            agents_array[i, j, 0] = agent_state[k, AgentInternalIndex.x()]
            agents_array[i, j, 1] = agent_state[k, AgentInternalIndex.y()]
            agents_array[i, j, 2] = np.cos(agent_state[k, AgentInternalIndex.heading()])
            agents_array[i, j, 3] = np.sin(agent_state[k, AgentInternalIndex.heading()])
            agents_array[i, j, 4] = agent_state[k, AgentInternalIndex.vx()]
            agents_array[i, j, 5] = agent_state[k, AgentInternalIndex.vy()]
            agents_array[i, j, 6] = agent_state[k, AgentInternalIndex.length()]
            agents_array[i, j, 7] = agent_state[k, AgentInternalIndex.width()]
            if agent_types[i][k] == TrackedObjectType.VEHICLE:
                agents_array[i, j, -3:] = [1, 0, 0]
            elif agent_types[i][k] == TrackedObjectType.PEDESTRIAN:
                agents_array[i, j, -3:] = [0, 1, 0]
            else: # BICYCLE
                agents_array[i, j, -3:] = [0, 0, 1]
        
    static_objects_array = np.zeros((static_objects.shape[0], 6))
    if static_objects.shape[0] != 0:
        local_coords_static_objects_states = absolute_to_relative_poses(static_objects, anchor_ego_state, 'static')
        local_coords_static_objects_states[:, :3] = se2_to_Frenet_bacth(local_coords_static_objects_states[:, :3], reference_line, start_point)
        static_objects_array[:, 0] = local_coords_static_objects_states[:, 0]
        static_objects_array[:, 1] = local_coords_static_objects_states[:, 1]
        static_objects_array[:, 2] = np.cos(local_coords_static_objects_states[:, 2])
        static_objects_array[:, 3] = np.sin(local_coords_static_objects_states[:, 2])
        static_objects_array[:, 4] = local_coords_static_objects_states[:, 3]
        static_objects_array[:, 5] = local_coords_static_objects_states[:, 4]

    static_objects = np.zeros((num_static, static_objects_array.shape[-1]+4), dtype=np.float32)
    static_distance_to_ego = np.linalg.norm(static_objects_array[:, :2], axis=-1)
    static_indices = list(np.argsort(static_distance_to_ego))[:num_static]

    for i, j in enumerate(static_indices):
        static_objects[i, :static_objects_array.shape[-1]] = static_objects_array[j, :static_objects_array.shape[-1]]
        if static_objects_types[j] == TrackedObjectType.CZONE_SIGN:
            static_objects[i, static_objects_array.shape[-1]:] = [1, 0, 0, 0]
        elif static_objects_types[j] == TrackedObjectType.BARRIER:
            static_objects[i, static_objects_array.shape[-1]:] = [0, 1, 0, 0]
        elif static_objects_types[j] == TrackedObjectType.TRAFFIC_CONE:
            static_objects[i, static_objects_array.shape[-1]:] = [0, 0, 1, 0]
        else: # GENERIC_OBJECT
            static_objects[i, static_objects_array.shape[-1]:] = [0, 0, 0, 1]

    if ego is not None:
        ego = ego.astype(np.float32)

    return ego, agents_array, static_objects


def _filter_agents_array(agents, reverse: bool = False):
    """
    Filter detections to keep only agents which appear in the first frame (or last frame if reverse=True)
    :param agents: The past agents in the scene. A list of [num_frames] arrays, each complying with the AgentInternalIndex schema
    :param reverse: if True, the last element in the list will be used as the filter
    :return: filtered agents in the same format as the input `agents` parameter
    """
    target_array = agents[-1] if reverse else agents[0]
    for i in range(len(agents)):

        rows = []
        for j in range(agents[i].shape[0]):
            if target_array.shape[0] > 0:
                agent_id: float = float(agents[i][j, int(AgentInternalIndex.track_token())])
                is_in_target_frame: bool = bool(
                    (agent_id == target_array[:, AgentInternalIndex.track_token()]).max()
                )
                if is_in_target_frame:
                    rows.append(agents[i][j, :].squeeze())

        if len(rows) > 0:
            agents[i] = np.stack(rows)
        else:
            agents[i] = np.empty((0, agents[i].shape[1]), dtype=np.float32)

    return agents

def _pad_agent_states_with_zeros(agent_trajectories):
    key_frame = agent_trajectories[0]
    track_id_idx = AgentInternalIndex.track_token()

    pad_agent_trajectories = np.zeros((len(agent_trajectories), key_frame.shape[0], key_frame.shape[1]), dtype=np.float32)
    for idx in range(len(agent_trajectories)):
        frame = agent_trajectories[idx]
        mapped_rows = frame[:, track_id_idx]

        for row_idx in range(key_frame.shape[0]):
            if row_idx in mapped_rows:
                pad_agent_trajectories[idx, row_idx] = frame[frame[:, track_id_idx]==row_idx]

    return pad_agent_trajectories

def agent_future_process(anchor_ego_state, future_tracked_objects, num_agents, agent_index):
    
    agent_future = _filter_agents_array(future_tracked_objects)
    local_coords_agent_states = []
    for agent_state in agent_future:
        local_coords_agent_states.append(absolute_to_relative_poses(agent_state, anchor_ego_state, 'agent'))
    padded_agent_states = _pad_agent_states_with_zeros(local_coords_agent_states)

    # fill agent features into the array
    agent_futures = np.zeros(shape=(num_agents, padded_agent_states.shape[0]-1, 3), dtype=np.float32)
    for i, j in enumerate(agent_index):
        agent_futures[i] = padded_agent_states[1:, j, [AgentInternalIndex.x(), AgentInternalIndex.y(), AgentInternalIndex.heading()]]

    return agent_futures
