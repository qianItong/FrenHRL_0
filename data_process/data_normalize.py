import numpy as np
import json

def normalize_data(data, refer_file = './normalization.json'):
    """
    Normalize the data based on the reference file.

    Args:
        data (numpy.ndarray): The data to be normalized.
        refer_file (numpy.ndarray): The reference file for normalization.
    target_keys = ["ego_current_state","neighbor_agents_past","static_objects","lanes","route_lanes"]

    Returns:
        numpy.ndarray: The normalized data.
    """
    standard = json.load(open(refer_file, 'r'))

    nomalize_dict(data, standard)

    # reshape the lanes and route_lanes
    lanes_speed_limit = data['lanes_speed_limit']
    lanes = data['lanes']
    new_lanes = np.zeros([lanes.shape[0], lanes.shape[1]*8 + 4 + 1])
    for i in range(lanes.shape[1]):
        new_lanes[:, i*8:i*8+8] = lanes[:, i,:8]
    new_lanes[:, -5:-1] = lanes[:, -1, -4:]
    new_lanes[:, -1] = lanes_speed_limit.reshape(-1)
    data['lanes'] = new_lanes

    route_lanes_speed_limit = data['route_lanes_speed_limit']
    route_lanes = data['route_lanes']
    new_route_lanes = np.zeros([route_lanes.shape[0], route_lanes.shape[1]*8 + 4 + 1])
    for i in range(route_lanes.shape[1]):
        new_route_lanes[:, i*8:i*8+8] = route_lanes[:, i,:8]
    new_route_lanes[:, -5:-1] = route_lanes[:, -1, -4:]
    new_route_lanes[:, -1] = route_lanes_speed_limit.reshape(-1)
    data['route_lanes'] = new_route_lanes
    for key in data.keys():
        if type(data[key]) == list:
            data[key] = np.array(data[key])
    return data

def nomalize_dict(data, standard):

    ego_current_state = data['ego_current_state']
    ego_current_state_mean = standard['ego_current_state']['mean']
    ego_current_state_std = standard['ego_current_state']['std']
    ego_current_state = (ego_current_state - ego_current_state_mean) / ego_current_state_std
    data['ego_current_state'] = ego_current_state

    neighbor_agents_past = data['neighbor_agents_past']
    neighbor_agents_past_mean = standard['neighbor_agents_past']['mean']
    neighbor_agents_past_std = standard['neighbor_agents_past']['std']
    mask = np.all(neighbor_agents_past == 0, axis=-1)
    neighbor_agents_past[:,:] = (neighbor_agents_past[:,:] - neighbor_agents_past_mean) / neighbor_agents_past_std
    neighbor_agents_past[mask] = 0
    data['neighbor_agents_past'] = neighbor_agents_past

    static_objects = data['static_objects']
    static_objects_mean = standard['static_objects']['mean']
    static_objects_std = standard['static_objects']['std']
    mask = np.all(static_objects == 0, axis=-1)
    static_objects[:] = (static_objects[:] - static_objects_mean) / static_objects_std
    static_objects[mask] = 0
    data['static_objects'] = static_objects

    lanes = data['lanes']
    lanes_mean = standard['lanes']['mean']
    lanes_std = standard['lanes']['std']
    mask = np.all(lanes == 0, axis=-1)
    lanes[:,:] = (lanes[:,:] - lanes_mean) / lanes_std
    lanes[mask] = 0
    data['lanes'] = lanes

    route_lanes = data['route_lanes']
    route_lanes_mean = standard['route_lanes']['mean']
    route_lanes_std = standard['route_lanes']['std']
    mask = np.all(route_lanes == 0, axis=-1)
    route_lanes[:,:] = (route_lanes[:,:] - route_lanes_mean) / route_lanes_std
    route_lanes[mask] = 0
    data['route_lanes'] = route_lanes

    reference_line = data['reference_line']
    reference_line_mean = standard['reference_line']['mean']
    reference_line_std = standard['reference_line']['std']
    reference_line[:] = (reference_line[:] - reference_line_mean) / reference_line_std
    data['reference_line'] = reference_line