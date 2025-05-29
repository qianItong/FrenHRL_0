from typing import Deque, List, Tuple
import numpy as np
import numpy.typing as npt
from scipy.interpolate import interp1d

import torch
import math
import copy
import matplotlib.pyplot as plt
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.planning.training.preprocessing.utils.agents_preprocessing import EgoInternalIndex, AgentInternalIndex
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
from nuplan.common.geometry.convert import relative_to_absolute_poses
from nuplan.planning.metrics.utils.state_extractors import approximate_derivatives

def se2_to_matrix(input_data):
    x = input_data[0]
    y = input_data[1]
    yaw = input_data[2]

    return np.array(
        [
            [np.cos(yaw), -np.sin(yaw), x],
            [np.sin(yaw), np.cos(yaw), y],
            [0, 0, 1]
        ]
    )

def se2_to_matrix_bacth(input_data):
    # 提取参数
    x = input_data[:, 0]
    y = input_data[:, 1]
    cos_phi = np.cos(input_data[:, 2])
    sin_phi = np.sin(input_data[:, 2])
    
    # 预分配内存
    batch_size = len(input_data)
    matrices = np.zeros((batch_size, 3, 3))
    
    # 填充矩阵
    matrices[:, 0, 0] = cos_phi
    matrices[:, 0, 1] = -sin_phi
    matrices[:, 0, 2] = x
    matrices[:, 1, 0] = sin_phi
    matrices[:, 1, 1] = cos_phi
    matrices[:, 1, 2] = y
    matrices[:, 2, 2] = 1
    
    return matrices

def matrix_to_se2(matrix):
    """
    Convert a 3x3 transformation matrix to SE(2) parameters (x, y, yaw).
    """
    x = matrix[0, 2]
    y = matrix[1, 2]
    yaw = np.arctan2(matrix[1, 0], matrix[0, 0])
    return np.array([x, y, yaw])

def matrix_to_se2_batch(matrices):
    """
    Convert a batch of 3x3 transformation matrices to SE(2) parameters (x, y, yaw).
    """
    x = matrices[:, 0, 2]
    y = matrices[:, 1, 2]
    yaw = np.arctan2(matrices[:, 1, 0], matrices[:, 0, 0])
    return np.stack((x, y, yaw), axis=1)

def se2_transform(input_data, target_data):
    """
    Transform a batch of SE(2) parameters to the target coordinate system.
    The input data is transformed to the target coordinate system defined by the target data.
    The target data is a single SE(2) parameter.
    The input data is a batch of SE(2) parameters.
    The function returns the transformed input data in the target coordinate system.
    :param input_data: A batch of SE(2) parameters (x, y, yaw).
    :param target_data: A single SE(2) parameter (x, y, yaw) defining the target coordinate system.
    :return: A batch of transformed SE(2) parameters (x, y, yaw) in the target coordinate system.
    """
    traget_matrix = se2_to_matrix(target_data)
    traget_matrix_inv = np.linalg.inv(traget_matrix)

    transforms = se2_to_matrix_bacth(input_data)
    transforms = np.matmul(traget_matrix_inv, transforms)

    return matrix_to_se2_batch(transforms), transforms
    
def velocity_transform(velocity, target_heading):
    v_x = velocity[:, 0] * np.cos(target_heading) + velocity[:, 1] * np.sin(target_heading)
    v_y = -velocity[:, 0] * np.sin(target_heading) + velocity[:, 1] * np.cos(target_heading)
    return np.stack((v_x, v_y), axis=1)

def local_to_global_SE2(local_pose, global_origin_pose):
    # 转换为齐次矩阵
    T_local = se2_to_matrix_bacth(local_pose) if local_pose.ndim > 1 else se2_to_matrix(local_pose)
    T_global_origin = se2_to_matrix(global_origin_pose)
    
    # 计算全局位姿: T_world = T_global_origin @ T_local
    T_world = np.matmul(T_global_origin, T_local)
    
    # 转换回 SE(2) 参数
    return matrix_to_se2_batch(T_world) if local_pose.ndim > 1 else matrix_to_se2_batch(T_world[np.newaxis])[0]

def absolute_to_relative_poses(agent_state, ego_state, agent_type = 'ego'):
    ego_se2 = np.array(ego_state, dtype=np.float32)
    ego_heading = ego_se2[-1]

    if agent_type == 'ego':
        global_state = agent_state[:,[EgoInternalIndex.x(), EgoInternalIndex.y(), EgoInternalIndex.heading()]]
        transformed_pose, transform= se2_transform(global_state, ego_se2)
        agent_state[:, [EgoInternalIndex.x(), EgoInternalIndex.y(), EgoInternalIndex.heading()]] = transformed_pose
        local_v = agent_state[:, [EgoInternalIndex.vx(), EgoInternalIndex.vy()]]
        local_a = agent_state[:, [EgoInternalIndex.ax(), EgoInternalIndex.ay()]]
        local_v = np.expand_dims(np.concatenate((local_v, np.zeros((local_v.shape[0], 1))), axis=-1), axis=-1)
        local_a = np.expand_dims(np.concatenate((local_a, np.zeros((local_a.shape[0], 1))), axis=-1), axis=-1)
        transformed_v = np.matmul(transform, local_v).squeeze(axis=-1)
        transformed_a = np.matmul(transform, local_a).squeeze(axis=-1)
        agent_state[:, [EgoInternalIndex.vx(), EgoInternalIndex.vy(), EgoInternalIndex.ax(), EgoInternalIndex.ay()]] = \
    np.column_stack((transformed_v[:, :2], transformed_a[:, :2]))
        
    elif agent_type == 'agent':
        global_state = agent_state[:,[AgentInternalIndex.x(), AgentInternalIndex.y(), AgentInternalIndex.heading()]]
        transformed_pose, transform= se2_transform(global_state, ego_se2)
        global_velocity = agent_state[:, [AgentInternalIndex.vx(), AgentInternalIndex.vy()]]
        transformed_velocity = velocity_transform(global_velocity, ego_heading)
        agent_state[:, [AgentInternalIndex.x(), AgentInternalIndex.y(), AgentInternalIndex.heading()]] = transformed_pose
        agent_state[:, [AgentInternalIndex.vx(), AgentInternalIndex.vy()]] = transformed_velocity
    
    elif agent_type == 'static':
        global_state = agent_state[:,[0,1,2]]
        transformed_pose, transform= se2_transform(global_state, ego_se2)
        agent_state[:, [EgoInternalIndex.x(), EgoInternalIndex.y(), EgoInternalIndex.heading()]] = transformed_pose

    else:
        raise ValueError("Invalid agent type. Expected 'ego', 'agent', or 'static'.")

    return agent_state

# =====================
# 2. Map coordination transformation
# =====================
def coordinates_to_local_frame(
    coords, anchor_state, precision = None
):
    """
    Transform a set of [x, y] coordinates without heading to the the given frame.
    :param coords: <np.array: num_coords, 2> Coordinates to be transformed, in the form [x, y].
    :param anchor_state: The coordinate frame to transform to, in the form [x, y, heading].
    :param precision: The precision with which to allocate the intermediate array. If None, then it will be inferred from the input precisions.
    :return: <np.array: num_coords, 2> Transformed coordinates.
    """
    if len(coords.shape) != 2 or coords.shape[1] != 2:
        raise ValueError(f"Unexpected coords shape: {coords.shape}")

    if precision is None:
        if coords.dtype != anchor_state.dtype:
            raise ValueError("Mixed datatypes provided to coordinates_to_local_frame without precision specifier.")
        precision = coords.dtype

    # torch.nn.functional.pad will crash with 0-length inputs.
    # In that case, there are no coordinates to transform.
    if coords.shape[0] == 0:
        return coords

    # Extract transform
    transform = se2_to_matrix(anchor_state)
    transform = np.linalg.inv(transform)

    # Transform the incoming coordinates to homogeneous coordinates
    #  So translation can be done with a simple matrix multiply.
    #
    # [x1, y1]  => [x1, y1, 1]
    # [x2, y2]     [x2, y2, 1]
    # ...          ...
    # [xn, yn]     [xn, yn, 1]
    coords = np.pad(coords, pad_width=((0, 0), (0, 1)), mode='constant', constant_values=1.0)

    # Perform the transformation, transposing so the shapes match
    coords = np.matmul(transform, coords.T)

    # Transform back from homogeneous coordinates to standard coordinates.
    #   Get rid of the scaling dimension and transpose so output shape matches input shape.
    result = coords.T
    result = result[:, :2]

    return result


def vector_set_coordinates_to_local_frame(
    coords,
    avails,
    anchor_state,
    output_precision = np.float32,
):
    """
    Transform the vector set map element coordinates from global frame to ego vehicle frame, as specified by
        anchor_state.
    :param coords: Coordinates to transform. <np.array: num_elements, num_points, 2>.
    :param avails: Availabilities mask identifying real vs zero-padded data in coords.
        <np.array: num_elements, num_points>.
    :param anchor_state: The coordinate frame to transform to, in the form [x, y, heading].
    :param output_precision: The precision with which to allocate output array.
    :return: Transformed coordinates.
    :raise ValueError: If coordinates dimensions are not valid or don't match availabilities.
    """


    # Flatten coords from (num_map_elements, num_points_per_element, 2) to
    #   (num_map_elements * num_points_per_element, 2) for easier processing.
    num_map_elements, num_points_per_element, _ = coords.shape
    coords = coords.reshape(num_map_elements * num_points_per_element, 2)

    # Apply transformation using adequate precision
    coords = coordinates_to_local_frame(coords, anchor_state, precision=np.float64)

    # Reshape to original dimensionality
    coords = coords.reshape(num_map_elements, num_points_per_element, 2)

    # Output with specified precision
    coords = coords.astype(output_precision)

    # ignore zero-padded data
    coords[~avails] = 0.0

    return coords


# =====================
# 3. Numpy-Tensor transformation
# =====================
def convert_to_model_inputs(data, device):
    tensor_data = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray) and v.dtype == np.bool_:
            tensor_data[k] = torch.tensor(v, dtype=torch.bool).unsqueeze(0).to(device)
        else:
            tensor_data[k] = torch.tensor(v, dtype=torch.float32).unsqueeze(0).to(device)

    return tensor_data

# =====================
# 4. Basic math functions
# =====================

def get_2d_rotation_matrix(orientation):
    qx, qy, qz, qw = orientation
    theta = 2 * np.arctan2(qz, qw)
    return np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ]), theta

def take_first_n(lst, n):
    length = len(lst)
    if length >= n:
        return lst[:n]
    else:
        last_element = lst[-1]
        result = copy.deepcopy(lst)
        remaining_count = n - length
        for _ in range(remaining_count):
            result.append(last_element)
        return result

def take_first_n_step(lst, n):
    length = len(lst)
    if length >= n:
        step = length / n
        return [lst[int(i * step)] for i in range(n)]
    else:
        return lst       
def extend_list(lst, n):
    length = len(lst)
    if len(lst) == 0:
        return []
    if len(lst) == 1:
        return [lst[0]]*n
    if length >= n:
        return lst[:n]
    else:
        while len(lst) < n:
            result = copy.deepcopy(lst)
            bias = 0
            for i in range(len(lst)-1):
                result.insert(i+1+bias, (lst[i]+lst[i+1])/2)
                bias += 1
                if len(result) == n:
                    return result
            lst = result

def take_first_n_zeros(lst, n):
    length = len(lst)
    if length >= n:
        return lst[:n]
    else:
        result = copy.deepcopy(lst)
        remaining_count = n - length
        for _ in range(remaining_count):
            result.append(0)

        return result

def list_padding(lst, n, padding_value=0):
    length = len(lst)
    if length >= n:
        return lst[:n]
    else:
        result = copy.deepcopy(lst)
        remaining_count = n - length
        for _ in range(remaining_count):
            result.append(padding_value)
        return result


def draw_rectangle(center, length, width, theta, color='red'):
    cx, cy = center
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])

    half_length = length / 2
    half_width = width / 2

    vertices = np.array([
        [-half_length, -half_width],
        [-half_length, half_width],
        [half_length, half_width],
        [half_length, -half_width]
    ])

    rotated_vertices = np.dot(vertices, R.T) + np.array([cx, cy])

    # plt.figure()
    plt.plot(*zip(*np.append(rotated_vertices, [rotated_vertices[0]], axis=0)), 'b-')
    plt.scatter(cx, cy, color=color)

def rotate_point(point, theta):
    x, y = point
    return np.array([
        x * np.cos(theta) - y * np.sin(theta),
        x * np.sin(theta) + y * np.cos(theta)
    ])

def point2line_distance(x,y,line) -> float:
    '''
    计算点到直线的距离
    x,y: 点坐标
    line: 直线坐标(list,[[x1,y1],[x2,y2]])
    '''
    x1, y1 = line[0]
    x2, y2 = line[1]
    if x1 == x2:
        return abs(x - x1)
    if y1 == y2:
        return abs(y - y1)
    k = (y2-y1)/(x2-x1)
    b = y1 - k*x1
    return abs(k*x-y+b)/math.sqrt(k*k+1)

def point2point_distance(x1,y1,x2,y2) -> float:
    '''
    计算两点之间的距离
    '''
    return math.hypot(x1-x2, y1-y2)

def point2curve_distance(x,y,curve) -> Tuple[float, int]:
    '''
    计算点到曲线的距离
    x,y: 点坐标
    curve: 曲线坐标(list,[[x1,y1],[x2,y2],...])
    '''
    if len(curve) == 0:
        KeyError('curve is empty')
    if len(curve) < 2:
        return point2point_distance(x,y,curve[0][0],curve[0][1])
    index = 0
    nearest_point = curve[0]
    min_distance = float('inf')
    for point in curve:
        distance = point2point_distance(x, y, point[0], point[1])
        if distance < min_distance:
            min_distance = distance
            index = curve.index(point)
            nearest_point = point

    another_point = curve[index+1] if index+1 < len(curve) else curve[index-1]

    return point2line_distance(x,y,[nearest_point,another_point]), index

def point2curve_nearest_point(x,y,curve) -> Tuple[float, list]:
    '''
    获取曲线上距离(x,y)最近的点的距离
    x,y: 点坐标
    curve: 曲线坐标(list,[[x1,y1],[x2,y2],...])
    '''
    min_distance = float('inf')
    p = []
    for point in curve:
        distance = point2point_distance(x, y, point[0], point[1])
        if distance < min_distance:
            min_distance = distance
            p = point
    return min_distance, p

def get_nearest_point(x,y,points,n) -> list:
    '''
    获取距离(x,y)最近的n个点
    x,y: 点坐标
    points: 点坐标列表(list,[[x1,y1],[x2,y2],...])
    n: 要获取的点的数量,按照原始顺序返回
    '''
    points_copy = copy.deepcopy(points)
    points_copy = np.array(points_copy)
    points_sorted_index = np.argsort(np.linalg.norm(points_copy - np.array([x, y]), axis=1))
    indices = points_sorted_index[:n]
    indices = indices.tolist()
    indices.sort()
    return [points[i] for i in indices]

def get_nearest_line(x, y, lines, n) -> Tuple[list, list]:
    '''
    获取距离(x,y)最近的n条线段，并返回它们的原始下标
    x, y: 点坐标
    lines: 线段坐标列表(list, [[x1, y1], [x2, y2], ...])
    n: 要获取的线段的数量
    mode: 选择距离的方式，distance表示距离，point表示点到点最近的线段
    '''
    lines_copy = copy.deepcopy(lines)
    indexed_lines = list(enumerate(lines_copy))  # 将线段与其原始下标绑定

    indexed_lines.sort(key=lambda line: point2curve_nearest_point(x, y, line[1])[0])

    index = 0
    while index < len(indexed_lines) and all_zeros(indexed_lines[index][1]):
        index += 1

    nearest_lines = indexed_lines[index:index + n]
    indices = [line[0] for line in nearest_lines]  # 提取原始下标
    coordinates = [line[1] for line in nearest_lines]  # 提取线段坐标

    # 返回最近的n条线段及其原始下标
    return coordinates, indices

def is_point_left_of_line(x,y,line) -> bool:
    '''
    判断点是否在线段的左侧
    x,y: 点坐标
    line: 直线坐标(list,[[x1,y1],[x2,y2]])
    '''

    x1, y1 = line[0]
    x2, y2 = line[1]
    if x1 == x2:
        return x < x1
    if y1 == y2:
        return y < y1
    k = (y2-y1)/(x2-x1)
    b = y1 - k*x1
    result = y > k*x+b
    if x2 < x1:
        result = not result
    return result

def vector_angle(v1, v2):
    """
    计算两个二维向量之间的角度（单位：弧度）。
    返回值在 [-π, π] 范围内。
    :param v1: 第一个向量，格式为 [x1, y1]
    :param v2: 第二个向量，格式为 [x2, y2]
    :return: 两个向量之间的夹角（单位: 度）
    返回负值表示v2在v1的左侧，正值表示v2在v1的右侧
    """
    v1 = np.array(v1)
    v2 = np.array(v2)
    dot_product = np.dot(v1, v2)
    cross_product = v1[0] * v2[1] - v1[1] * v2[0]
    angle = np.arctan2(cross_product, dot_product)
    return np.degrees(angle)

def project_point(p, a, b):
    '''
    计算点到线段的投影点
    p: 点坐标
    a: 线段起点坐标
    b: 线段终点坐标
    返回值: 投影点坐标，投影点在线段上的参数t
    '''
    ax, ay = a
    bx, by = b
    apx = p[0] - ax
    apy = p[1] - ay
    abx = bx - ax
    aby = by - ay

    if abx == 0 and aby == 0:
        return a, 0.0

    t = (apx * abx + apy * aby) / (abx**2 + aby**2)
    proj_x = ax + t * abx
    proj_y = ay + t * aby
    return (proj_x, proj_y), t

def find_nearest_segment(p, curve):
    '''
    找到曲线上距离点p最近的线段
    p: 点坐标
    curve: 曲线坐标(list,[[x1,y1],[x2,y2],...])
    返回值: 最近线段的索引，最近线段上的参数t
    '''
    min_dist_sq = float('inf')
    best_i = -1
    best_t = 0.0

    for i in range(len(curve) - 1):
        a = curve[i]
        b = curve[i+1]

        dist_sq = (p[0]-a[0])**2 + (p[1]-a[1])**2+ (p[0]-b[0])**2 + (p[1]-b[1])**2

        if dist_sq < min_dist_sq:
            _, t = project_point(p, a, b)
            min_dist_sq = dist_sq
            best_i = i
            best_t = t

    return best_i, best_t

def compute_cum_dist(curve):
    '''
    计算曲线上每个点到起点的累计距离
    curve: 曲线坐标(list,[[x1,y1],[x2,y2],...])
    返回值: 累计距离列表
    '''
    cum_dist = [0.0]
    for i in range(1, len(curve)):
        a = curve[i-1]
        b = curve[i]
        dx = b[0] - a[0]
        dy = b[1] - a[1]
        cum_dist.append(cum_dist[-1] + math.hypot(dx, dy))
    return cum_dist

def calculate_curve_distance(p1, p2, curve):
    '''
    计算曲线上两点之间的距离
    '''
    if len(curve) < 2:
        return math.hypot(p2[0]-p1[0], p2[1]-p1[1])
    cum_dist = compute_cum_dist(curve)
    
    # Find nearest segment and parameter for p1
    i1, t1 = find_nearest_segment(p1, curve)
    a1, b1 = curve[i1], curve[i1+1]
    seg_len1 = math.hypot(b1[0]-a1[0], b1[1]-a1[1])
    s1 = cum_dist[i1] + t1 * seg_len1
    
    # Find nearest segment and parameter for p2
    i2, t2 = find_nearest_segment(p2, curve)
    a2, b2 = curve[i2], curve[i2+1]
    seg_len2 = math.hypot(b2[0]-a2[0], b2[1]-a2[1])
    s2 = cum_dist[i2] + t2 * seg_len2
    
    return abs(s2 - s1)

def get_cloeset_line_to_line(target_line, lines,angle_tolerance=None) -> Tuple[list, int, float]:
    min_distance = float('inf')
    closest_line = None
    index = 0
    min_index = -1
    
    for line in lines:
        if len(line) == 2:
            distance, _ = point2curve_nearest_point(line[0][0], line[0][1], target_line)
        else:
            p1 = []
            p2 = []
            q1 = []
            q2 = []
            min_1 = float('inf')
            min_2 = float('inf')
            min_3 = float('inf')
            for point in target_line:
                distance,q_tmp = point2curve_nearest_point(point[0], point[1], line)
                if distance < min_1:
                    min_3 = min_2
                    min_2 = min_1
                    min_1 = distance
                    p2 = p1
                    p1 = point
                    q2 = q1
                    q1 = q_tmp
                elif distance < min_2:
                    min_3 = min_2
                    min_2 = distance
                    q2 = q_tmp
                    p2 = point
                elif distance < min_3:
                    min_3 = distance
            distance = (min_1 + min_2 + min_3) / 3
            if angle_tolerance is not None:
                p1_index = target_line.index(p1)
                p2_index = target_line.index(p2)
                q1_index = line.index(q1)
                q2_index = line.index(q2)
                if p1_index > p2_index:
                    p1, p2 = p2, p1
                if q1_index > q2_index:
                    q1, q2 = q2, q1
                angle = vector_angle([p1[0]-p2[0], p1[1]-p2[1]], [q1[0]-q2[0], q1[1]-q2[1]])
                if abs(angle) > angle_tolerance:
                    distance += 1000

        if distance < min_distance:
            min_distance = distance
            closest_line = line
            min_index = index
        index += 1
    return closest_line, min_index, min_distance

def all_zeros(arr):
    """
    判断数组是否全
    :param arr: 输入数组
    :return: 如果全为0，返回True，否则返回False
    """
    if type(arr) == torch.Tensor:
        return torch.all(arr == 0)
    elif type(arr) == np.ndarray:
        return np.all(arr == 0)
    elif type(arr) == list:
        arr = np.array(arr)
        return np.all(arr == 0)
    else:
        raise ValueError("Unsupported data type. Expected torch.Tensor, np.ndarray or list.")
    
def draw_figure(data, save_path):
    lanes = data['lanes']
    route_lanes = data['route_lanes']
    reference_line = data['reference_line']
    trajectory = data['trajectory']
    next_frame_agents = data['next_frame_agents']
    neighbor_agents_past = data['neighbor_agents_past']
    Frenet_trajectory = data['Frenet_trajectory']

    plt.figure()
    plt.subplot(1, 3, 1)
    for lane in lanes:
        plt.plot(lane[:, 0], lane[:, 1], 'b')
        # plt.scatter(lane[0, 0], lane[0, 1], c='r', marker='o', alpha=0.1)
        # plt.scatter(lane[-1, 0], lane[-1, 1], c='b', marker='*')
    for route_lane in route_lanes:
        plt.plot(route_lane[:, 0], route_lane[:, 1], 'r')
        # plt.scatter(route_lane[0, 0], route_lane[0, 1], c='r', marker='o',alpha=0.5)
        # plt.scatter(route_lane[-1, 0], route_lane[-1, 1], c='b', marker='*')
    plt.plot(reference_line[:, 0], reference_line[:, 1], 'r')
    # for point in reference_line:
    #     plt.scatter(point[0], point[1], c='r', marker='o', s=7)

    for state in trajectory:
        future_point = [state[0], state[1]]
        plt.scatter(future_point[0], future_point[1], c='k', marker='+')
    current_agents = next_frame_agents
    for i in range(current_agents.shape[0]):
        draw_rectangle([current_agents[i, 0], current_agents[i, 1]], current_agents[i, 5], current_agents[i, 6], current_agents[i, 2], color='r')
    current_agents = neighbor_agents_past[-1]
    for i in range(current_agents.shape[0]):
        draw_rectangle([current_agents[i, 0], current_agents[i, 1]], current_agents[i, 5], current_agents[i, 6], current_agents[i, 2], color='b')
    plt.axis('equal')

    plt.subplot(1, 3, 2)
    plt.plot(Frenet_trajectory[:, 0], 'bo-')
    plt.axis('equal')
    plt.subplot(1, 3, 3)
    plt.plot(Frenet_trajectory[:, 1], 'ro-')
    plt.axis('equal')
    
    plt.savefig(save_path)



def trajectory_to_state(trajectory):
    """
    将轨迹转换为状态,包括位置、速度、加速度、角度
    
    """
    time_step = 0.5
    trajectory = np.array(trajectory)
    num_points = trajectory.shape[0]
    if num_points < 2:
        raise ValueError("轨迹点数不足，无法计算速度和加速度")
    if num_points < 3:
        velocity = np.zeros((num_points, 2))
        acceleration = np.zeros((num_points, 2))
        velocity[0] = (trajectory[1] - trajectory[0]) / time_step
        acceleration[0] = (velocity[1] - velocity[0]) / time_step
        heading = np.zeros((num_points, 1))
        heading[0] = np.arctan2(velocity[0][1], velocity[0][0])
        return np.concatenate((trajectory, heading, velocity, acceleration), axis=1)
    else:
        velocity = np.zeros((num_points, 2))
        acceleration = np.zeros((num_points, 2))
        velocity[0] = (trajectory[1] - trajectory[0]) / time_step
        velocity[1:-1] = (trajectory[2:] - trajectory[:-2]) / (2 * time_step)
        velocity[-1] = (trajectory[-1] - trajectory[-2]) / time_step
        acceleration[0] = (velocity[1] - velocity[0]) / time_step
        acceleration[1:-1] = (velocity[2:] - velocity[:-2]) / (2 * time_step)
        acceleration[-1] = (velocity[-1] - velocity[-2]) / time_step
        heading = np.zeros((num_points,))
        heading[0] = np.arctan2(velocity[0][1], velocity[0][0])
        heading[1:-1] = (np.arctan2(velocity[2:, 1], velocity[2:, 0]) + np.arctan2(velocity[:-2, 1], velocity[:-2, 0])) / 2
        heading[-1] = np.arctan2(velocity[-1][1], velocity[-1][0])
        heading = np.expand_dims(heading, axis=1)
        return np.concatenate((trajectory, heading, velocity, acceleration), axis=1)