import math
from typing import List, Dict, Tuple, Set
import numpy as np
from Model.util import find_point_by_arc_length
from data_process.utils import compute_cum_dist, find_nearest_segment, get_nearest_line, point2line_distance,point2point_distance,take_first_n_step,extend_list,calculate_curve_distance, is_point_left_of_line, point2curve_distance,all_zeros,point2curve_nearest_point,vector_angle,get_cloeset_line_to_line
import copy

def get_single_reference_line(
        route_lanes: List,
        length: int,
        lanes: List,
        mission_point: List,
        _last_relative_trajectory=None,
)-> np.ndarray:
    """
    Get the reference line from the route lanes and past trajectory.
    :param route_lanes: List of route lanes.
    :return: Reference line as a list of points for Frenet coordinates.
    """
    if all_zeros(route_lanes) and all_zeros(lanes):
        step = 100.0/length
        reference_line = [[i*step, 0] for i in range(length)]
        return np.array(reference_line).reshape(-1, 2)
    if mission_point is None:
        mission_point = [0, 0]
    reference_line = []
    all_lanes = []
    for lane in lanes:
        if not all_zeros(lane):
            all_lanes.append(lane)
    
    routes = []
    for lane in route_lanes:
        if not all_zeros(lane):
            routes.append(lane)
        if not (lane in all_lanes):
            all_lanes.append(lane)
    
    distance = float('inf')
    if len(routes) > 0:
        lines_copy = copy.deepcopy(routes)
        distance = [point2curve_nearest_point(0,0, line)[0] for line in lines_copy]
        indexed_lines = sorted(enumerate(distance), key=lambda x: x[1])
        index = 0
        choose_index = []
        choose_list = []
        while index < len(indexed_lines) and all_zeros(lines_copy[indexed_lines[index][0]]):
            index += 1
        while index < len(indexed_lines):
            if indexed_lines[index][1] < 3:
                choose_index.append(indexed_lines[index][0])
                choose_list.append(lines_copy[indexed_lines[index][0]])
            index += 1
        if len(choose_index) == 1:
            current_route_lane = lines_copy[choose_index[0]]
        elif len(choose_index) > 1:
            if _last_relative_trajectory is None:
                current_route_lane = get_nearest_line(0, 0, choose_list, 1)[0][0]
            else:
                total_distance = [0] * len(choose_list)
                for point in _last_relative_trajectory:
                    distance_list = [point2curve_nearest_point(point[0], point[1], choose_list[i])[0] for i in range(len(choose_list))]
                    for i in range(len(choose_list)):
                        total_distance[i] += distance_list[i]
                    max_distance = max(total_distance)
                    min_distance = min(total_distance)
                    # if max_distance - min_distance > 10:
                    #     break
                choose_index = total_distance.index(min_distance)
                current_route_lane = choose_list[choose_index]
        else:
            current_route_lane = get_nearest_line(0,0, all_lanes, 1)[0][0]
    else:
        current_route_lane = get_nearest_line(0,0, all_lanes, 1)[0][0]

    visited = [False] * len(all_lanes)
    reference_line.extend(current_route_lane)
    visited[all_lanes.index(current_route_lane)] = True
    end_point = current_route_lane[-1]
    find_next_route_lane = True
    while(find_next_route_lane):
        find_next_route_lane = False
        choose_list = []
        for i in range(len(routes)):
            current_route = routes[i]
            if visited[all_lanes.index(current_route)]:
                continue
            begin_point = current_route[0]
            distance = point2point_distance(begin_point[0], begin_point[1], end_point[0], end_point[1])
            vector = [current_route[1][0] - current_route[0][0], current_route[1][1] - current_route[0][1]]
            if distance < 2: 
                find_next_route_lane = True
                choose_list.append(current_route)
                visited[all_lanes.index(current_route)] = True

        if not find_next_route_lane:
            for i in range(len(all_lanes)):
                if visited[i]:
                    continue
                current_lane = all_lanes[i]
                begin_point = current_lane[0]
                distance = point2point_distance(begin_point[0], begin_point[1], end_point[0], end_point[1])
                vector = [current_lane[1][0] - current_lane[0][0], current_lane[1][1] - current_lane[0][1]]
                if distance < 2 and abs(vector_angle(vector, [1, 0]))<=90 and point2curve_nearest_point(current_lane[-1][0], current_lane[-1][1], reference_line)[0] > 3:
                    find_next_route_lane = True
                    choose_list.append(current_lane)
                    visited[i] = True
        if len(choose_list) == 1:
            reference_line.extend(choose_list[0][1:])
            end_point = choose_list[0][-1]
        elif len(choose_list) > 1:
            if _last_relative_trajectory is None:
                choose_lane = get_nearest_line(mission_point[0], mission_point[1], choose_list, 1)[0][0]
            else:
                index = 0
                min_index = 0
                min_distance = float('inf')
                for point in _last_relative_trajectory:
                    if index >= len(_last_relative_trajectory)-3:
                        break
                    distance = point2point_distance(point[0], point[1], end_point[0], end_point[1])
                    if distance < min_distance:
                        min_distance = distance
                        min_index = index
                    index += 1
                choose_lane, _, _ = get_cloeset_line_to_line(_last_relative_trajectory[min_index+1:].tolist(),choose_list)
            reference_line.extend(choose_lane[1:])
            end_point = choose_lane[-1]
        else:
            break
    

    reference_line = fix_list_length(reference_line, length)

    return np.array(reference_line).reshape(-1, 2)


def fix_list_length(lane_list: List, max_len: int) -> List:
    """
    Fix the length of the lane list to the max length.
    :param lane_list: List of lanes.
    :param max_len: Max length of the lane list.
    :return: Fixed length lane list.
    """
    if len(lane_list) > max_len:
        lane_list = take_first_n_step(lane_list, max_len)
    elif len(lane_list) < max_len:
        lane_x = [lane_list[i][0] for i in range(len(lane_list))]
        lane_y = [lane_list[i][1] for i in range(len(lane_list))]
        extend_lane_x = extend_list(lane_x, max_len)
        extend_lane_y = extend_list(lane_y, max_len)
        lane_list = [[extend_lane_x[i], extend_lane_y[i]] for i in range(max_len)]
    return lane_list

def se2_ro_Frenet(
        relative_se2: np.ndarray,
        reference_line: np.ndarray,
        start_point: np.ndarray,
) -> np.ndarray:
    """
    Convert the relative SE2 coordinates to Frenet coordinates.
    :param relative_se2: Relative SE2 coordinates as a list of points.
    :param reference_line: Reference line as a list of points.
    :return: Frenet coordinates as a list of points.
    """
    x = relative_se2[0]
    y = relative_se2[1]
    heading = relative_se2[2]

    index, t = find_nearest_segment([x, y], reference_line)
    F_y = point2line_distance(x, y, [reference_line[index], reference_line[index+1]])
    another_index = index + 1
    pos_y = is_point_left_of_line(x, y, [reference_line[index], reference_line[another_index]])
    F_y = F_y if pos_y else -F_y
    F_x = calculate_curve_distance(start_point, [x,y], reference_line)
    frenet_heading = np.arctan2(reference_line[another_index][1] - reference_line[index][1],
                                reference_line[another_index][0] - reference_line[index][0])
    F_heading = heading - frenet_heading

    return np.array([F_x, F_y, F_heading])

def se2_to_Frenet_bacth(
        relative_se2: np.ndarray,
        reference_line: np.ndarray,
        start_point: np.ndarray,
):
    """
    Convert the relative SE2 coordinates to Frenet coordinates.
    :param relative_se2: Relative SE2 coordinates as a list of points.
    :param reference_line: Reference line as a list of points.
    :param start_point: Start point of the reference line(ego_x ego_y).
    :return: Frenet coordinates as a list of points.
    """
    frenet_trajectory = []
    for i in range(len(relative_se2)):
        point = relative_se2[i]
        F_point = se2_ro_Frenet(point, reference_line, start_point)
        frenet_trajectory.append(F_point)
    return np.array(frenet_trajectory)

def relative_to_Frenet(
        start: np.ndarray,
        trajectory: np.ndarray,
        reference_line: np.ndarray,
) -> np.ndarray:
    """
    Convert the trajectory to Frenet coordinates.
    :param trajectory: Trajectory as a list of points.
    :param reference_line: Reference line as a list of points.
    :return: Frenet coordinates as a list of points.
    """
    trajectory = trajectory.tolist()
    reference_line = reference_line.tolist()
    frenet_trajectory = []

    index, t = find_nearest_segment(start[:2], reference_line)
    F_self_y = point2line_distance(start[0], start[1], [reference_line[index], reference_line[index+1]])
    another_index = index + 1

    pos_self_y = is_point_left_of_line(start[0], start[1], [reference_line[index], reference_line[another_index]])
    F_self_y = F_self_y if pos_self_y else -F_self_y
    F_self_x = 0
    frenet_heading = np.arctan2(reference_line[another_index][1] - reference_line[index][1],
                                reference_line[another_index][0] - reference_line[index][0])
    F_self_heading = start[2] - frenet_heading
    F_self = np.array([F_self_x, F_self_y, F_self_heading])

    for i in range(len(trajectory)):
        point = trajectory[i]
        x = point[0]
        y = point[1]

        index, t = find_nearest_segment([x, y], reference_line)
        F_y = point2line_distance(x, y, [reference_line[index], reference_line[index+1]])
        another_index = index + 1
        pos_y = is_point_left_of_line(x, y, [reference_line[index], reference_line[another_index]])
        F_y = F_y if pos_y else -F_y
        F_x = calculate_curve_distance(start[:2], point, reference_line)
        # if x < -0.2 and abs(y)< 2:
        #     F_x = -F_x
        F_y = F_y - F_self_y
        frenet_heading = np.arctan2(reference_line[another_index][1] - reference_line[index][1],
                                reference_line[another_index][0] - reference_line[index][0])
        F_heading = point[2] - frenet_heading
        frenet_trajectory.append(np.array([float(F_x), float(F_y), float(F_heading)]))

    return np.array(frenet_trajectory),F_self

def Frent_to_relative(frenet_trajectory, reference_line, start_point):
    """
    将轨迹从车道中心线坐标系转换到相对坐标系
    :param frenet_trajectory: 轨迹点
    :param lane_center_line: 车道中心线
    :param start_point: 自车当前位置
    :return: 相对坐标系下的轨迹点
    """
    trajectory = []
    cum_dist = compute_cum_dist(reference_line)
    i0, t0 = find_nearest_segment(start_point, reference_line)
    a, b = reference_line[i0], reference_line[i0 + 1]
    seg_len0 = math.hypot(b[0] - a[0], b[1] - a[1])
    s0 = cum_dist[i0] + (t0 * seg_len0)

    F_self_y = point2line_distance(start_point[0], start_point[1], [reference_line[i0], reference_line[i0+1]])
    another_index = i0 + 1
    pos_self_y = is_point_left_of_line(start_point[0], start_point[1], [reference_line[i0], reference_line[another_index]])
    F_self_y = F_self_y if pos_self_y else -F_self_y

    for i in range(len(frenet_trajectory)):
        point = frenet_trajectory[i]
        s_i = point[0]+s0
        d_i = point[1]+F_self_y
        project_point, theta = find_point_by_arc_length(reference_line, s_i)
        dx = -d_i * math.sin(theta)
        dy = d_i * math.cos(theta)
        x = project_point[0] + dx - start_point[0]
        y = project_point[1] + dy - start_point[1]
        trajectory.append([x, y])

    return np.array(trajectory)

def egoSE2_to_Frenet(
        ego_state: np.ndarray,
        reference_line: np.ndarray,
) -> np.ndarray:
    start = [ego_state[0], ego_state[1], 0]
    index, t = find_nearest_segment(start[:2], reference_line)
    F_self_y = point2line_distance(start[0], start[1], [reference_line[index], reference_line[index+1]])
    another_index = index + 1
    pos_self_y = is_point_left_of_line(start[0], start[1], [reference_line[index], reference_line[another_index]])
    F_self_y = F_self_y if pos_self_y else -F_self_y

    F_self_x = 0
    frenet_heading = np.arctan2(reference_line[another_index][1] - reference_line[index][1],
                                reference_line[another_index][0] - reference_line[index][0])
    F_self_heading = start[2] - frenet_heading
    F_self = np.array([F_self_x, F_self_y, F_self_heading])

    return F_self