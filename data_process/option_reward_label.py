import math
from typing import List, Dict, Tuple, Set
import numpy as np
from data_process.utils import draw_rectangle,get_nearest_line,point2point_distance,take_first_n_step,extend_list,calculate_curve_distance, get_nearest_point, point2curve_distance,point2curve_nearest_point,all_zeros

OPTIONS = ['stop','follow','left','right','left_U','right_U','back'] #,'invalid'
LANE_BIAS = 1.0
STOP_BIAS = 0.2
BACK_BIAS = -2

REACTION_TIME = 2   # 反应时间
A_TOLERANCE = 6 # 加速度容忍度
MAX_ACCELERATION = 5

def get_option(F_trajectory) -> Dict:
    """
    Add option to the data.
    :param data: Data to be added.
    :param ego_agent_past_trajectory: Ego agent past trajectory.
    :return: last_option, option.
    """
    all_trajectory = F_trajectory[-18:]
    
    all_x = all_trajectory[:, 0]
    all_y = all_trajectory[:, 1]

    lastOption_forward = False
    current_forward = False
    lastOption_back = False
    current_back = False
    nextOption_forward = False
    nextOption_back = False
    lastOption_x = all_x[:4]
    current_x = all_x[1:5]
    nextOption_x = all_x[2:6]

    for i in range(4):
        for j in range(i):
            lastOption_delta = lastOption_x[i] - lastOption_x[j]
            current_delta = current_x[i] - current_x[j]
            nextOption_delta = nextOption_x[i] - nextOption_x[j]
            if lastOption_delta > STOP_BIAS:
                lastOption_forward = True
            elif lastOption_delta < BACK_BIAS:
                lastOption_back = True
            if current_delta > STOP_BIAS:
                current_forward = True
            elif current_delta < BACK_BIAS:
                current_back = True
            if nextOption_delta > STOP_BIAS:
                nextOption_forward = True
            elif nextOption_delta < BACK_BIAS:
                nextOption_back = True
    if lastOption_back:
        last_option = 'back'
    elif lastOption_forward:
        last_option = 'forward'
    else:
        last_option = 'stop'

    if current_back:
        option = 'back'
    elif current_forward:
        option = 'forward'
    else:
        option = 'stop'
    
    if nextOption_back:
        next_option = 'back'
    elif nextOption_forward:
        next_option = 'forward'
    else:
        next_option = 'stop'


    lastOption_y = all_y[:14]
    current_y = all_y[1:15]
    nextOption_y = all_y[2:16]

    lastOptin_has_left = False
    lastOptin_has_right = False
    lastOptin_left_first = False
    has_left = False
    has_right = False
    left_first = False
    nextOption_has_left = False
    nextOption_has_right = False
    nextOption_left_first = False


    for i in range(14):
        for j in range(i):
            lastOption_delta = lastOption_y[i] - lastOption_y[j]
            current_delta = current_y[i] - current_y[j]
            nextOption_delta = nextOption_y[i] - nextOption_y[j]
            if lastOption_delta > LANE_BIAS:
                lastOptin_has_left = True
                if not lastOptin_has_right:
                    lastOptin_left_first = True
            elif lastOption_delta < -LANE_BIAS:
                lastOptin_has_right = True
            if current_delta > LANE_BIAS:
                has_left = True
                if not has_right:
                    left_first = True
            elif current_delta < -LANE_BIAS:
                has_right = True
            if nextOption_delta > LANE_BIAS:
                nextOption_has_left = True
                if not nextOption_has_right:
                    nextOption_left_first = True
            elif nextOption_delta < -LANE_BIAS:
                nextOption_has_right = True
    
    if option == 'forward':
        if has_left and not has_right:
            option = 'left'
        elif has_right and not has_left:
            option = 'right'
        elif has_left and has_right:
            option = 'left_U' if left_first else 'right_U'
        else:
            option = 'follow'

    if last_option == 'forward':
        if lastOptin_has_left and not lastOptin_has_right:
            last_option = 'left'
        elif lastOptin_has_right and not lastOptin_has_left:
            last_option = 'right'
        elif lastOptin_has_left and lastOptin_has_right:
            last_option = 'left_U' if lastOptin_left_first else 'right_U'
        else:
            last_option = 'follow'

    if next_option == 'forward':
        if nextOption_has_left and not nextOption_has_right:
            next_option = 'left'
        elif nextOption_has_right and not nextOption_has_left:
            next_option = 'right'
        elif nextOption_has_left and nextOption_has_right:
            next_option = 'left_U' if nextOption_left_first else 'right_U'
        else:
            next_option = 'follow'

    return {
        'last_option': OPTIONS.index(last_option),
        'option': OPTIONS.index(option),
        'next_option': OPTIONS.index(next_option),
    }

def estimate_velocity_acceleration(old_ego: List, ego: List, next_ego: List, timestep=0.5) -> List:
    """
    Estimate the velocity and acceleration of the ego agent.
    :param old_ego: Old ego agent state.
    :param ego: Ego agent state.
    :param next_ego: Next ego agent state.
    :return: Estimated velocity and acceleration.
    """
    ego_vx = (next_ego[0] - old_ego[0]) / (timestep * 2)
    ego_vy = (next_ego[1] - old_ego[1]) / (timestep * 2)
    ego_ax = (ego_vx - old_ego[3]) / timestep
    ego_ay = (ego_vy - old_ego[4]) / timestep
    return [ego_vx, ego_vy, ego_ax, ego_ay]


def follow_reward(data):
    # TODO: Improve the reward design
    reward_dict = {}
    lanes = data['lanes'][:,:,:2].tolist()
    option = data['option']
    reference_line = data['reference_line'].tolist()
    lanes_speed_limit = data['lanes_speed_limit']
    lanes_has_speed_limit = data['lanes_has_speed_limit']
    agents = data['next_frame_agents']
    ego = data['ego_agent_future'][0]
    old_ego = data['ego_current_state']
    next_ego = data['ego_agent_future'][1]
    ego_x,ego_y,ego_h = ego[0],ego[1],ego[2]
    v_a_estimate = estimate_velocity_acceleration(old_ego, ego, next_ego)
    ego_vx,ego_vy = v_a_estimate[0],v_a_estimate[1]
    ego_ax,ego_ay = v_a_estimate[2],v_a_estimate[3]
    _, lane_index = get_nearest_line(ego_x, ego_y, lanes, 1)
    current_lane = reference_line
    lane_index = lane_index[0]
    refer_points = get_nearest_point(ego_x, ego_y, reference_line, 2)

    ego_v = math.hypot(ego_vx, ego_vy)
    ego_a = math.hypot(ego_ax, ego_ay)

    if option != 'follow':
        point_1 = refer_points[0]
        point_2 = refer_points[1]
        if point_1[0] > point_2[0]:
            point_1, point_2 = point_2, point_1
        theta = math.atan2(point_2[1] - point_1[1], point_2[0] - point_1[0])
        ego_v_change = ego_v * math.sin(theta)
        ego_a_change = ego_a * math.sin(theta)
        ego_v = ego_v * math.cos(theta)
        ego_a = ego_a * math.cos(theta)

    speed_limit = lanes_speed_limit[lane_index][0] if lanes_has_speed_limit[lane_index] else 0
    if speed_limit > 0 and ego_vx > speed_limit:
        reward_dict['over_speed'] = -(ego_vx - speed_limit) * 0.5
    if ego_a > A_TOLERANCE:
        reward_dict['over_acceleration'] = -(ego_a - A_TOLERANCE) * 0.5
    lane_bias,_ = point2curve_distance(ego_x, ego_y, current_lane)
    if option == 'follow':
        lane_bias_weight = 1.0
    else:
        lane_bias_weight = 0.6
    reward_dict['lane_bias'] = -lane_bias * lane_bias_weight

    on_route_agents_front = []
    on_route_agents_back = []
    for agent in agents:
        agent_x,agent_y = agent[0],agent[1]
        distance,_ = point2curve_nearest_point(agent_x, agent_y, current_lane)
        if distance < 2:
            if agent_x > ego_x:
                on_route_agents_front.append(agent)
            else:
                on_route_agents_back.append(agent)
            
    on_route_agents_front.sort(key=lambda x: math.hypot(x[0]-ego_x, x[1]-ego_y))
    on_route_agents_back.sort(key=lambda x: math.hypot(x[0]-ego_x, x[1]-ego_y))
    if len(on_route_agents_front) > 0:
        front_agent = on_route_agents_front[0]
        distance = calculate_curve_distance([ego_x, ego_y], [front_agent[0], front_agent[1]], reference_line)
        
        front_agent_v = math.hypot(front_agent[3], front_agent[4])
        if distance < ego_v * 1.5 and abs(front_agent_v - ego_v) < 1:
            reward_dict['front_agent_distance'] = (distance - ego_v * 1.5) * 0.5
        elif distance > ego_v * 4 and front_agent_v > ego_v:
            reward_dict['front_agent_distance'] = -(front_agent_v - ego_v) * 0.1
    
    if len(on_route_agents_back) > 0:
        back_agent = on_route_agents_back[0]
        distance = calculate_curve_distance([ego_x, ego_y], [back_agent[0], back_agent[1]], reference_line)
        back_agent_v = math.hypot(back_agent[3], back_agent[4])

        if ego_ax < 0:
            break_distance = back_agent_v ** 2 / (2 * A_TOLERANCE) + ego_v * REACTION_TIME
            ego_break_distance = ego_v ** 2 / (2 * ego_ax)
            end_distance = distance - abs(break_distance) + abs(ego_break_distance)
            if end_distance < back_agent[5]/2 + 2.0:
                reward_dict['back_agent_distance'] = -(end_distance - back_agent[5]/2 - 2.0)

    reward = sum(reward_dict.values())

    return reward, reward_dict

def stop_reward(data):
    lanes = data['lanes'][:,:,:2].tolist()
    reference_line = data['reference_line'].tolist()
    agents = data['next_frame_agents']
    ego = data['ego_agent_future'][0]
    ego_x,ego_y,ego_h = ego[0],ego[1],ego[2]

    current_lane_state, lane_index = get_nearest_line(ego_x, ego_y, lanes, 1)
    current_lane = reference_line
    lane_index = lane_index[0]     
    current_lane_state = current_lane_state[0]
    light_state = current_lane_state[-1][-4:]
    stop_signal = False
    if light_state[1] == 1 or light_state[2] == 1 or light_state[3] == 1:
        stop_signal = True
    
    on_route_agents_front = []
    for agent in agents:
        agent_x,agent_y = agent[0],agent[1]
        distance,_ = point2curve_nearest_point(agent_x, agent_y, current_lane)
        if distance < 2:
            if agent_x > ego_x:
                on_route_agents_front.append(agent)
            
    on_route_agents_front.sort(key=lambda x: math.hypot(x[0]-ego_x, x[1]-ego_y))

    if len(on_route_agents_front) > 0:
        front_agent = on_route_agents_front[0]
        distance = calculate_curve_distance([ego_x, ego_y], [front_agent[0], front_agent[1]], reference_line)
        free_distance = distance - front_agent[5]/2 - 2.0
    else:
        free_distance = 100

    if not stop_signal and free_distance > 10:
        reward = -2
    else:
        reward = -0.5
    
    return reward, {'stop':reward}

def back_reward(data):
    reference_line = data['reference_line'].tolist()
    ego = data['ego_agent_future'][0]
    ego_x,ego_y,ego_h = ego[0],ego[1],ego[2]

    route_distance,_ = point2curve_distance(ego_x, ego_y, reference_line)

    if route_distance > 3: # not on route
        reward = -0.5
    else:
        reward = -2
    return reward, {'back':reward}

def get_action_reward(data):
    """
    Get the action reward.
    :param data: Data to be added.
    :return: Action reward.
    """
    option = data['option']
    if option == 'stop':
        reward, reward_dict = stop_reward(data)
    elif option == 'back':
        reward, reward_dict = back_reward(data)
    elif option == 'invalid':
        reward = -12
        reward_dict = {'invalid':reward}
    else:
        reward, reward_dict = follow_reward(data)
    
    return reward+1, reward_dict

def get_option_reward(option, reward, last_option,trajectory, route_lanes):
    """
    Get the option reward.
    :param data: Data to be added.
    :return: Option reward.
    """
    if option == 'stop':
        if reward > 0:
            option_reward = 0.8
        else:
            option_reward = -1
    elif option == 'back':
        if reward > 0:
            option_reward = 0.5
        else:
            option_reward = -1
    elif option == 'invalid':
        option_reward = -1
    elif option == 'follow':
        option_reward = 0.9
    else:
        option_reward = 1

    if last_option == option:
        option_reward += 0.1

    min_bias_from_route = 0
    if not all_zeros(route_lanes):
        min_bias_from_route = float('inf')
        for lane in route_lanes:
            end_point = trajectory[-1]
            distance,_ = point2curve_nearest_point(end_point[0], end_point[1], lane)
            lane_begin_x = lane[0][0]
            lane_end_x = lane[-1][0]
            if not (lane_begin_x < end_point[0] < lane_end_x):
                continue
            if distance < min_bias_from_route:
                min_bias_from_route = distance
    if min_bias_from_route > 8 and min_bias_from_route < 1000:
        option_reward -= 0.6

    return option_reward
    
