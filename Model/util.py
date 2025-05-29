import math
from typing import Tuple
import torch
import numpy as np

X_MAX = 6.5
X_MIN = 0
Y_MAX = 1
Y_MIN = -1
X_MEAN =0
X_STD = 6.5
Y_MEAN =-1
Y_STD = 1

def regularize(x, MIN, MAX):
    batch_size, seq_len = x.shape
    indices = torch.arange(seq_len, device=x.device).float() + 1  # [16]
    min_i = MIN * indices  # [16]
    max_i = MAX * indices  # [16]
    
    # Expand min_i and max_i to match batch dimension
    min_i = min_i.unsqueeze(0).expand(batch_size, -1)  # [batch_size, 16]
    max_i = max_i.unsqueeze(0).expand(batch_size, -1)  # [batch_size, 16]
    
    # Normalize to [-1, 1] range
    result = (x - min_i)*2 / (max_i - min_i) -1
    return result.float()

def de_regularize(normalized, MIN, MAX):
    batch_size, seq_len = normalized.shape
    indices = torch.arange(seq_len, device=normalized.device).float() + 1  # [16]
    min_i = MIN * indices  # [16]
    max_i = MAX * indices  # [16]
    
    # Expand min_i and max_i to match batch dimension
    min_i = min_i.unsqueeze(0).expand(batch_size, -1)  # [batch_size, 16]
    max_i = max_i.unsqueeze(0).expand(batch_size, -1)  # [batch_size, 16]
    
    # Denormalize back to original range
    original = (normalized + 1) * (max_i - min_i) / 2 + min_i
    return original.float()

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
    # curve_lst = curve.tolist()
    # curve_lst = curve
    if type(curve) == np.ndarray:
        curve_lst = curve.tolist()
    else:
        curve_lst = curve
    min_distance = float('inf')
    for point in curve_lst:
        distance = point2point_distance(x, y, point[0], point[1])
        if distance < min_distance:
            min_distance = distance
            index = curve_lst.index(point)
            nearest_point = point

    another_point = curve_lst[index+1] if index+1 < len(curve_lst) else curve_lst[index-1]

    return point2line_distance(x,y,[nearest_point,another_point]), index

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

def segment_interpolation(a, b, t):
    """
    线段插值
    :param a: 线段起点
    :param b: 线段终点
    :param t: 插值参数
    :return: 插值点
    """
    a = np.array(a)
    b = np.array(b)
    return a + (b - a) * t

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

def find_point_by_arc_length(curve, s):
    """
    根据弧长在曲线上找到点
    :param start: 起点
    :param curve: 曲线
    :param s_i: 弧长
    :return: 点坐标
    """
    if len(curve) == 0:
        raise KeyError('curve is empty')
    if len(curve) < 2:
        return curve[0]
    
    cum_dist = compute_cum_dist(curve)
    total_length = cum_dist[-1]
    
    if s > total_length:
        a, b = curve[-2], curve[-1]
        dx = b[0] - a[0]
        dy = b[1] - a[1]
        theta = math.atan2(dy, dx)
        t = ((s - total_length) / math.hypot(dx, dy)) + 1
        return segment_interpolation(a, b, t), theta

    elif s < 0:
        a, b = curve[0], curve[1]
        dx = b[0] - a[0]
        dy = b[1] - a[1]
        theta = math.atan2(dy, dx)
        t = s / math.hypot(dx, dy)
        return segment_interpolation(a, b, t), theta
    
    else:
        for i in range(len(cum_dist) - 1):
            if cum_dist[i] <= s <= cum_dist[i + 1]:
                a, b = curve[i], curve[i + 1]
                dx = b[0] - a[0]
                dy = b[1] - a[1]
                theta = math.atan2(dy, dx)
                t = (s - cum_dist[i]) / math.hypot(dx, dy)
                return segment_interpolation(a, b, t), theta
    return None

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
    F_self_y = start[1]
   
    for i in range(len(trajectory)):
        point = trajectory[i]
        x = point[0]
        y = point[1]

        F_x = calculate_curve_distance(start, point, reference_line)
        idx, t = find_nearest_segment(point, reference_line)
        if x < -0.2 and abs(y)< 2:
            F_x = -F_x
        a = reference_line[idx]
        b = reference_line[idx + 1]
        F_y = point2line_distance(x, y, [a, b])
        if not is_point_left_of_line(x, y, [a, b]):
            F_y = -F_y
        F_y = F_y - F_self_y
        frenet_trajectory.append([float(F_x), float(F_y)])

    return np.array(frenet_trajectory)