from typing import *
import hashlib
import numpy as np
import math
import random


def get_file_hash(file: str) -> str:
    sha256 = hashlib.sha256()
    # Read the file from the path
    with open(file, "rb") as f:
        # Update the hash with the file content
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256.update(byte_block)
    return sha256.hexdigest()

# ===============LOW DISCREPANCY SEQUENCES================

# 质数表（用于 Halton 序列）  
PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71]  

def radical_inverse(base, n):
    val = 0
    inv_base = 1.0 / base
    inv_base_n = inv_base
    while n > 0:
        digit = n % base
        val += digit * inv_base_n
        n //= base
        inv_base_n *= inv_base
    return val

def halton_sequence(dim, n):
    return [radical_inverse(PRIMES[dim], n) for dim in range(dim)]

def hammersley_sequence(dim, n, num_samples):
    return [n / num_samples] + halton_sequence(dim - 1, n)

def sphere_hammersley_sequence(n, num_samples, offset=(0, 0)):
    u, v = hammersley_sequence(2, n, num_samples)
    u += offset[0] / num_samples
    v += offset[1]
    u = 2 * u if u < 0.25 else 2 / 3 * u + 1 / 3
    theta = np.arccos(1 - 2 * u) - np.pi / 2
    phi = v * 2 * np.pi
    return [phi, theta]

def equator_cameras_sequence(i, num_views, allow_theta_fixed=False, return_radians=True):  
    """  
    根据索引 i 生成单个相机的赤道位置，phi 均匀分布，theta 可选随机。  
    
    参数:  
    i (int): 当前相机的索引（从 0 开始）。  
    num_views (int): 相机位置的总数量。  
    allow_theta_random (bool): 是否允许 theta 在 -20 到 40 度之间随机分布。  
    return_radians (bool): 是否返回弧度值（默认返回角度值）。  
    
    返回:  
    tuple: 包含 (phi, theta) 的元组，单位为度或弧度。  
    """  
    phi = (360 / num_views) * i  # 均匀分布 phi  
    if allow_theta_fixed:  
        # theta = random.uniform(-20, 40)  # theta 在 -20 到 40 度之间随机  
        theta = (20 - (-20)) * i / num_views + (-20)
    else:  
        theta = 0  # 固定 theta 为 0 度，表示赤道平面  
    
    if return_radians:  
        # 将角度转换为弧度  
        phi = math.radians(phi)  
        theta = math.radians(theta)  
    
    return [phi, theta]

def cameras_sequence_fixed_26_views():
    theta = [-45, 0, 45]
    phi = list(range(0, 360, 45))

    camera_poses = []

    for t in theta:
        for p in phi:
            camera_poses.append((math.radians(p), math.radians(t)))
    
    # top and bottom
    camera_poses.append((math.radians(0), math.radians(90)))
    camera_poses.append((math.radians(0), math.radians(-90)))
    
    return camera_poses

def sphere_hammersley_sequence(n, num_samples, offset=(0, 0)):
    u, v = hammersley_sequence(2, n, num_samples)
    u += offset[0] / num_samples
    v += offset[1]
    u = 2 * u if u < 0.25 else 2 / 3 * u + 1 / 3
    theta = np.arccos(1 - 2 * u) - np.pi / 2
    phi = v * 2 * np.pi
    return [phi, theta]

def hemisphere_hammersley_sequence(n, num_samples, offset=(0, 0)):  
    # 生成Hammersley序列  
    u, v = hammersley_sequence(2, n, num_samples)  
    
    # 应用偏移  
    u += offset[0] / num_samples  
    v += offset[1]  
    
    # 保留非线性映射  
    u = 2 * u if u < 0.25 else 2 / 3 * u + 1 / 3  
    
    # 将u和v映射到[-1, 1]范围内  
    u = 2 * u - 1  # u从[-1, 1]  
    v = 2 * v - 1  # v从[-1, 1]  
    
    # 将u映射到[-π/2, π/2]范围内的极角theta  
    theta = np.arcsin(u)  # theta在[-π/2, π/2]范围内  
    
    # 将v映射到[-π/2, π/2]范围内的方位角phi  
    phi = np.arcsin(v)  # phi在[-π/2, π/2]范围内  
    return [phi, theta]  