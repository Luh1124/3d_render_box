
import os
import open3d as o3d
import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def point_cloud_intersection(points1, points2, threshold=1e-4):
    """
    计算两组点云坐标的交集
    
    参数:
    - points1: 第一组点云坐标 (N, 3)
    - points2: 第二组点云坐标 (M, 3)
    - threshold: 点匹配的距离阈值
    
    返回:
    - intersect_points: 交集点坐标
    """
    # 构造 KDTree
    tree1 = cKDTree(points1)
    
    # 查找交集点
    intersect_indices = []
    for point in points2:
        # 查找最近点
        dist, idx = tree1.query(point, k=1)
        if dist <= threshold:
            intersect_indices.append(idx)
    
    # 获取唯一的交集点
    intersect_indices = np.unique(intersect_indices)
    intersect_points = points1[intersect_indices]
    
    return intersect_points

def visualize_point_clouds(points1, points2, intersect_points, output_path='point_cloud_visualization.png'):  
    """  
    可视化点云，分三个子图显示  
    
    参数:  
    - points1: 第一组点云坐标  
    - points2: 第二组点云坐标  
    - intersect_points: 交集点坐标  
    - output_path: 输出图像路径  
    """  
    # 创建3D图，3个子图  
    fig = plt.figure(figsize=(18, 6))  

    # 第一个子图：第一组点云  
    ax1 = fig.add_subplot(131, projection='3d')  
    ax1.scatter(points1[:, 0], points1[:, 1], points1[:, 2],   
                c='red', alpha=0.1, s=0.1, label='Points Set 1')  
    ax1.set_title('Points Set 1')  
    ax1.set_xlabel('X')  
    ax1.set_ylabel('Y')  
    ax1.set_zlabel('Z')  
    ax1.legend()  

    # 第二个子图：第二组点云  
    ax2 = fig.add_subplot(132, projection='3d')  
    ax2.scatter(points2[:, 0], points2[:, 1], points2[:, 2],   
                c='green', alpha=0.1, s=0.1, label='Points Set 2')  
    ax2.set_title('Points Set 2')  
    ax2.set_xlabel('X')  
    ax2.set_ylabel('Y')  
    ax2.set_zlabel('Z')  
    ax2.legend()  

    # 第三个子图：交集点云  
    ax3 = fig.add_subplot(133, projection='3d')  
    ax3.scatter(intersect_points[:, 0], intersect_points[:, 1], intersect_points[:, 2],   
                c='blue', s=0.1, label='Intersection Points')  
    ax3.set_title('Intersection Points')  
    ax3.set_xlabel('X')  
    ax3.set_ylabel('Y')  
    ax3.set_zlabel('Z')  
    ax3.legend()  

    # 调整布局并保存  
    plt.tight_layout()  
    plt.savefig(output_path, dpi=300, bbox_inches='tight')  
    plt.close()  

def main():
    # 示例数据：两组随机3D点坐标
    voxel_ply = "/baai-cwm-1/baai_cwm_ml/cwm/hong.li/code/3dgen/data/zeroverse/3diclight_matsynth/even_render_6k_voxel/level_1/0a25c2a1ed9c9af847c3dd29a6e6caaf23050887bd9bbf9064faa943fa9ac157.ply"
    npz_path = "/baai-cwm-1/baai_cwm_ml/cwm/hong.li/code/3dgen/data/zeroverse/3diclight_matsynth/even_slat_6k/dino_surface_slats/level_1/0a25c2a1ed9c9af847c3dd29a6e6caaf23050887bd9bbf9064faa943fa9ac157.npz"


    # 载入 ply 点云
    pcd = o3d.io.read_point_cloud(voxel_ply) 
    points1 = ((np.asarray(pcd.points) + 0.5) * 64).astype(np.uint8)# (N,3)

    # 载入 np_data
    np_data = (np.load(npz_path)['coords']).astype(np.uint8)
    
    # 读取点云数据
    points1 = ((np.asarray(pcd.points) + 0.5) * 64).astype(np.uint8)  # 假设读取的点云数据为 (N, 3)
    points2 = np_data  # 假设读取的点云数据为 (M, 3)

    # 计算交集
    intersect_points = point_cloud_intersection(points1, points2, threshold=0.1)

    # 可视化点云
    visualize_point_clouds(points1, points2, intersect_points)

    # 打印结果
    print(f"第一组点云大小: {len(points1)}")
    print(f"第二组点云大小: {len(points2)}")
    print(f"交集点云大小: {len(intersect_points)}")

if __name__ == "__main__":
    main()
