import torch
import numpy as np
import os
import sys
import matplotlib
matplotlib.use('Agg')  # 使用无界面后端
import matplotlib.pyplot as plt
from pytorch3d.structures import Meshes
from pytorch3d.io import load_obj
from pytorch3d.ops import sample_points_from_meshes
import argparse
import open3d as o3d
import math
import cv2  # 用于生成视频

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def normalize_pointcloud(points):
    """
    只将点云移动到质心，不进行缩放
    """
    if torch.is_tensor(points):
        points = points.cpu().numpy()
    centroid = np.mean(points, axis=0)
    centered_points = points - centroid
    return torch.from_numpy(centered_points).float().to(device)
    
# 全局变量存储第一帧的归一化参数（质心和最大距离）
first_frame_params = None

def align_pointclouds(pcls, is_first_frame=False):
    """
    对齐多个点云到同一个参考系统：
      - 第一帧时计算并保存每个点云的质心和最大距离（用于归一化）
      - 后续帧使用第一帧保存的参数进行归一化
      - 对于所有来自 OBJ 的点云（即 dirs 中前5个），统一执行坐标交换和180°旋转
    """
    global first_frame_params
    aligned_pcls = []
    
    if is_first_frame:
        first_frame_params = []
        for i, pcl in enumerate(pcls):
            points = pcl.cpu().numpy()
            centroid = np.mean(points, axis=0)
            centered_points = points - centroid
            max_dist = np.max(np.sqrt(np.sum(centered_points ** 2, axis=1)))
            first_frame_params.append((centroid, max_dist))
            print(f"First frame params {i}: centroid = {centroid}, max_dist = {max_dist}")
    
    if first_frame_params is None:
        raise ValueError("first_frame_params not initialized! Process the first frame first.")
    
    for i, pcl in enumerate(pcls):
        points = pcl.cpu().numpy()
        centroid, max_dist = first_frame_params[i]
        centered_points = points - centroid
        normalized_points = centered_points / max_dist
        
        # 对于所有 OBJ 数据（索引 < 5）统一做坐标交换与180°旋转
        if i < 5:
            swapped_points = normalized_points.copy()
            swapped_points[:, 0] = normalized_points[:, 1]
            swapped_points[:, 1] = normalized_points[:, 0]
            theta = np.pi
            rotation_matrix = np.array([
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta),  np.cos(theta), 0],
                [0, 0, 1]
            ])
            normalized_points = np.dot(swapped_points, rotation_matrix.T)
        
        norm_pcl = torch.from_numpy(normalized_points).float().to(device)
        # 缩放并平移到目标位置
        norm_pcl = norm_pcl * 2
        norm_pcl = norm_pcl + torch.tensor([2.5, 1.75, 0.0]).to(device)
        aligned_pcls.append(norm_pcl)
    
    return aligned_pcls

# 此处移除 compute_metrics 和 compute_icp_metrics

def convert_objs_to_pcls(obj_dir, out_dir, num_points=5000):
    """将 OBJ 文件转换为点云（npy 格式）"""
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    obj_files = sorted([f for f in os.listdir(obj_dir) if f.endswith('.obj')])
    print(f"Found {len(obj_files)} OBJ files in directory {obj_dir}")
    
    for obj_file in obj_files:
        try:
            if '_' in obj_file:
                frame_num = int(obj_file.split('_')[0])
            else:
                frame_num = int(''.join(filter(str.isdigit, obj_file[:4])))
            
            verts, faces, aux = load_obj(os.path.join(obj_dir, obj_file))
            # 将 y 轴取反（若原代码中有此处理）
            verts[:, 1] = -verts[:, 1]
            
            mesh = Meshes(verts=[verts], faces=[faces.verts_idx])
            sampled_points = sample_points_from_meshes(mesh, num_points)
            
            out_file = os.path.join(out_dir, f'{frame_num}.npy')
            np.save(out_file, sampled_points.squeeze(0).cpu().numpy())
            print(f"Processed {obj_file} -> {out_file}")
        except Exception as e:
            print(f"Error processing {obj_file}: {str(e)}")
            continue

def sample_pcd(pcd, num_points):
    """从 PCD 点云中采样指定数量的点"""
    points = np.asarray(pcd.points)
    if len(points) == num_points:
        return points
    elif len(points) > num_points:
        indices = np.random.choice(len(points), num_points, replace=False)
        return points[indices]
    else:
        indices = np.random.choice(len(points), num_points, replace=True)
        return points[indices]

def load_pointcloud(file_path, num_points=None):
    """加载点云（支持 npy, txt, pcd 格式）"""
    try:
        if file_path.endswith('.npy'):
            points = torch.from_numpy(np.load(file_path)).float().to(device)
        elif file_path.endswith('.txt'):
            points = np.loadtxt(file_path)
            if num_points is not None:
                if points.shape[0] > num_points:
                    indices = np.random.choice(points.shape[0], num_points, replace=False)
                    points = points[indices]
                elif points.shape[0] < num_points:
                    indices = np.random.choice(points.shape[0], num_points, replace=True)
                    points = points[indices]
            points = torch.from_numpy(points).float().to(device)
        elif file_path.endswith('.pcd'):
            pcd = o3d.io.read_point_cloud(file_path)
            if num_points is not None:
                points_np = sample_pcd(pcd, num_points)
                points = torch.from_numpy(points_np).float().to(device)
            else:
                points = torch.from_numpy(np.asarray(pcd.points)).float().to(device)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
        print(f"Loaded {file_path}: {points.shape[0]} points")
        return points
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None

def plot_single_view(pcl, label, color):
    """
    绘制单个视图，返回图像（numpy 数组）
    """
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection='3d')
    x, y, z = pcl.cpu().numpy().T
    ax.scatter(x, y, z, s=0.2, c=[color])
    ax.set_title(label)
    ax.set_xlim([0, 5])
    ax.set_ylim([0, 5])
    ax.set_zlim([-1, 5])
    ax.view_init(10, -70)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    plt.tight_layout()
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return img

def plot_overlaid_view(pcls, labels, colors):
    """
    绘制叠加视图，将所有点云绘制在同一图中，返回图像
    """
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection='3d')
    for pcl, label, color in zip(pcls, labels, colors):
        x, y, z = pcl.cpu().numpy().T
        ax.scatter(x, y, z, s=0.2, c=[color], label=label)
    ax.set_title("Overlaid")
    ax.set_xlim([0, 5])
    ax.set_ylim([0, 5])
    ax.set_zlim([-1, 5])
    ax.view_init(10, -70)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    plt.tight_layout()
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return img

def run_visualization_videos(dirs, out_dir, total_steps, num_points, frame_rate=25):
    """
    整体流程：
      - 对每一步加载各来源点云（共 5 个 OBJ、3 个 TXT、1 个 PCD），并对齐
      - 为每个视图分别生成单帧图像（不计算指标），同时生成一个叠加视图帧
      - 将这 10 个小图像（9 单独视图 + 1 叠加视图）排列成 2 行 5 列，合成一帧，然后生成视频
      - 最后一帧额外重复 3 秒
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    # 自定义标签与颜色，顺序与 dirs 对应（5 个 OBJ，3 个 TXT，1 个 PCD，共 9 个视图）
    custom_labels = [
        'Diffcloud(lifting)',
        'PINN on Diffsim(stretching)',
        'Physnet on Diffsim(lifting)',
        'Physnet on Diffsim(wind)',
        'Diffcloud(wind)',
        'Diffcp(lifting)',
        'PINN on Diffcp(stretching)',
        'Physnet on Diffcp(lifting)',
        'Real Point Cloud'
    ]
    custom_colors = [
        (0.6, 0.0, 0.0, 1.0),   # OBJ1
        (0.0, 0.6, 0.0, 1.0),   # OBJ2
        (0.0, 0.0, 0.6, 1.0),   # OBJ3
        (0.8, 0.5, 0.0, 1.0),   # OBJ4
        (0.6, 0.0, 0.6, 1.0),   # OBJ5
        (0.0, 0.6, 0.6, 1.0),   # TXT1
        (0.5, 0.5, 0.0, 1.0),   # TXT2
        (0.7, 0.7, 0.0, 1.0),   # TXT3（指定颜色）
        (0.0, 0.0, 0.0, 1.0)    # PCD（真实点云）
    ]
    
    # 本次处理将生成 10 个小图像（9 个单独视图 + 1 叠加视图），并排列成 2 行 5 列
    composite_frames = []
    
    for step in range(1, total_steps + 1):
        print(f"\nProcessing step {step}")
        pcls = []
        for i, (dir_path, fmt) in enumerate(dirs):
            file_path = os.path.join(dir_path, f'{step}.{fmt}')
            pcl = load_pointcloud(file_path, num_points if fmt in ['txt', 'pcd'] else None)
            if pcl is None:
                print(f"Missing point cloud for step {step} in {dir_path}")
                return
            pcls.append(pcl)
        
        # 对齐各视图（第一帧保存归一化参数）
        aligned_pcls = align_pointclouds(pcls, is_first_frame=(step == 1))
        
        # 为每个视图分别生成单帧图像
        step_images = []
        for i in range(len(dirs)):
            img = plot_single_view(aligned_pcls[i], custom_labels[i], custom_colors[i])
            step_images.append(img)
        # 生成叠加视图帧
        overlaid_img = plot_overlaid_view(aligned_pcls, custom_labels, custom_colors)
        step_images.append(overlaid_img)  # 此时共 10 张图
        
        # 将 10 张图排列成 2 行 5 列（假设所有图像大小一致）
        row1 = np.hstack(step_images[0:5])
        row2 = np.hstack(step_images[5:10])
        composite = np.vstack([row1, row2])
        composite_frames.append(composite)
    
    # 将 composite_frames 合成视频，最后一帧额外重复 3 秒
    if not composite_frames:
        print("No frames generated.")
        return
    height, width, _ = composite_frames[0].shape
    video_path = os.path.join(out_dir, "composite_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_path, fourcc, frame_rate, (width, height))
    
    for frame in composite_frames:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video_writer.write(frame_bgr)
    
    last_frame = composite_frames[-1]
    last_frame_bgr = cv2.cvtColor(last_frame, cv2.COLOR_RGB2BGR)
    for _ in range(frame_rate * 3):
        video_writer.write(last_frame_bgr)
    video_writer.release()
    print(f"Composite video saved to {video_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # TXT 目录
    parser.add_argument('-t1', '--txt_dir1', type=str, 
                        default='/home/david/Downloads/diffcp-main/src/greyshakediffcp', 
                        help='First directory containing TXT point clouds')
    parser.add_argument('-t2', '--txt_dir2', type=str, 
                        default='/home/david/Downloads/diffcp-main/src/greyshakepinn', 
                        help='Second directory containing TXT point clouds')
    parser.add_argument('-t3', '--txt_dir3', type=str, 
                        default='/home/david/Downloads/diffcp-main/src/greyshakepinn', 
                        help='Third directory containing TXT point clouds')
    # PCD 目录
    parser.add_argument('-p', '--pcd_dir', type=str, 
                        default='/home/david/Downloads/david_shake/grey/ru4', 
                        help='Directory containing PCD point clouds')
    # OBJ 目录：共 5 个 OBJ trajectory
    parser.add_argument('-o1', '--obj_dir1', type=str,
                        default='/home/david/Downloads/kentuen/greydiffcloudliftingshake',
                        help='First OBJ trajectory directory')
    parser.add_argument('-o2', '--obj_dir2', type=str,
                        default='/home/david/Downloads/kentuen/greypinnstretchingshake',
                        help='Second OBJ trajectory directory')
    parser.add_argument('-o3', '--obj_dir3', type=str,
                        default='/home/david/Downloads/kentuen/greyphysnetliftshake',
                        help='Third OBJ trajectory directory')
    parser.add_argument('-o4', '--obj_dir4', type=str,
                        default='/home/david/Downloads/kentuen/greyphysnetwindshake',
                        help='Fourth OBJ trajectory directory')
    parser.add_argument('-o5', '--obj_dir5', type=str,
                        default='/home/david/Downloads/kentuen/greydiffcloudwindshake',
                        help='Fifth OBJ trajectory directory')
    parser.add_argument('-s', '--steps', type=int, default=134, 
                        help='Total simulation steps')
    parser.add_argument('--out_dir', type=str, default='allshakepink2_videos', 
                        help='Output directory')
    parser.add_argument('--points', type=int, default=10000, 
                        help='Number of points to sample from meshes (also used for TXT/PCD sampling)')
    
    args = parser.parse_args()
    
    # 将所有 5 个 OBJ 目录转换为点云
    obj_pcl_dir1 = os.path.join(args.out_dir, 'obj_pcls1')
    obj_pcl_dir2 = os.path.join(args.out_dir, 'obj_pcls2')
    obj_pcl_dir3 = os.path.join(args.out_dir, 'obj_pcls3')
    obj_pcl_dir4 = os.path.join(args.out_dir, 'obj_pcls4')
    obj_pcl_dir5 = os.path.join(args.out_dir, 'obj_pcls5')
    
    convert_objs_to_pcls(args.obj_dir1, obj_pcl_dir1, args.points)
    convert_objs_to_pcls(args.obj_dir2, obj_pcl_dir2, args.points)
    convert_objs_to_pcls(args.obj_dir3, obj_pcl_dir3, args.points)
    convert_objs_to_pcls(args.obj_dir4, obj_pcl_dir4, args.points)
    convert_objs_to_pcls(args.obj_dir5, obj_pcl_dir5, args.points)
    
    # 更新 dirs 顺序：前 5 个为 OBJ trajectory，接下来 3 个 TXT，最后 1 个真实 PCD
    dirs = [
        (obj_pcl_dir1, 'npy'),
        (obj_pcl_dir2, 'npy'),
        (obj_pcl_dir3, 'npy'),
        (obj_pcl_dir4, 'npy'),
        (obj_pcl_dir5, 'npy'),
        (args.txt_dir1, 'txt'),
        (args.txt_dir2, 'txt'),
        (args.txt_dir3, 'txt'),
        (args.pcd_dir, 'pcd')
    ]
    
    run_visualization_videos(dirs, args.out_dir, args.steps, args.points, frame_rate=25)
    print("Visualization completed.")
