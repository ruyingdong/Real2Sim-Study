import torch
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from pytorch3d.structures import Meshes
from pytorch3d.io import load_obj
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import chamfer_distance
import argparse
import open3d as o3d
import math

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
    
# 全局变量存储第一帧的参数（质心和最大距离）
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
        raise ValueError("first_frame_params not initialized! Must process first frame first.")
    
    for i, pcl in enumerate(pcls):
        points = pcl.cpu().numpy()
        centroid, max_dist = first_frame_params[i]
        centered_points = points - centroid
        normalized_points = centered_points / max_dist
        
        # 对于所有 OBJ 数据（索引 < 5）统一做坐标交换与旋转
        if i < 5:
            swapped_points = normalized_points.copy()
            # 交换 x 与 y
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

def compute_metrics(src_pcl, tgt_pcl):
    """
    计算 Chamfer Distance 和 Hausdorff Distance
    """
    if src_pcl.dim() == 2:
        src_pcl = src_pcl.unsqueeze(0)
    if tgt_pcl.dim() == 2:
        tgt_pcl = tgt_pcl.unsqueeze(0)
    chamfer_dist, _ = chamfer_distance(src_pcl, tgt_pcl)
    
    src_np = src_pcl.squeeze(0).cpu().numpy()
    tgt_np = tgt_pcl.squeeze(0).cpu().numpy()
    
    src_o3d = o3d.geometry.PointCloud()
    src_o3d.points = o3d.utility.Vector3dVector(src_np)
    tgt_o3d = o3d.geometry.PointCloud()
    tgt_o3d.points = o3d.utility.Vector3dVector(tgt_np)
    
    dist_src_tgt = np.asarray(src_o3d.compute_point_cloud_distance(tgt_o3d))
    dist_tgt_src = np.asarray(tgt_o3d.compute_point_cloud_distance(src_o3d))
    hausdorff_dist = max(np.max(dist_src_tgt), np.max(dist_tgt_src))
    
    return {
        'chamfer': chamfer_dist.item(),
        'hausdorff': hausdorff_dist
    }

def compute_icp_metrics(src_pcl, tgt_pcl, threshold=0.2):
    """
    计算 ICP 的评价指标：fitness 和 inlier_rmse
    src_pcl, tgt_pcl 为 torch tensor，形状为 (N, 3)
    threshold 为判断内点的距离阈值
    """
    src_np = src_pcl.cpu().numpy()
    tgt_np = tgt_pcl.cpu().numpy()
    
    src_o3d = o3d.geometry.PointCloud()
    src_o3d.points = o3d.utility.Vector3dVector(src_np)
    tgt_o3d = o3d.geometry.PointCloud()
    tgt_o3d.points = o3d.utility.Vector3dVector(tgt_np)
    
    trans_init = np.eye(4)
    evaluation = o3d.pipelines.registration.evaluate_registration(
        src_o3d, tgt_o3d, threshold, trans_init)
    
    return {
        'fitness': evaluation.fitness,
        'inlier_rmse': evaluation.inlier_rmse
    }

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

def plot_all_views(pcls, labels, metrics, title=""):
    """
    采用网格布局，将各个单独视图和一个叠加视图均以相同子图大小绘制。
    这里计算总子图数 = 单独视图数 + 1（叠加图），并采用固定网格（每行4个子图）
    """
    num_individual = len(pcls)
    num_total = num_individual + 1
    num_cols = 4
    num_rows = math.ceil(num_total / num_cols)
    
    # 定义深色调颜色，每个视图一个颜色
    custom_colors = [
        (0.6, 0.0, 0.0, 1.0),   # dark red
        (0.0, 0.6, 0.0, 1.0),   # dark green
        (0.0, 0.0, 0.6, 1.0),   # dark blue
        (0.8, 0.5, 0.0, 1.0),   # dark orange
        (0.6, 0.0, 0.6, 1.0),   # dark magenta
        (1.0, 1.0, 0.0, 1.0),   # Diffcp(lifting): yellow
        (0.7, 0.7, 0.0, 1.0),  # dark olive
        (0.7, 0.7, 0.0, 1.0)    # black
    ]
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(4*num_cols, 4*num_rows), subplot_kw={'projection': '3d'})
    axes = np.array(axes).reshape(-1)
    
    # 绘制每个单独视图
    for idx in range(num_individual):
        ax = axes[idx]
        pcl = pcls[idx]
        x, y, z = pcl.cpu().numpy().T
        ax.scatter(x, y, z, s=0.2, c=[custom_colors[idx % len(custom_colors)]], label=labels[idx])
        ax.set_title(labels[idx])
        ax.set_xlim([0, 5])
        ax.set_ylim([0, 5])
        ax.set_zlim([-1, 5])
        ax.view_init(10, -70)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        if metrics is not None and idx < num_individual - 1:
            metrics_text = f"Chamfer: {metrics[idx]['chamfer']:.6f}\nHausdorff: {metrics[idx]['hausdorff']:.6f}"
            ax.text2D(0.05, -0.15, metrics_text, transform=ax.transAxes, fontsize=8)
    
    # 绘制叠加视图，将所有点云绘制在一起
    ax_overlay = axes[num_individual]
    for idx in range(num_individual):
        pcl = pcls[idx]
        x, y, z = pcl.cpu().numpy().T
        ax_overlay.scatter(x, y, z, s=0.2, c=[custom_colors[idx % len(custom_colors)]], label=labels[idx])
    ax_overlay.set_title("Overlaid")
    ax_overlay.set_xlim([0, 5])
    ax_overlay.set_ylim([0, 5])
    ax_overlay.set_zlim([-1, 5])
    ax_overlay.view_init(10, -70)
    ax_overlay.set_xticks([])
    ax_overlay.set_yticks([])
    ax_overlay.set_zticks([])
    
    # 隐藏多余子图
    for j in range(num_total, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    plt.savefig(title, bbox_inches='tight', pad_inches=0.5)
    plt.close(fig)

def run_visualization(dirs, out_dir, total_steps, num_points):
    """
    整体流程：
      - 加载各步点云（包括5个 OBJ trajectory、2个 TXT 和 1个真实 PCD）
      - 对齐点云（使用第一帧的参数归一化，对所有 OBJ 数据均执行坐标交换与180°旋转）
      - 计算 Chamfer、Hausdorff 以及 ICP 指标（均与最后一个点云比较）
      - 绘制每步图像（包括叠加视图），并保存所有指标曲线及文本文件（含每步数据和平均值）
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    # 自定义标签，与 dirs 顺序一一对应
    custom_labels = [
        'Diffcloud(lifting)',
        'PINN on Diffsim(stretching)',
        'Physnet on Diffsim(lifting)',
        'Physnet on Diffsim(wind)',
        'Diffcloud(wind)',
        'Diffcp(lifting)',
        'PINN on Diffcp(stretching)',
        'Real Point Cloud'
    ]
    
    # 仅对前7个（OBJ Trajectory 1~5以及 TXT 1、TXT 2）与真实点云进行指标计算
    all_metrics = {label: {'chamfer': [], 'hausdorff': []} for label in custom_labels[:-1]}
    
    metrics_steps_text = ""
    
    for step in range(1, total_steps + 1):
        print(f"\nProcessing step {step}")
        pcls = []
        for i, (dir_path, format_type) in enumerate(dirs):
            file_path = os.path.join(dir_path, f'{step}.{format_type}')
            # 对于 pcd 文件传入采样点数，其它格式直接加载
            pcl = load_pointcloud(file_path, num_points if format_type in ['txt', 'pcd'] else None)
            if pcl is None:
                print(f"Missing point cloud for step {step}")
                return
            pcls.append(pcl)
        
        aligned_pcls = align_pointclouds(pcls, is_first_frame=(step == 1))
        
        step_metrics = []
        step_text = f"Step {step}:\n"
        for i in range(len(aligned_pcls) - 1):
            metrics = compute_metrics(aligned_pcls[i], aligned_pcls[-1])
            step_metrics.append(metrics)
            label = custom_labels[i]
            for metric_name, value in metrics.items():
                all_metrics[label][metric_name].append(value)
            step_text += (f"{custom_labels[i]} - Chamfer: {metrics['chamfer']:.6f}, Hausdorff: {metrics['hausdorff']:.6f}\n")
            print(f"\nMetrics for {custom_labels[i]}:")
            print(f"  Chamfer Distance: {metrics['chamfer']:.6f}")
            print(f"  Hausdorff Distance: {metrics['hausdorff']:.6f}")
        metrics_steps_text += step_text + "\n"
        
        plot_all_views(
            pcls=aligned_pcls,
            labels=custom_labels,
            metrics=step_metrics,
            title=os.path.join(out_dir, f'step{step:03d}.png')
        )
    
    with open(os.path.join(out_dir, "metrics_steps.txt"), 'w') as f:
        f.write(metrics_steps_text)
    
    # 绘制并保存指标曲线及文本
    avg_metrics_text = ""
    for metric_name in ['chamfer', 'hausdorff']:
        plt.figure(figsize=(12, 6))
        for label in custom_labels[:-1]:
            plt.plot(range(1, total_steps + 1), all_metrics[label][metric_name], label=label)
        plt.xlabel('Step')
        plt.ylabel(f'{metric_name.capitalize()} Distance')
        plt.title(f'{metric_name.capitalize()} Distance over Steps')
        plt.legend(fontsize=8)
        plt.grid(True)
        plt.savefig(os.path.join(out_dir, f'{metric_name}_distances.png'))
        plt.close()
        
        with open(os.path.join(out_dir, f'{metric_name}_distances.txt'), 'w') as f:
            for step in range(total_steps):
                f.write(f"Step {step + 1}:\n")
                for label in custom_labels[:-1]:
                    f.write(f"{label}: {all_metrics[label][metric_name][step]:.6f}\n")
                f.write("\n")
            f.write(f"\nAverage {metric_name.capitalize()} Distance over {total_steps} steps:\n")
            for label in custom_labels[:-1]:
                avg_val = np.mean(all_metrics[label][metric_name])
                f.write(f"{label}: {avg_val:.6f}\n")
                avg_metrics_text += f"{label} {metric_name} avg: {avg_val:.6f}\n"
    
    with open(os.path.join(out_dir, "metrics_average.txt"), 'w') as f:
        f.write("Average Metrics over all steps:\n")
        f.write(avg_metrics_text)
    
    # 生成组合图：将 Chamfer 和 Hausdorff 曲线分别放在左右两个子图中
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    for label in custom_labels[:-1]:
        ax1.plot(range(1, total_steps+1), all_metrics[label]['chamfer'], label=label)
        ax2.plot(range(1, total_steps+1), all_metrics[label]['hausdorff'], label=label)
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Chamfer Distance")
    ax1.set_title("Chamfer Distance over Steps")
    ax1.legend(fontsize=8)
    ax1.grid(True)
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Hausdorff Distance")
    ax2.set_title("Hausdorff Distance over Steps")
    ax2.legend(fontsize=8)
    ax2.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "combined_metrics.png"))
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # TXT 目录
    parser.add_argument('-t1', '--txt_dir1', type=str, 
                        default='/home/david/Downloads/diffcp-main/src/greyfling', 
                        help='First directory containing TXT point clouds')
    parser.add_argument('-t2', '--txt_dir2', type=str, 
                        default='/home/david/Downloads/diffcp-main/src/greyflingpinn', 
                        help='Second directory containing TXT point clouds')
    # PCD 目录
    parser.add_argument('-p', '--pcd_dir', type=str, 
                        default='/home/david/Downloads/david_fling_new/grey_fling/ru4', 
                        help='Directory containing PCD point clouds')
    # OBJ 目录：总共5个 OBJ trajectory
    parser.add_argument('-o1', '--obj_dir1', type=str,
                        default='/home/david/Downloads/Benchmark Dataset/Diffcloud-lift/flinggrey',
                        help='First OBJ trajectory directory')
    parser.add_argument('-o2', '--obj_dir2', type=str,
                        default='/home/david/Downloads/Benchmark Dataset/PINN-stretching-diffsim/flinggreypinn',
                        help='Second OBJ trajectory directory')
    parser.add_argument('-o3', '--obj_dir3', type=str,
                        default='/home/david/Downloads/Benchmark Dataset/Physnet-lift/greyphysnetfling 1/greyphysnetfling',
                        help='Third OBJ trajectory directory')
    parser.add_argument('-o4', '--obj_dir4', type=str,
                        default='/home/david/Downloads/Benchmark Dataset/Physnet-lift/greyphysnetfling 1/greyphysnetfling',
                        help='Fourth OBJ trajectory directory')
    parser.add_argument('-o5', '--obj_dir5', type=str,
                        default='/home/david/Downloads/Benchmark Dataset/Diffcloud - wind/diffsimwindgreyfling',
                        help='Fifth OBJ trajectory directory')
    parser.add_argument('-s', '--steps', type=int, default=129, 
                        help='Total simulation steps')
    parser.add_argument('--out_dir', type=str, default='allflinggreytrytrytry', 
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
    
    # 更新 dirs 顺序：前5个为 OBJ trajectory，接下来2个 TXT，最后1个真实 PCD
    dirs = [
        (obj_pcl_dir1, 'npy'),
        (obj_pcl_dir2, 'npy'),
        (obj_pcl_dir3, 'npy'),
        (obj_pcl_dir4, 'npy'),
        (obj_pcl_dir5, 'npy'),
        (args.txt_dir1, 'txt'),
        (args.txt_dir2, 'txt'),
        (args.pcd_dir, 'pcd')
    ]
    
    run_visualization(dirs, args.out_dir, args.steps, args.points)
    print("Visualization completed.")
